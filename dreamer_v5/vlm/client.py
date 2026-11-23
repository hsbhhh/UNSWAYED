from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import AsyncVLMClient, VIEW_B1, VIEW_B2


@dataclass
class VLMRequest:
    image: np.ndarray
    env_id: int
    episode_id: int
    frame_id: int
    view_id: int
    generation_id: int


@dataclass
class VLMResult:
    patch_scores: np.ndarray
    patch_features: np.ndarray
    valid: bool
    frame_id: int
    env_id: int
    episode_id: int
    view_id: int
    generation_id: int
    request_id: int
    error: str | None
    metadata: dict[str, Any]


class MockVLMClient:
    def __init__(self, config):
        self.grid_size = int(config.external_vlm.grid_size)
        self.feature_dim = int(config.external_vlm.feature_dim)
        self.backend = "mock"

    def infer(self, requests: Iterable[VLMRequest]) -> list[VLMResult]:
        out = []
        for request in requests:
            scores = _scores_from_image(request.image, self.grid_size)
            features = _features_from_scores(scores, self.feature_dim)
            out.append(
                VLMResult(
                    patch_scores=scores.astype(np.float32),
                    patch_features=features.astype(np.float32),
                    valid=True,
                    frame_id=int(request.frame_id),
                    env_id=int(request.env_id),
                    episode_id=int(request.episode_id),
                    view_id=int(request.view_id),
                    generation_id=int(request.generation_id),
                    request_id=-1,
                    error=None,
                    metadata={"backend": "mock", "status": "ok"},
                )
            )
        return out

    def close(self) -> None:
        pass

    def stats_snapshot(self) -> dict[str, float]:
        return {
            "latest_frame_exact_match": 1.0,
            "view_exact_match": 1.0,
            "cross_view_contamination_count": 0.0,
        }


class ResidentVLMClient:
    """Formal resident VLM client wrapper.

    The training pipeline calls this class instead of manipulating
    shared-memory details directly.
    """

    def __init__(self, config, *, env_num: int):
        self.backend = "resident"
        vlm_config = {
            "clip_len": int(config.external_vlm.get("clip_len", 8)),
            "grid_shape": [int(config.external_vlm.grid_size), int(config.external_vlm.grid_size)],
            "external": {
                "shared_memory_name": config.external_vlm.shared_memory_name,
                "env_num": int(env_num),
                "input_height": int(config.external_vlm.input_height),
                "input_width": int(config.external_vlm.input_width),
                "feature_dim": int(config.external_vlm.feature_dim),
                "view_num": int(config.external_vlm.get("view_num", 2)),
                "strict_per_frame": bool(config.external_vlm.strict_per_frame),
                "strict_timeout_sec": float(config.external_vlm.strict_timeout_sec),
                "strict_poll_interval_ms": float(config.external_vlm.strict_poll_interval_ms),
                "heartbeat_timeout_sec": float(config.external_vlm.heartbeat_timeout_sec),
                "require_server": bool(config.external_vlm.require_server),
            },
            "cache": {"max_age": int(config.external_vlm.cache_max_age), "fallback": str(config.external_vlm.cache_fallback)},
            "vcfa": {"vlm_grid_size": int(config.external_vlm.grid_size), "query_dim": int(config.external_vlm.feature_dim)},
            "semantic": {"num_frames": int(config.external_vlm.get("clip_len", 8))},
            "strict_per_frame": bool(config.external_vlm.strict_per_frame),
            "online_stride": int(config.external_vlm.online_stride),
        }
        self.client = AsyncVLMClient.from_config(vlm_config, env_num=env_num)

    def infer(self, requests: Iterable[VLMRequest]) -> list[VLMResult]:
        requests = list(requests)
        if not requests:
            return []
        images = np.stack([np.asarray(req.image, dtype=np.uint8) for req in requests])
        trans = {
            "image": torch.as_tensor(images),
            "env_id": torch.as_tensor([req.env_id for req in requests], dtype=torch.int64).reshape(len(requests), 1),
            "view_id": torch.as_tensor([req.view_id for req in requests], dtype=torch.int64).reshape(len(requests), 1),
            "episode_id": torch.as_tensor([req.episode_id for req in requests], dtype=torch.int64).reshape(len(requests), 1),
            "frame_id": torch.as_tensor([req.frame_id for req in requests], dtype=torch.int64).reshape(len(requests), 1),
            "env_step": torch.as_tensor([req.frame_id for req in requests], dtype=torch.int64).reshape(len(requests), 1),
            "is_first": torch.as_tensor([req.frame_id == 0 for req in requests], dtype=torch.bool).reshape(len(requests), 1),
        }
        payload = self.client.attach_transition(trans, step=max(req.frame_id for req in requests), wait_for_current=True)
        scores = payload["vlm_patch_scores"].detach().cpu().numpy().astype(np.float32)
        features = payload["vlm_patch_features"].detach().cpu().numpy().astype(np.float32)
        valid = payload["vlm_valid"].detach().cpu().numpy().reshape(len(requests)).astype(bool)
        out = []
        for idx, req in enumerate(requests):
            validate_vlm_response_identity(req, payload, idx)
            out.append(
                VLMResult(
                    patch_scores=scores[idx],
                    patch_features=features[idx],
                    valid=bool(valid[idx]),
                    frame_id=int(req.frame_id),
                    env_id=int(req.env_id),
                    episode_id=int(req.episode_id),
                    view_id=int(req.view_id),
                    generation_id=_payload_scalar(payload, "vlm_frame_generation_id", idx),
                    request_id=_payload_scalar(payload, "vlm_request_id", idx),
                    error=None,
                    metadata={
                        "backend": "resident",
                        "status": "ok",
                        "view_exact_match": bool(_payload_scalar(payload, "vlm_view_exact_match", idx)),
                        "latest_frame_exact_match": bool(
                            _payload_scalar(payload, "vlm_latest_frame_exact_match", idx)
                        ),
                    },
                )
            )
        return out

    def close(self) -> None:
        self.client.close()

    def stats_snapshot(self) -> dict[str, float]:
        return self.client.stats_snapshot()


def create_vlm_client(config, backend: str | None = None, *, env_num: int | None = None):
    backend = str(backend or config.external_vlm.backend)
    if backend == "mock":
        return MockVLMClient(config)
    if backend == "resident":
        return ResidentVLMClient(config, env_num=int(env_num or config.env.env_num))
    raise ValueError(f"unsupported VLM backend={backend!r}")


def infer_b1_b2(client, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    b1_requests = [
        VLMRequest(
            image=record["raw_b1"],
            env_id=int(record["env_id"]),
            episode_id=int(record["episode_id"]),
            frame_id=int(record["observation_index"]),
            view_id=VIEW_B1,
            generation_id=int(record["outgoing_action_index"]),
        )
        for record in records
    ]
    b2_requests = [
        VLMRequest(
            image=record["raw_b2"],
            env_id=int(record["env_id"]),
            episode_id=int(record["episode_id"]),
            frame_id=int(record["observation_index"]),
            view_id=VIEW_B2,
            generation_id=int(record["outgoing_action_index"]),
        )
        for record in records
    ]
    b1 = client.infer(b1_requests)
    b2 = client.infer(b2_requests)
    enriched = []
    for record, res1, res2 in zip(records, b1, b2, strict=True):
        expected_identity = (
            int(record["env_id"]),
            int(record["episode_id"]),
            int(record["observation_index"]),
        )
        if (res1.env_id, res1.episode_id, res1.frame_id) != expected_identity:
            raise RuntimeError(f"B1 VLM response identity mismatch: expected={expected_identity}, got={res1}")
        if (res2.env_id, res2.episode_id, res2.frame_id) != expected_identity:
            raise RuntimeError(f"B2 VLM response identity mismatch: expected={expected_identity}, got={res2}")
        row = dict(record)
        row["patch_scores_b1"] = res1.patch_scores
        row["patch_scores_b2"] = res2.patch_scores
        row["patch_features_b1"] = res1.patch_features
        row["patch_features_b2"] = res2.patch_features
        row["vlm_pair_valid"] = bool(
            res1.valid and res2.valid and record.get("foreground_mask_valid", True)
        )
        if int(res1.view_id) != VIEW_B1 or int(res2.view_id) != VIEW_B2:
            raise RuntimeError(
                f"VLM pair response view mismatch: b1={res1.view_id}, b2={res2.view_id}"
            )
        row["vlm_backend"] = res1.metadata.get("backend", "unknown")
        enriched.append(row)
    return enriched


def infer_k_views(
    client,
    records: list[dict[str, Any]],
    *,
    num_views: int,
) -> list[dict[str, Any]]:
    """Infer all K same-state views while preserving an explicit K dimension."""

    count = int(num_views)
    if count not in {2, 4, 6}:
        raise ValueError(f"K-view VLM requires num_views in {{2,4,6}}, got {count}")
    requests = []
    for record in records:
        images = np.asarray(record["raw_b_views"], dtype=np.uint8)
        if images.ndim != 4 or images.shape[0] != count or images.shape[-1] != 3:
            raise ValueError(f"raw_b_views must be [{count},H,W,3], got {images.shape}")
        requests.extend(
            VLMRequest(
                image=images[view_id],
                env_id=int(record["env_id"]),
                episode_id=int(record["episode_id"]),
                frame_id=int(record["observation_index"]),
                view_id=int(view_id),
                generation_id=int(record["outgoing_action_index"]),
            )
            for view_id in range(count)
        )
    results = client.infer(requests)
    if len(results) != len(records) * count:
        raise RuntimeError(
            f"VLM returned {len(results)} results for {len(records)} records and K={count}"
        )
    enriched = []
    for row_index, record in enumerate(records):
        view_results = results[row_index * count : (row_index + 1) * count]
        expected_identity = (
            int(record["env_id"]),
            int(record["episode_id"]),
            int(record["observation_index"]),
        )
        for expected_view, result in enumerate(view_results):
            actual_identity = (result.env_id, result.episode_id, result.frame_id)
            if actual_identity != expected_identity or int(result.view_id) != expected_view:
                raise RuntimeError(
                    "K-view VLM response identity mismatch: "
                    f"expected={expected_identity + (expected_view,)}, "
                    f"actual={actual_identity + (int(result.view_id),)}"
                )
        row = dict(record)
        row["patch_scores"] = np.stack([result.patch_scores for result in view_results]).astype(np.float32)
        row["patch_features"] = np.stack([result.patch_features for result in view_results]).astype(np.float32)
        all_valid = bool(
            all(result.valid for result in view_results)
            and record.get("foreground_mask_valid", True)
            and record.get(
                "multi_view_valid",
                record.get("six_view_valid", record.get("four_view_valid", False)),
            )
        )
        row["vlm_views_valid"] = all_valid
        row["vlm_pair_valid"] = all_valid
        row["vlm_backend"] = view_results[0].metadata.get("backend", "unknown")
        enriched.append(row)
    return enriched


def infer_counterfactual_views(
    client,
    records: list[dict[str, Any]],
    *,
    counterfactual_k: int,
) -> list[dict[str, Any]]:
    """Compatibility alias; production uses infer_k_views."""

    return infer_k_views(client, records, num_views=counterfactual_k)


def validate_vlm_response_identity(
    request: VLMRequest,
    payload: dict[str, torch.Tensor],
    index: int,
) -> None:
    expected = {
        "env_id": int(request.env_id),
        "episode_id": int(request.episode_id),
        "frame_id": int(request.frame_id),
        "view_id": int(request.view_id),
    }
    actual = {
        "env_id": _payload_scalar(payload, "vlm_env_id", index),
        "episode_id": _payload_scalar(payload, "vlm_episode_id", index),
        "frame_id": _payload_scalar(payload, "vlm_frame_id", index),
        "view_id": _payload_scalar(payload, "vlm_view_id", index),
    }
    source = {
        "env_id": _payload_scalar(payload, "vlm_semantic_source_env_id", index),
        "episode_id": _payload_scalar(payload, "vlm_semantic_source_episode_id", index),
        "frame_id": _payload_scalar(payload, "vlm_semantic_source_frame_id", index),
        "view_id": _payload_scalar(payload, "vlm_semantic_source_view_id", index),
    }
    request_id = _payload_scalar(payload, "vlm_request_id", index)
    source_request_id = _payload_scalar(payload, "vlm_semantic_source_request_id", index)
    generation = _payload_scalar(payload, "vlm_frame_generation_id", index)
    source_generation = _payload_scalar(payload, "vlm_semantic_source_frame_generation_id", index)
    mismatches = [
        name
        for name in expected
        if actual[name] != expected[name] or source[name] != expected[name]
    ]
    if request_id != source_request_id:
        mismatches.append("request_id")
    if generation != source_generation:
        mismatches.append("generation_id")
    if not bool(_payload_scalar(payload, "vlm_latest_frame_exact_match", index)):
        mismatches.append("latest_frame_exact_match")
    if not bool(_payload_scalar(payload, "vlm_view_exact_match", index)):
        mismatches.append("view_exact_match")
    if mismatches:
        raise RuntimeError(
            "Strict VLM response identity mismatch: "
            f"fields={sorted(set(mismatches))}, expected={expected}, actual={actual}, source={source}, "
            f"request_id={request_id}, source_request_id={source_request_id}, "
            f"generation={generation}, source_generation={source_generation}"
        )


def _payload_scalar(payload: dict[str, torch.Tensor], key: str, index: int) -> int:
    if key not in payload:
        raise RuntimeError(f"Strict VLM response is missing identity field {key!r}")
    value = torch.as_tensor(payload[key]).reshape(len(torch.as_tensor(payload[key])), -1)[index, 0]
    return int(value.detach().cpu())


def _scores_from_image(image: np.ndarray, grid_size: int) -> np.ndarray:
    img = torch.as_tensor(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    gray = (0.45 * img[:, 0:1] + 0.35 * img[:, 1:2] + 0.20 * (1.0 - img[:, 2:3])).clamp(0, 1)
    pooled = F.adaptive_avg_pool2d(gray, (int(grid_size), int(grid_size))).reshape(-1)
    lo, hi = pooled.amin(), pooled.amax()
    return ((pooled - lo) / (hi - lo).clamp_min(1e-6)).numpy()


def _features_from_scores(scores: np.ndarray, feature_dim: int) -> np.ndarray:
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    count = int(scores_t.numel())
    side = int(round(count ** 0.5))
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1), (xx * yy).reshape(-1)], dim=-1)
    base = torch.cat([scores_t.reshape(-1, 1), pos, scores_t.reshape(-1, 1) * pos], dim=-1)
    repeat = (int(feature_dim) + base.shape[-1] - 1) // base.shape[-1]
    return base.repeat(1, repeat)[:, : int(feature_dim)].numpy()
