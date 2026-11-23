from __future__ import annotations

import math
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_VERSION = 6
VIEW_B1 = 0
VIEW_B2 = 1
VIEW_COUNT = 2


@dataclass(frozen=True)
class VLMSharedSpec:
    name: str
    env_num: int
    clip_len: int
    height: int
    width: int
    grid_size: int
    feature_dim: int
    view_num: int = VIEW_COUNT

    @property
    def token_count(self) -> int:
        return int(self.grid_size) * int(self.grid_size)


@dataclass(frozen=True)
class FrameSnapshot:
    env_id: int
    view_id: int
    clip: np.ndarray
    slot_view_ids: np.ndarray
    latest_frame: np.ndarray
    frame_id: int
    env_step: int
    episode_id: int
    timestamp_ns: int
    generation_id: int
    write_index: int
    consistent: bool = True


@dataclass(frozen=True)
class SemanticSnapshot:
    env_id: int
    view_id: int
    semantic_map: np.ndarray
    patch_features: np.ndarray
    generation_id: int
    source_request_id: int
    source_frame_id: int
    source_frame_generation_id: int
    source_env_step: int
    source_episode_id: int
    source_env_id: int
    source_view_id: int
    semantic_timestamp_ns: int
    inference_start_ns: int
    inference_end_ns: int
    checkpoint_hash: int
    preprocess_hash: int
    valid: bool
    consistent: bool = True


ARRAY_SPECS = {
    "frames": ("uint8", ("env_num", "view_num", "clip_len", "height", "width", 3)),
    "frame_slot_view_id": ("int8", ("env_num", "view_num", "clip_len")),
    # Seqlock-style writer state. Odd means a frame write is in progress,
    # even means readers may accept generation-before == generation-after.
    "frame_version": ("int64", ("env_num", "view_num")),
    "frame_generation_id": ("int64", ("env_num", "view_num")),
    "frame_id": ("int64", ("env_num", "view_num")),
    "frame_env_id": ("int32", ("env_num", "view_num")),
    "frame_view_id": ("int8", ("env_num", "view_num")),
    "frame_env_step": ("int64", ("env_num", "view_num")),
    "frame_timestamp_ns": ("int64", ("env_num", "view_num")),
    "frame_episode_id": ("int64", ("env_num", "view_num")),
    "frame_valid": ("uint8", ("env_num", "view_num")),
    "write_index": ("int32", ("env_num", "view_num")),
    "episode_version": ("int64", ("env_num", "view_num")),
    "env_episode_id": ("int64", ("env_num",)),
    "request_step": ("int64", ("env_num", "view_num")),
    "request_id": ("int64", ("env_num", "view_num")),
    "request_env_id": ("int32", ("env_num", "view_num")),
    "request_view_id": ("int8", ("env_num", "view_num")),
    "next_request_id": ("int64", (1,)),
    "request_frame_generation_id": ("int64", ("env_num", "view_num")),
    "request_frame_id": ("int64", ("env_num", "view_num")),
    "request_episode_id": ("int64", ("env_num", "view_num")),
    "request_timestamp_ns": ("int64", ("env_num", "view_num")),
    "request_pending": ("uint8", ("env_num", "view_num")),
    # Seqlock-style semantic writer state. Odd means tensors/metadata are
    # being published, even means committed. semantic_generation_id is the
    # per-env monotonic committed generation visible in diagnostics.
    "semantic_version": ("int64", ("env_num", "view_num")),
    "semantic_generation_id": ("int64", ("env_num", "view_num")),
    "semantic_source_request_id": ("int64", ("env_num", "view_num")),
    "semantic_source_step": ("int64", ("env_num", "view_num")),
    "semantic_source_frame_id": ("int64", ("env_num", "view_num")),
    "semantic_source_frame_generation_id": ("int64", ("env_num", "view_num")),
    "semantic_source_env_step": ("int64", ("env_num", "view_num")),
    "semantic_source_episode_id": ("int64", ("env_num", "view_num")),
    "semantic_source_env_id": ("int32", ("env_num", "view_num")),
    "semantic_source_view_id": ("int8", ("env_num", "view_num")),
    "semantic_timestamp_ns": ("int64", ("env_num", "view_num")),
    "vlm_inference_start_ns": ("int64", ("env_num", "view_num")),
    "vlm_inference_end_ns": ("int64", ("env_num", "view_num")),
    "vlm_checkpoint_hash": ("int64", ("env_num", "view_num")),
    "vlm_preprocess_hash": ("int64", ("env_num", "view_num")),
    "semantic_episode_version": ("int64", ("env_num", "view_num")),
    "semantic_valid": ("uint8", ("env_num", "view_num")),
    "semantic_map": ("float16", ("env_num", "view_num", "grid_size", "grid_size")),
    "patch_features": ("float16", ("env_num", "view_num", "grid_size", "grid_size", "feature_dim")),
    "completed_per_env": ("int64", ("env_num", "view_num")),
    "last_completed_frame_id": ("int64", ("env_num", "view_num")),
    "last_completed_episode_id": ("int64", ("env_num", "view_num")),
    "service_start_ns": ("int64", (1,)),
    "heartbeat_ns": ("int64", (1,)),
    "protocol_spec": ("int64", (8,)),
    "stats": ("int64", (48,)),
}


STAT_SUBMITTED = 0
STAT_COMPLETED = 1
STAT_DROPPED = 2
STAT_CACHE_HIT = 3
STAT_FALLBACK = 4
STAT_STALE = 5
STAT_SERVER_LOOPS = 6
STAT_PROTOCOL_VERSION = 7
STAT_FRAME_WRITES = 8
STAT_SEMANTIC_PUBLISH = 9
STAT_INFERENCE = 10
STAT_INFERENCE_FAILURE = 11
STAT_DUPLICATE_FRAME_SKIP = 12
STAT_CONSISTENT_READ_FAIL = 13
STAT_DROPPED_INTERMEDIATE_FRAMES = 14
STAT_EXACT_FRAME_MATCH = 15
STAT_FRAME_GAP = 16
STAT_OUT_OF_ORDER = 17
STAT_CROSS_ENV_MISMATCH = 18
STAT_CROSS_EPISODE_MISMATCH = 19
STAT_FUTURE_FRAME_VIOLATION = 20
STAT_STALE_RESULT_REJECTED = 21
STAT_OLD_EPISODE_RESULT_REJECTED = 22
STAT_INFERENCE_BATCH_TOTAL = 23
STAT_INFERENCE_BATCH_COUNT = 24
STAT_INFERENCE_BATCH_MAX = 25
STAT_REQUEST_WAIT_MS_TOTAL = 26
STAT_REQUEST_WAIT_MS_MAX = 27
STAT_REQUEST_QUEUE_DEPTH_MAX = 28
STAT_EXACT_VIEW_MATCH = 29
STAT_CROSS_VIEW_MISMATCH = 30
STAT_CROSS_VIEW_CONTAMINATION = 31


class VLMSharedMemory:
    def __init__(self, spec: VLMSharedSpec, *, create: bool = False):
        self.spec = spec
        self.create = bool(create)
        self.blocks: dict[str, shared_memory.SharedMemory] = {}
        self.arrays: dict[str, np.ndarray] = {}
        for key, (dtype_name, shape_names) in ARRAY_SPECS.items():
            dtype = np.dtype(dtype_name)
            shape = tuple(_resolve_dim(spec, dim) for dim in shape_names)
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            shm_name = f"{spec.name}_{key}"
            block = _create_or_attach(shm_name, size, create=self.create)
            self.blocks[key] = block
            actual_size = len(block.buf)
            if actual_size < size:
                self.close(unlink=False)
                raise RuntimeError(
                    "VLM shared-memory shape mismatch while attaching "
                    f"{shm_name}: key={key}, dtype={dtype}, expected_shape={shape}, "
                    f"expected_bytes={size}, actual_bytes={actual_size}, spec={spec}, "
                    "generation=unavailable_before_successful_attach. "
                    "Start the VLM server and Dreamer/eval with the same env_num, clip_len, "
                    "input size, grid_size, feature_dim, and protocol version."
                )
            self.arrays[key] = np.ndarray(shape, dtype=dtype, buffer=block.buf)
            if self.create:
                self.arrays[key].fill(0)
        if self.create:
            self.arrays["protocol_spec"][:] = _shared_spec_values(spec)
            self.arrays["stats"][STAT_PROTOCOL_VERSION] = PROTOCOL_VERSION
            self.arrays["service_start_ns"][0] = int(time.time_ns())
            self.arrays["frame_id"].fill(-1)
            self.arrays["frame_env_id"].fill(-1)
            self.arrays["frame_view_id"].fill(-1)
            self.arrays["frame_slot_view_id"].fill(-1)
            self.arrays["frame_env_step"].fill(-1)
            self.arrays["frame_episode_id"].fill(-1)
            self.arrays["env_episode_id"].fill(-1)
            self.arrays["semantic_source_step"].fill(-1)
            self.arrays["semantic_source_request_id"].fill(-1)
            self.arrays["semantic_source_frame_id"].fill(-1)
            self.arrays["semantic_source_frame_generation_id"].fill(-1)
            self.arrays["semantic_source_env_step"].fill(-1)
            self.arrays["semantic_source_episode_id"].fill(-1)
            self.arrays["semantic_source_env_id"].fill(-1)
            self.arrays["semantic_source_view_id"].fill(-1)
            self.arrays["request_id"].fill(-1)
            self.arrays["request_env_id"].fill(-1)
            self.arrays["request_view_id"].fill(-1)
            self.arrays["request_frame_generation_id"].fill(-1)
            self.arrays["request_frame_id"].fill(-1)
            self.arrays["request_episode_id"].fill(-1)
            self.arrays["last_completed_frame_id"].fill(-1)
            self.arrays["last_completed_episode_id"].fill(-1)
        else:
            expected_spec = _shared_spec_values(spec)
            actual_spec = self.arrays["protocol_spec"].astype(np.int64).copy()
            if not np.array_equal(actual_spec, expected_spec):
                generation = "unavailable"
                if "semantic_generation_id" in self.arrays:
                    generation = int(self.arrays["semantic_generation_id"].max(initial=0))
                self.close(unlink=False)
                raise RuntimeError(
                    "VLM shared-memory protocol spec mismatch while attaching "
                    f"{spec.name}: expected={_shared_spec_dict(expected_spec)}, "
                    f"actual={_shared_spec_dict(actual_spec)}, generation={generation}. "
                    "Start the VLM server and Dreamer/eval with the same env_num, clip_len, "
                    "height, width, grid_size, feature_dim, and protocol."
                )
            self.arrays["semantic_source_env_id"].fill(-1)
            self.arrays["semantic_source_view_id"].fill(-1)
            self.arrays["semantic_episode_version"].fill(-1)
            self.arrays["request_id"].fill(-1)
            self.arrays["request_env_id"].fill(-1)
            self.arrays["request_view_id"].fill(-1)
            self.arrays["request_frame_generation_id"].fill(-1)
            self.arrays["request_frame_id"].fill(-1)
            self.arrays["request_episode_id"].fill(-1)
            self.arrays["request_timestamp_ns"].fill(0)
            self.arrays["last_completed_frame_id"].fill(-1)
            self.arrays["last_completed_episode_id"].fill(-1)

    def close(self, *, unlink: bool = False) -> None:
        for block in self.blocks.values():
            block.close()
            if unlink:
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass

    def unlink(self) -> None:
        self.close(unlink=True)

    def stable_clip(self, env_id: int, view_id: int) -> tuple[np.ndarray | None, int, int, int]:
        snap = self.read_consistent_frame_snapshot(env_id, view_id)
        if snap is None:
            version = int(self.arrays["frame_version"][int(env_id), _validate_view_id(view_id, self.spec)])
            return None, version, -1, -1
        return snap.clip, snap.generation_id, snap.write_index, snap.episode_id

    def read_consistent_frame_snapshot(
        self, env_id: int, view_id: int, *, max_retries: int = 10
    ) -> FrameSnapshot | None:
        env_id = int(env_id)
        view_id = _validate_view_id(view_id, self.spec)
        for _ in range(max(1, int(max_retries))):
            before = int(self.arrays["frame_version"][env_id, view_id])
            if before <= 0 or before % 2 or not bool(self.arrays["frame_valid"][env_id, view_id]):
                continue
            write_index = int(self.arrays["write_index"][env_id, view_id])
            episode = int(self.arrays["episode_version"][env_id, view_id])
            clip = np.concatenate(
                [
                    self.arrays["frames"][env_id, view_id, write_index:],
                    self.arrays["frames"][env_id, view_id, :write_index],
                ],
                axis=0,
            ).copy()
            slot_view_ids = np.concatenate(
                [
                    self.arrays["frame_slot_view_id"][env_id, view_id, write_index:],
                    self.arrays["frame_slot_view_id"][env_id, view_id, :write_index],
                ]
            ).copy()
            frame_id = int(self.arrays["frame_id"][env_id, view_id])
            env_step = int(self.arrays["frame_env_step"][env_id, view_id])
            timestamp_ns = int(self.arrays["frame_timestamp_ns"][env_id, view_id])
            frame_episode = int(self.arrays["frame_episode_id"][env_id, view_id])
            frame_view = int(self.arrays["frame_view_id"][env_id, view_id])
            frame_env = int(self.arrays["frame_env_id"][env_id, view_id])
            generation = int(self.arrays["frame_generation_id"][env_id, view_id])
            after = int(self.arrays["frame_version"][env_id, view_id])
            if (
                before == after
                and after % 2 == 0
                and generation > 0
                and frame_episode == episode
                and frame_env == env_id
                and frame_view == view_id
            ):
                return FrameSnapshot(
                    env_id=env_id,
                    view_id=view_id,
                    clip=clip,
                    slot_view_ids=slot_view_ids,
                    latest_frame=np.asarray(clip[-1], dtype=np.uint8).copy(),
                    frame_id=frame_id,
                    env_step=env_step,
                    episode_id=episode,
                    timestamp_ns=timestamp_ns,
                    generation_id=generation,
                    write_index=write_index,
                    consistent=True,
                )
        self.arrays["stats"][STAT_CONSISTENT_READ_FAIL] += 1
        return None

    def read_consistent_semantic_snapshot(
        self, env_id: int, view_id: int, *, max_retries: int = 10
    ) -> SemanticSnapshot | None:
        env_id = int(env_id)
        view_id = _validate_view_id(view_id, self.spec)
        for _ in range(max(1, int(max_retries))):
            before = int(self.arrays["semantic_version"][env_id, view_id])
            if before <= 0 or before % 2 or not bool(self.arrays["semantic_valid"][env_id, view_id]):
                continue
            semantic = self.arrays["semantic_map"][env_id, view_id].astype(np.float32).copy()
            features = self.arrays["patch_features"][env_id, view_id].astype(np.float32).copy()
            generation = int(self.arrays["semantic_generation_id"][env_id, view_id])
            source_request_id = int(self.arrays["semantic_source_request_id"][env_id, view_id])
            source_frame_id = int(self.arrays["semantic_source_frame_id"][env_id, view_id])
            source_frame_generation = int(self.arrays["semantic_source_frame_generation_id"][env_id, view_id])
            source_env_step = int(self.arrays["semantic_source_env_step"][env_id, view_id])
            source_episode_id = int(self.arrays["semantic_source_episode_id"][env_id, view_id])
            source_env_id = int(self.arrays["semantic_source_env_id"][env_id, view_id])
            source_view_id = int(self.arrays["semantic_source_view_id"][env_id, view_id])
            semantic_timestamp_ns = int(self.arrays["semantic_timestamp_ns"][env_id, view_id])
            inference_start_ns = int(self.arrays["vlm_inference_start_ns"][env_id, view_id])
            inference_end_ns = int(self.arrays["vlm_inference_end_ns"][env_id, view_id])
            checkpoint_hash = int(self.arrays["vlm_checkpoint_hash"][env_id, view_id])
            preprocess_hash = int(self.arrays["vlm_preprocess_hash"][env_id, view_id])
            after = int(self.arrays["semantic_version"][env_id, view_id])
            if before == after and after % 2 == 0 and generation > 0:
                return SemanticSnapshot(
                    env_id=env_id,
                    view_id=view_id,
                    semantic_map=semantic,
                    patch_features=features,
                    generation_id=generation,
                    source_request_id=source_request_id,
                    source_frame_id=source_frame_id,
                    source_frame_generation_id=source_frame_generation,
                    source_env_step=source_env_step,
                    source_episode_id=source_episode_id,
                    source_env_id=source_env_id,
                    source_view_id=source_view_id,
                    semantic_timestamp_ns=semantic_timestamp_ns,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                    checkpoint_hash=checkpoint_hash,
                    preprocess_hash=preprocess_hash,
                    valid=True,
                    consistent=True,
                )
        self.arrays["stats"][STAT_CONSISTENT_READ_FAIL] += 1
        return None


class AsyncVLMClient:
    def __init__(self, spec: VLMSharedSpec, config: dict[str, Any], *, env_num: int):
        self.spec = spec
        self.config = dict(config or {})
        self.env_num = int(env_num)
        self.shared = VLMSharedMemory(spec, create=False)
        protocol_version = int(self.shared.arrays["stats"][STAT_PROTOCOL_VERSION])
        if protocol_version != PROTOCOL_VERSION:
            self.shared.close(unlink=False)
            raise RuntimeError(
                f"VLM shared-memory protocol mismatch: client={PROTOCOL_VERSION}, server={protocol_version}"
            )
        self.online_stride = max(1, int(self.config.get("online_stride", 8)))
        cache_cfg = dict(self.config.get("cache", {}) or {})
        external_cfg = dict(self.config.get("external", {}) or {})
        self.strict_per_frame = bool(
            self.config.get("strict_per_frame", False) or external_cfg.get("strict_per_frame", False)
        )
        if self.strict_per_frame:
            self.online_stride = 1
        self.max_age = int(cache_cfg.get("max_age", max(2 * self.online_stride, 16)))
        self.fallback = str(cache_cfg.get("fallback", "identity"))
        if self.strict_per_frame:
            self.max_age = 0
            self.fallback = "error"
        self.require_server = bool(external_cfg.get("require_server", True))
        self.heartbeat_timeout_sec = float(external_cfg.get("heartbeat_timeout_sec", 30))
        self.strict_timeout_sec = float(external_cfg.get("strict_timeout_sec", external_cfg.get("request_timeout_sec", 300)))
        self.strict_poll_interval_sec = float(external_cfg.get("strict_poll_interval_ms", 2.0)) / 1000.0
        self._tick = 0
        self._last_semantic_version = np.zeros((self.env_num, self.spec.view_num), dtype=np.int64) - 1
        self._last_valid = {
            "semantic": np.ones(
                (self.env_num, self.spec.view_num, self.spec.grid_size, self.spec.grid_size), dtype=np.float32
            ),
            "features": np.zeros(
                (self.env_num, self.spec.view_num, self.spec.token_count, self.spec.feature_dim), dtype=np.float32
            ),
        }

    @classmethod
    def from_config(cls, vlm_config: dict[str, Any], *, env_num: int) -> "AsyncVLMClient":
        external = dict(vlm_config.get("external", {}) or {})
        cache = dict(vlm_config.get("cache", {}) or {})
        vcfa = dict(vlm_config.get("vcfa", {}) or {})
        grid = int(vcfa.get("vlm_grid_size", vlm_config.get("grid_shape", [8, 8])[0]))
        feature_dim = int(vcfa.get("query_dim", external.get("feature_dim", 64)))
        spec = VLMSharedSpec(
            name=str(external.get("shared_memory_name", "walker_vlm_service")),
            env_num=int(external.get("env_num", env_num)),
            clip_len=int(vlm_config.get("clip_len", vlm_config.get("semantic", {}).get("num_frames", 8))),
            height=int(external.get("input_height", 64)),
            width=int(external.get("input_width", 64)),
            grid_size=grid,
            feature_dim=feature_dim,
            view_num=int(external.get("view_num", VIEW_COUNT)),
        )
        merged = dict(vlm_config)
        merged["cache"] = cache
        return cls(spec, merged, env_num=env_num)

    def attach_transition(
        self,
        trans,
        *,
        step: int,
        wait_for_current: bool = False,
        timeout_sec: float = 0.0,
        poll_interval_sec: float = 0.01,
    ):
        image = _to_numpy_image(trans["image"])
        bsz = int(image.shape[0])
        env_ids = _to_numpy_int(trans.get("env_id"), bsz, default=np.arange(bsz, dtype=np.int64))
        view_ids = _to_numpy_int(trans.get("view_id"), bsz, required=True)
        episode_ids = _to_numpy_int(trans.get("episode_id"), bsz, required=True)
        frame_ids = _to_numpy_int(trans.get("frame_id"), bsz, required=True)
        env_steps = _to_numpy_int(trans.get("env_step"), bsz, default=frame_ids)
        _validate_stream_batch(self.spec, env_ids, view_ids)
        # Use vector-environment ticks for VLM freshness. The trainer's `step`
        # is global env frames and jumps by env_num * action_repeat each loop,
        # which would make a cache with max_age=16 expire immediately at env_num=32.
        cache_step = int(self._tick)
        is_first = _to_numpy_bool(trans.get("is_first"), bsz)
        self._write_frames(
            image,
            is_first,
            env_ids=env_ids,
            view_ids=view_ids,
            episode_ids=episode_ids,
            frame_ids=frame_ids,
            env_steps=env_steps,
        )
        target_generations = [
            int(self.shared.arrays["frame_generation_id"][env_id, view_id])
            for env_id, view_id in zip(env_ids, view_ids, strict=True)
        ]
        target_episodes = [
            int(self.shared.arrays["episode_version"][env_id, view_id])
            for env_id, view_id in zip(env_ids, view_ids, strict=True)
        ]
        target_frame_ids = [
            int(self.shared.arrays["frame_id"][env_id, view_id])
            for env_id, view_id in zip(env_ids, view_ids, strict=True)
        ]
        scheduled, target_request_ids = self._submit_round_robin(env_ids, view_ids, cache_step)
        wait_success = False
        wait_elapsed_ms = 0.0
        should_wait = bool(wait_for_current or self.strict_per_frame)
        if should_wait:
            wait_timeout = float(timeout_sec)
            wait_poll = float(poll_interval_sec)
            if self.strict_per_frame:
                wait_timeout = self.strict_timeout_sec if wait_timeout <= 0 else wait_timeout
                wait_poll = self.strict_poll_interval_sec if wait_poll <= 0 else wait_poll
            t0 = time.perf_counter()
            wait_success = self.wait_for_current_semantics(
                bsz,
                env_ids=env_ids,
                view_ids=view_ids,
                target_request_ids=target_request_ids,
                target_generations=target_generations,
                target_episodes=target_episodes,
                target_frame_ids=target_frame_ids,
                timeout_sec=float(wait_timeout),
                poll_interval_sec=float(wait_poll),
            )
            wait_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if self.strict_per_frame and not wait_success:
                raise TimeoutError(
                    "Timed out waiting for exact VLM results in strict_per_frame mode: "
                    f"batch={bsz}, timeout_sec={float(wait_timeout):.3f}, "
                    f"env_ids={env_ids.tolist()}, view_ids={view_ids.tolist()}, "
                    f"request_ids={target_request_ids}, frame_ids={target_frame_ids}, episodes={target_episodes}"
                )
        payload = self._read_payload(
            bsz,
            cache_step,
            env_ids=env_ids,
            view_ids=view_ids,
            target_request_ids=target_request_ids,
            target_frame_ids=target_frame_ids,
            target_generations=target_generations,
            target_episodes=target_episodes,
        )
        heartbeat_age = heartbeat_age_sec(self.shared)
        if self.require_server and heartbeat_age > self.heartbeat_timeout_sec:
            raise RuntimeError(
                f"VLM server heartbeat is stale ({heartbeat_age:.1f}s > {self.heartbeat_timeout_sec:.1f}s). "
                "Restart the VLM server or set model.vlm.external.require_server=false to allow fallback-only training."
            )
        for key, value in payload.items():
            trans[key] = value
        trans["vlm_global_env_step"] = torch.full((bsz, 1), int(step), dtype=torch.int64)
        trans["vlm_env_id"] = torch.as_tensor(env_ids, dtype=torch.int64).reshape(bsz, 1)
        trans["vlm_view_id"] = torch.as_tensor(view_ids, dtype=torch.int64).reshape(bsz, 1)
        trans["vlm_requests_scheduled"] = torch.full((bsz, 1), int(scheduled), dtype=torch.int32)
        trans["vlm_request_id"] = torch.as_tensor(target_request_ids, dtype=torch.int64).reshape(bsz, 1)
        if should_wait:
            trans["vlm_sync_wait_success"] = torch.full((bsz, 1), bool(wait_success), dtype=torch.bool)
            trans["vlm_sync_wait_ms"] = torch.full((bsz, 1), float(wait_elapsed_ms), dtype=torch.float32)
        self._tick += 1
        return trans

    def wait_for_current_semantics(
        self,
        batch: int,
        *,
        env_ids: np.ndarray | list[int] | None = None,
        view_ids: np.ndarray | list[int] | None = None,
        target_request_ids: list[int] | None = None,
        target_generations: list[int] | None = None,
        target_episodes: list[int] | None = None,
        target_frame_ids: list[int] | None = None,
        timeout_sec: float = 0.0,
        poll_interval_sec: float = 0.01,
    ) -> bool:
        batch = int(batch)
        env_ids = np.arange(batch, dtype=np.int64) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        view_ids = np.zeros((batch,), dtype=np.int64) if view_ids is None else np.asarray(view_ids, dtype=np.int64)
        _validate_stream_batch(self.spec, env_ids, view_ids)
        if target_request_ids is None:
            target_request_ids = [
                int(self.shared.arrays["request_id"][env_id, view_id])
                for env_id, view_id in zip(env_ids, view_ids, strict=True)
            ]
        if target_generations is None:
            target_generations = [
                int(self.shared.arrays["frame_generation_id"][env_id, view_id])
                for env_id, view_id in zip(env_ids, view_ids, strict=True)
            ]
        if target_episodes is None:
            target_episodes = [
                int(self.shared.arrays["episode_version"][env_id, view_id])
                for env_id, view_id in zip(env_ids, view_ids, strict=True)
            ]
        if target_frame_ids is None:
            target_frame_ids = [
                int(self.shared.arrays["frame_id"][env_id, view_id])
                for env_id, view_id in zip(env_ids, view_ids, strict=True)
            ]
        deadline = time.perf_counter() + max(0.0, float(timeout_sec))
        poll = max(0.001, float(poll_interval_sec))
        while True:
            aligned = True
            for index, (env_id, view_id) in enumerate(zip(env_ids, view_ids, strict=True)):
                sem = self.shared.read_consistent_semantic_snapshot(env_id, view_id, max_retries=3)
                if sem is None:
                    aligned = False
                    break
                if (
                    int(sem.source_env_id) != int(env_id)
                    or int(sem.source_view_id) != int(view_id)
                    or int(sem.source_request_id) != int(target_request_ids[index])
                    or int(sem.source_frame_generation_id) != int(target_generations[index])
                    or int(sem.source_frame_id) != int(target_frame_ids[index])
                    or int(sem.source_episode_id) != int(target_episodes[index])
                ):
                    aligned = False
                    break
            if aligned:
                return True
            heartbeat_age = heartbeat_age_sec(self.shared)
            if self.require_server and heartbeat_age > self.heartbeat_timeout_sec:
                raise RuntimeError(
                    f"VLM server heartbeat is stale while waiting for exact semantics: "
                    f"{heartbeat_age:.1f}s > {self.heartbeat_timeout_sec:.1f}s"
                )
            if time.perf_counter() >= deadline:
                return False
            time.sleep(min(poll, max(0.0, deadline - time.perf_counter())))

    def stats_snapshot(self) -> dict[str, float]:
        stats = self.shared.arrays["stats"].astype(np.float64).copy()
        completed = max(float(stats[STAT_COMPLETED]), 1.0)
        pending = self.shared.arrays["request_pending"].astype(np.int64)
        completed_per_env = self.shared.arrays["completed_per_env"].astype(np.float64).copy()
        batch_count = max(float(stats[STAT_INFERENCE_BATCH_COUNT]), 1.0)
        submitted = float(stats[STAT_SUBMITTED])
        completed_raw = float(stats[STAT_COMPLETED])
        inflight = float(pending.sum())
        return {
            "requests_submitted": submitted,
            "requests_completed": completed_raw,
            "requests_inflight": inflight,
            "requests_dropped": float(stats[STAT_DROPPED]),
            "cache_hit_rate": float(stats[STAT_CACHE_HIT]) / max(float(stats[STAT_CACHE_HIT] + stats[STAT_FALLBACK]), 1.0),
            "fallback_count": float(stats[STAT_FALLBACK]),
            "stale_count": float(stats[STAT_STALE]),
            "server_loops": float(stats[STAT_SERVER_LOOPS]),
            "completed_per_loop": float(stats[STAT_COMPLETED]) / max(float(stats[STAT_SERVER_LOOPS]), 1.0),
            "frame_writes": float(stats[STAT_FRAME_WRITES]),
            "semantic_publish": float(stats[STAT_SEMANTIC_PUBLISH]),
            "inference_count": float(stats[STAT_INFERENCE]),
            "inference_failure_count": float(stats[STAT_INFERENCE_FAILURE]),
            "duplicate_frame_skip_count": float(stats[STAT_DUPLICATE_FRAME_SKIP]),
            "consistent_read_fail_count": float(stats[STAT_CONSISTENT_READ_FAIL]),
            "dropped_intermediate_frames": float(stats[STAT_DROPPED_INTERMEDIATE_FRAMES]),
            "exact_frame_match_fraction": float(stats[STAT_EXACT_FRAME_MATCH]) / completed,
            "latest_frame_exact_match": float(stats[STAT_EXACT_FRAME_MATCH]) / completed,
            "view_exact_match": float(stats[STAT_EXACT_VIEW_MATCH]) / completed,
            "cross_view_mismatch_count": float(stats[STAT_CROSS_VIEW_MISMATCH]),
            "cross_view_contamination_count": float(stats[STAT_CROSS_VIEW_CONTAMINATION]),
            "frame_gap_count": float(stats[STAT_FRAME_GAP]),
            "out_of_order_count": float(stats[STAT_OUT_OF_ORDER]),
            "cross_env_mismatch_count": float(stats[STAT_CROSS_ENV_MISMATCH]),
            "cross_episode_mismatch_count": float(stats[STAT_CROSS_EPISODE_MISMATCH]),
            "future_frame_violation_count": float(stats[STAT_FUTURE_FRAME_VIOLATION]),
            "request_queue_depth": inflight,
            "request_queue_depth_max": float(stats[STAT_REQUEST_QUEUE_DEPTH_MAX]),
            "oldest_request_wait_ms": _oldest_pending_wait_ms(self.shared),
            "inference_batch_size_mean": float(stats[STAT_INFERENCE_BATCH_TOTAL]) / batch_count,
            "inference_batch_size_max": float(stats[STAT_INFERENCE_BATCH_MAX]),
            "vlm_frames_per_second": completed_raw / max(_service_uptime_sec(self.shared), 1e-6),
            "per_env_inflight_max": float(pending.max()) if pending.size else 0.0,
            "per_env_completed_min": float(completed_per_env.min()) if completed_per_env.size else 0.0,
            "per_env_completed_max": float(completed_per_env.max()) if completed_per_env.size else 0.0,
            "stale_result_rejected": float(stats[STAT_STALE_RESULT_REJECTED]),
            "old_episode_result_rejected": float(stats[STAT_OLD_EPISODE_RESULT_REJECTED]),
            "request_wait_ms_mean": float(stats[STAT_REQUEST_WAIT_MS_TOTAL]) / completed,
            "request_wait_ms_max": float(stats[STAT_REQUEST_WAIT_MS_MAX]),
            "submitted_completed_inflight_gap": submitted - completed_raw - inflight,
        }

    def close(self) -> None:
        self.shared.close(unlink=False)

    def _write_frames(
        self,
        image: np.ndarray,
        is_first: np.ndarray,
        *,
        env_ids: np.ndarray,
        view_ids: np.ndarray,
        episode_ids: np.ndarray,
        frame_ids: np.ndarray,
        env_steps: np.ndarray,
    ) -> None:
        arrays = self.shared.arrays
        reset_envs: set[int] = set()
        for index, env_id_raw in enumerate(env_ids):
            env_id = int(env_id_raw)
            requested_episode = int(episode_ids[index])
            current_episode = int(arrays["env_episode_id"][env_id])
            if current_episode != requested_episode:
                if current_episode >= 0 and not bool(is_first[index]):
                    raise RuntimeError(
                        "VLM episode changed without is_first: "
                        f"env_id={env_id}, current_episode={current_episode}, requested_episode={requested_episode}"
                    )
                reset_envs.add(env_id)
        for env_id in sorted(reset_envs):
            pending_views = np.where(arrays["request_pending"][env_id] > 0)[0].tolist()
            for view_id in pending_views:
                self._wait_for_request_slot(env_id, int(view_id))
            requested = {
                int(episode_ids[index]) for index, candidate in enumerate(env_ids) if int(candidate) == env_id
            }
            if len(requested) != 1:
                raise RuntimeError(f"Conflicting episode ids in one VLM batch for env_id={env_id}: {sorted(requested)}")
            self._reset_env_streams(env_id, requested.pop())

        for index, (env_id_raw, view_id_raw) in enumerate(zip(env_ids, view_ids, strict=True)):
            env_id = int(env_id_raw)
            view_id = _validate_view_id(view_id_raw, self.spec)
            episode = int(episode_ids[index])
            requested_frame_id = int(frame_ids[index])
            if int(arrays["env_episode_id"][env_id]) != episode:
                raise RuntimeError(
                    f"VLM env episode identity mismatch env_id={env_id}, requested={episode}, "
                    f"shared={int(arrays['env_episode_id'][env_id])}"
                )
            version = int(arrays["frame_version"][env_id, view_id])
            if version % 2:
                version += 1
            arrays["frame_version"][env_id, view_id] = version + 1
            frame = _resize_or_check_image(image[index], self.spec.height, self.spec.width)
            timestamp_ns = int(time.time_ns())
            if not bool(arrays["frame_valid"][env_id, view_id]):
                arrays["frames"][env_id, view_id, ...] = frame[None]
                arrays["frame_slot_view_id"][env_id, view_id, ...] = view_id
                arrays["write_index"][env_id, view_id] = 0
            else:
                previous_frame_id = int(arrays["frame_id"][env_id, view_id])
                if self.strict_per_frame and requested_frame_id != previous_frame_id + 1:
                    arrays["frame_version"][env_id, view_id] = version + 2
                    raise RuntimeError(
                        "Strict VLM per-view frame sequence mismatch: "
                        f"env_id={env_id}, view_id={view_id}, previous={previous_frame_id}, "
                        f"requested={requested_frame_id}"
                    )
                write = int(arrays["write_index"][env_id, view_id])
                arrays["frames"][env_id, view_id, write] = frame
                arrays["frame_slot_view_id"][env_id, view_id, write] = view_id
                arrays["write_index"][env_id, view_id] = (write + 1) % self.spec.clip_len
            arrays["episode_version"][env_id, view_id] = episode
            arrays["frame_id"][env_id, view_id] = requested_frame_id
            arrays["frame_env_id"][env_id, view_id] = env_id
            arrays["frame_view_id"][env_id, view_id] = view_id
            arrays["frame_env_step"][env_id, view_id] = int(env_steps[index])
            arrays["frame_episode_id"][env_id, view_id] = episode
            arrays["frame_timestamp_ns"][env_id, view_id] = timestamp_ns
            arrays["frame_generation_id"][env_id, view_id] += 1
            arrays["frame_valid"][env_id, view_id] = 1
            arrays["stats"][STAT_FRAME_WRITES] += 1
            arrays["frame_version"][env_id, view_id] = version + 2

    def _reset_env_streams(self, env_id: int, episode_id: int) -> None:
        arrays = self.shared.arrays
        env_id = int(env_id)
        for view_id in range(self.spec.view_num):
            frame_version = int(arrays["frame_version"][env_id, view_id])
            if frame_version % 2:
                frame_version += 1
            arrays["frame_version"][env_id, view_id] = frame_version + 1
            semantic_version = int(arrays["semantic_version"][env_id, view_id])
            if semantic_version % 2:
                semantic_version += 1
            arrays["semantic_version"][env_id, view_id] = semantic_version + 1
            arrays["frames"][env_id, view_id].fill(0)
            arrays["frame_slot_view_id"][env_id, view_id].fill(-1)
            arrays["write_index"][env_id, view_id] = 0
            arrays["frame_valid"][env_id, view_id] = 0
            arrays["frame_id"][env_id, view_id] = -1
            arrays["frame_env_id"][env_id, view_id] = env_id
            arrays["frame_view_id"][env_id, view_id] = view_id
            arrays["frame_episode_id"][env_id, view_id] = episode_id
            arrays["episode_version"][env_id, view_id] = episode_id
            arrays["semantic_valid"][env_id, view_id] = 0
            arrays["semantic_episode_version"][env_id, view_id] = -1
            arrays["semantic_source_episode_id"][env_id, view_id] = -1
            arrays["semantic_source_env_id"][env_id, view_id] = -1
            arrays["semantic_source_view_id"][env_id, view_id] = -1
            arrays["request_pending"][env_id, view_id] = 0
            arrays["request_id"][env_id, view_id] = -1
            arrays["frame_version"][env_id, view_id] = frame_version + 2
            arrays["semantic_version"][env_id, view_id] = semantic_version + 2
        arrays["env_episode_id"][env_id] = int(episode_id)

    def _submit_round_robin(
        self, env_ids: np.ndarray, view_ids: np.ndarray, step: int
    ) -> tuple[int, list[int]]:
        arrays = self.shared.arrays
        phase = self._tick % self.online_stride
        count = 0
        request_ids = [-1 for _ in range(len(env_ids))]
        for index, (env_id_raw, view_id_raw) in enumerate(zip(env_ids, view_ids, strict=True)):
            env_id = int(env_id_raw)
            view_id = _validate_view_id(view_id_raw, self.spec)
            stream_index = env_id * self.spec.view_num + view_id
            high_priority = arrays["semantic_valid"][env_id, view_id] == 0
            if not self.strict_per_frame and (stream_index % self.online_stride) != phase and not high_priority:
                continue
            if arrays["request_pending"][env_id, view_id]:
                if self.strict_per_frame:
                    self._wait_for_request_slot(env_id, view_id)
                else:
                    arrays["stats"][STAT_DROPPED] += 1
            request_id = int(arrays["next_request_id"][0])
            arrays["next_request_id"][0] = request_id + 1
            request_ids[index] = request_id
            arrays["request_step"][env_id, view_id] = step
            arrays["request_id"][env_id, view_id] = request_id
            arrays["request_env_id"][env_id, view_id] = env_id
            arrays["request_view_id"][env_id, view_id] = view_id
            arrays["request_frame_generation_id"][env_id, view_id] = int(
                arrays["frame_generation_id"][env_id, view_id]
            )
            arrays["request_frame_id"][env_id, view_id] = int(arrays["frame_id"][env_id, view_id])
            arrays["request_episode_id"][env_id, view_id] = int(arrays["episode_version"][env_id, view_id])
            arrays["request_timestamp_ns"][env_id, view_id] = int(time.time_ns())
            arrays["request_pending"][env_id, view_id] = 1
            arrays["stats"][STAT_SUBMITTED] += 1
            depth = int(arrays["request_pending"][: self.spec.env_num, : self.spec.view_num].sum())
            arrays["stats"][STAT_REQUEST_QUEUE_DEPTH_MAX] = max(
                int(arrays["stats"][STAT_REQUEST_QUEUE_DEPTH_MAX]), depth
            )
            count += 1
        return count, request_ids

    def _wait_for_request_slot(self, env_id: int, view_id: int) -> None:
        arrays = self.shared.arrays
        view_id = _validate_view_id(view_id, self.spec)
        deadline = time.perf_counter() + max(0.001, float(self.strict_timeout_sec))
        poll = max(0.001, float(self.strict_poll_interval_sec))
        while bool(arrays["request_pending"][int(env_id), view_id]):
            heartbeat_age = heartbeat_age_sec(self.shared)
            if self.require_server and heartbeat_age > self.heartbeat_timeout_sec:
                raise RuntimeError(
                    f"VLM server heartbeat is stale while waiting for request slot "
                    f"env_id={env_id}, view_id={view_id}: "
                    f"{heartbeat_age:.1f}s > {self.heartbeat_timeout_sec:.1f}s"
                )
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for VLM request slot env_id={env_id}, view_id={view_id}."
                )
            time.sleep(min(poll, max(0.0, deadline - time.perf_counter())))

    def _read_payload(
        self,
        batch: int,
        step: int,
        *,
        env_ids: np.ndarray,
        view_ids: np.ndarray,
        target_request_ids: list[int] | None = None,
        target_frame_ids: list[int] | None = None,
        target_generations: list[int] | None = None,
        target_episodes: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        arrays = self.shared.arrays
        sem, feats = [], []
        source_step, age, valid, identity = [], [], [], []
        fresh, aligned, consistent, stale_episode, version_error = [], [], [], [], []
        frame_ids, frame_generations, frame_steps, frame_episodes, frame_times = [], [], [], [], []
        request_ids, sem_source_request_ids = [], []
        sem_generations, sem_source_frames, sem_source_frame_generations = [], [], []
        sem_source_steps, sem_source_episodes, sem_source_envs, sem_source_views, sem_times = [], [], [], [], []
        inference_start, inference_end, age_env_steps, age_ms = [], [], [], []
        latest_exact_values, view_exact_values = [], []
        for index, (env_id_raw, view_id_raw) in enumerate(zip(env_ids, view_ids, strict=True)):
            env_id = int(env_id_raw)
            view_id = _validate_view_id(view_id_raw, self.spec)
            frame_snap = self.shared.read_consistent_frame_snapshot(env_id, view_id)
            sem_snap = self.shared.read_consistent_semantic_snapshot(env_id, view_id)
            frame_id = frame_snap.frame_id if frame_snap is not None else int(arrays["frame_id"][env_id, view_id])
            frame_generation = (
                frame_snap.generation_id
                if frame_snap is not None
                else int(arrays["frame_generation_id"][env_id, view_id])
            )
            frame_step = (
                frame_snap.env_step
                if frame_snap is not None
                else int(arrays["frame_env_step"][env_id, view_id])
            )
            frame_episode = (
                frame_snap.episode_id
                if frame_snap is not None
                else int(arrays["episode_version"][env_id, view_id])
            )
            frame_time = (
                frame_snap.timestamp_ns
                if frame_snap is not None
                else int(arrays["frame_timestamp_ns"][env_id, view_id])
            )
            target_request_id = (
                int(target_request_ids[index])
                if target_request_ids is not None and index < len(target_request_ids)
                else int(arrays["request_id"][env_id, view_id])
            )
            target_frame_id = (
                int(target_frame_ids[index])
                if target_frame_ids is not None and index < len(target_frame_ids)
                else int(frame_id)
            )
            target_generation = (
                int(target_generations[index])
                if target_generations is not None and index < len(target_generations)
                else int(frame_generation)
            )
            target_episode = (
                int(target_episodes[index])
                if target_episodes is not None and index < len(target_episodes)
                else int(frame_episode)
            )
            sem_valid = sem_snap is not None and bool(sem_snap.valid) and bool(sem_snap.consistent)
            same_episode = bool(sem_valid and sem_snap.source_episode_id == frame_episode)
            same_env = bool(sem_valid and sem_snap.source_env_id == env_id)
            same_view = bool(sem_valid and sem_snap.source_view_id == view_id and sem_snap.view_id == view_id)
            if sem_valid:
                raw_age_frames = int(frame_id - sem_snap.source_frame_id)
                raw_age_steps = int(frame_step - sem_snap.source_env_step)
                raw_age_ms = max(0.0, (float(frame_time) - float(sem_snap.semantic_timestamp_ns)) / 1e6)
            else:
                raw_age_frames = self.max_age + 1
                raw_age_steps = self.max_age + 1
                raw_age_ms = math.inf
            is_version_error = bool(
                sem_valid and (raw_age_frames < 0 or raw_age_steps < 0 or not same_env or not same_view)
            )
            is_fresh = bool(
                sem_valid and same_episode and same_view and not is_version_error and raw_age_frames <= self.max_age
            )
            is_aligned = bool(is_fresh and raw_age_frames == 0 and sem_snap.source_frame_generation_id == frame_generation)
            latest_exact = bool(
                sem_valid
                and same_env
                and same_view
                and sem_snap.source_episode_id == frame_episode
                and sem_snap.source_frame_id == frame_id
                and sem_snap.source_frame_generation_id == frame_generation
            )
            exact_request = bool(
                sem_valid
                and same_env
                and same_view
                and sem_snap.source_request_id == target_request_id
                and sem_snap.source_episode_id == target_episode
                and sem_snap.source_frame_id == target_frame_id
                and sem_snap.source_frame_generation_id == target_generation
            )
            if self.strict_per_frame and not exact_request:
                self._record_strict_mismatch(
                    env_id,
                    view_id=view_id,
                    frame_id=frame_id,
                    frame_generation=frame_generation,
                    frame_episode=frame_episode,
                    target_request_id=target_request_id,
                    target_frame_id=target_frame_id,
                    target_generation=target_generation,
                    target_episode=target_episode,
                    sem_snap=sem_snap,
                )
                raise RuntimeError(
                    "Strict VLM exact-frame mismatch: "
                    f"env_id={env_id}, view_id={view_id}, request_id={target_request_id}, frame_id={target_frame_id}, "
                    f"episode_id={target_episode}, frame_generation={target_generation}, "
                    f"semantic={sem_snap}"
                )
            use_cache = bool(exact_request if self.strict_per_frame else is_fresh)
            if use_cache:
                sem.append(sem_snap.semantic_map)
                feats.append(sem_snap.patch_features.reshape(self.spec.token_count, self.spec.feature_dim))
                source_step.append(sem_snap.source_env_step)
                age.append(max(0, raw_age_frames))
                valid.append(True)
                identity.append(False)
                self._last_semantic_version[env_id, view_id] = sem_snap.generation_id
                self._last_valid["semantic"][env_id, view_id] = sem[-1]
                self._last_valid["features"][env_id, view_id] = feats[-1]
                arrays["stats"][STAT_CACHE_HIT] += 1
            else:
                if sem_valid and same_episode:
                    arrays["stats"][STAT_STALE] += 1
                fallback = self._fallback_for(env_id, view_id)
                sem.append(fallback["semantic"])
                feats.append(fallback["features"])
                source_step.append(-1 if sem_snap is None else sem_snap.source_env_step)
                age.append(max(0, raw_age_frames if raw_age_frames < 10**8 else self.max_age + 1))
                valid.append(False)
                identity.append(self.fallback == "identity")
                arrays["stats"][STAT_FALLBACK] += 1
            fresh.append(is_fresh)
            aligned.append(is_aligned)
            consistent.append(bool(frame_snap is not None and (sem_snap is None or sem_snap.consistent)))
            stale_episode.append(bool(sem_valid and not same_episode))
            version_error.append(is_version_error)
            latest_exact_values.append(latest_exact)
            view_exact_values.append(same_view)
            frame_ids.append(frame_id)
            frame_generations.append(frame_generation)
            frame_steps.append(frame_step)
            frame_episodes.append(frame_episode)
            frame_times.append(frame_time)
            request_ids.append(target_request_id)
            sem_source_request_ids.append(-1 if sem_snap is None else sem_snap.source_request_id)
            sem_generations.append(0 if sem_snap is None else sem_snap.generation_id)
            sem_source_frames.append(-1 if sem_snap is None else sem_snap.source_frame_id)
            sem_source_frame_generations.append(-1 if sem_snap is None else sem_snap.source_frame_generation_id)
            sem_source_steps.append(-1 if sem_snap is None else sem_snap.source_env_step)
            sem_source_episodes.append(-1 if sem_snap is None else sem_snap.source_episode_id)
            sem_source_envs.append(-1 if sem_snap is None else sem_snap.source_env_id)
            sem_source_views.append(-1 if sem_snap is None else sem_snap.source_view_id)
            sem_times.append(0 if sem_snap is None else sem_snap.semantic_timestamp_ns)
            inference_start.append(0 if sem_snap is None else sem_snap.inference_start_ns)
            inference_end.append(0 if sem_snap is None else sem_snap.inference_end_ns)
            age_env_steps.append(max(0, raw_age_steps if raw_age_steps < 10**8 else self.max_age + 1))
            age_ms.append(0.0 if not math.isfinite(raw_age_ms) else raw_age_ms)
        return {
            "vlm_semantic_map": torch.as_tensor(np.stack(sem), dtype=torch.float16),
            "vlm_patch_scores": torch.as_tensor(np.stack(sem), dtype=torch.float16).reshape(batch, self.spec.token_count),
            "vlm_patch_features": torch.as_tensor(np.stack(feats), dtype=torch.float16),
            "vlm_source_step": torch.as_tensor(source_step, dtype=torch.int64).reshape(batch, 1),
            "vlm_age": torch.as_tensor(age, dtype=torch.int64).reshape(batch, 1),
            "vlm_age_frames": torch.as_tensor(age, dtype=torch.int64).reshape(batch, 1),
            "vlm_age_env_steps": torch.as_tensor(age_env_steps, dtype=torch.int64).reshape(batch, 1),
            "vlm_age_ms": torch.as_tensor(age_ms, dtype=torch.float32).reshape(batch, 1),
            "vlm_valid": torch.as_tensor(valid, dtype=torch.bool).reshape(batch, 1),
            "vlm_fresh": torch.as_tensor(fresh, dtype=torch.bool).reshape(batch, 1),
            "vlm_aligned": torch.as_tensor(aligned, dtype=torch.bool).reshape(batch, 1),
            "vlm_snapshot_consistent": torch.as_tensor(consistent, dtype=torch.bool).reshape(batch, 1),
            "vlm_stale_episode": torch.as_tensor(stale_episode, dtype=torch.bool).reshape(batch, 1),
            "vlm_version_error": torch.as_tensor(version_error, dtype=torch.bool).reshape(batch, 1),
            "vlm_fallback_identity": torch.as_tensor(identity, dtype=torch.bool).reshape(batch, 1),
            "vlm_latest_frame_exact_match": torch.as_tensor(latest_exact_values, dtype=torch.bool).reshape(batch, 1),
            "vlm_view_exact_match": torch.as_tensor(view_exact_values, dtype=torch.bool).reshape(batch, 1),
            "vlm_env_id": torch.as_tensor(env_ids, dtype=torch.int64).reshape(batch, 1),
            "vlm_view_id": torch.as_tensor(view_ids, dtype=torch.int64).reshape(batch, 1),
            "vlm_frame_id": torch.as_tensor(frame_ids, dtype=torch.int64).reshape(batch, 1),
            "vlm_frame_generation_id": torch.as_tensor(frame_generations, dtype=torch.int64).reshape(batch, 1),
            "vlm_frame_env_step": torch.as_tensor(frame_steps, dtype=torch.int64).reshape(batch, 1),
            "vlm_episode_id": torch.as_tensor(frame_episodes, dtype=torch.int64).reshape(batch, 1),
            "vlm_frame_timestamp_ns": torch.as_tensor(frame_times, dtype=torch.int64).reshape(batch, 1),
            "vlm_request_id": torch.as_tensor(request_ids, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_request_id": torch.as_tensor(sem_source_request_ids, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_generation_id": torch.as_tensor(sem_generations, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_frame_id": torch.as_tensor(sem_source_frames, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_frame_generation_id": torch.as_tensor(sem_source_frame_generations, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_env_step": torch.as_tensor(sem_source_steps, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_episode_id": torch.as_tensor(sem_source_episodes, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_env_id": torch.as_tensor(sem_source_envs, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_source_view_id": torch.as_tensor(sem_source_views, dtype=torch.int64).reshape(batch, 1),
            "vlm_semantic_timestamp_ns": torch.as_tensor(sem_times, dtype=torch.int64).reshape(batch, 1),
            "vlm_inference_start_ns": torch.as_tensor(inference_start, dtype=torch.int64).reshape(batch, 1),
            "vlm_inference_end_ns": torch.as_tensor(inference_end, dtype=torch.int64).reshape(batch, 1),
        }

    def _fallback_for(self, env_id: int, view_id: int) -> dict[str, np.ndarray]:
        if self.fallback == "last_valid":
            return {
                "semantic": self._last_valid["semantic"][env_id, view_id],
                "features": self._last_valid["features"][env_id, view_id],
            }
        return _fallback_payload(
            self.spec, 1, mode=self.fallback, offset=env_id * self.spec.view_num + view_id
        )

    def _record_strict_mismatch(
        self,
        env_id: int,
        *,
        view_id: int,
        frame_id: int,
        frame_generation: int,
        frame_episode: int,
        target_request_id: int,
        target_frame_id: int,
        target_generation: int,
        target_episode: int,
        sem_snap: SemanticSnapshot | None,
    ) -> None:
        arrays = self.shared.arrays
        if sem_snap is None:
            arrays["stats"][STAT_STALE_RESULT_REJECTED] += 1
            return
        if int(sem_snap.source_env_id) != int(env_id):
            arrays["stats"][STAT_CROSS_ENV_MISMATCH] += 1
        if int(sem_snap.source_view_id) != int(view_id):
            arrays["stats"][STAT_CROSS_VIEW_MISMATCH] += 1
        if int(sem_snap.source_episode_id) != int(target_episode):
            arrays["stats"][STAT_CROSS_EPISODE_MISMATCH] += 1
            if int(sem_snap.source_episode_id) < int(target_episode):
                arrays["stats"][STAT_OLD_EPISODE_RESULT_REJECTED] += 1
        if int(sem_snap.source_frame_id) > int(target_frame_id) or int(sem_snap.source_frame_id) > int(frame_id):
            arrays["stats"][STAT_FUTURE_FRAME_VIOLATION] += 1
        if int(sem_snap.source_frame_id) < int(target_frame_id):
            arrays["stats"][STAT_STALE_RESULT_REJECTED] += 1
        if int(sem_snap.source_request_id) != int(target_request_id):
            arrays["stats"][STAT_OUT_OF_ORDER] += 1
        if int(sem_snap.source_frame_generation_id) != int(target_generation):
            arrays["stats"][STAT_OUT_OF_ORDER] += 1
        if int(frame_generation) != int(target_generation) or int(frame_episode) != int(target_episode):
            arrays["stats"][STAT_OUT_OF_ORDER] += 1


def publish_semantic_snapshot(
    shared: VLMSharedMemory,
    env_id: int,
    view_id: int,
    frame: FrameSnapshot,
    semantic_map: np.ndarray,
    patch_features: np.ndarray,
    *,
    inference_start_ns: int,
    inference_end_ns: int,
    checkpoint_hash: int = 0,
    preprocess_hash: int = 0,
    request_id: int | None = None,
) -> int:
    arrays = shared.arrays
    env_id = int(env_id)
    view_id = _validate_view_id(view_id, shared.spec)
    if int(frame.env_id) != env_id or int(frame.view_id) != view_id:
        raise RuntimeError(
            "Cannot publish VLM semantics to a different stream: "
            f"target=({env_id},{view_id}), frame=({frame.env_id},{frame.view_id})"
        )
    version = int(arrays["semantic_version"][env_id, view_id])
    if version % 2:
        version += 1
    arrays["semantic_version"][env_id, view_id] = version + 1
    arrays["semantic_valid"][env_id, view_id] = 0
    arrays["semantic_map"][env_id, view_id] = np.asarray(semantic_map, dtype=np.float16).reshape(
        arrays["semantic_map"][env_id, view_id].shape
    )
    arrays["patch_features"][env_id, view_id] = np.asarray(patch_features, dtype=np.float16).reshape(
        arrays["patch_features"][env_id, view_id].shape
    )
    now_ns = int(time.time_ns())
    generation = int(arrays["semantic_generation_id"][env_id, view_id]) + 1
    arrays["semantic_generation_id"][env_id, view_id] = generation
    if request_id is None:
        request_id = int(arrays["request_id"][env_id, view_id])
    arrays["semantic_source_request_id"][env_id, view_id] = int(request_id)
    arrays["semantic_source_step"][env_id, view_id] = int(frame.env_step)
    arrays["semantic_source_env_step"][env_id, view_id] = int(frame.env_step)
    arrays["semantic_source_frame_id"][env_id, view_id] = int(frame.frame_id)
    arrays["semantic_source_frame_generation_id"][env_id, view_id] = int(frame.generation_id)
    arrays["semantic_source_episode_id"][env_id, view_id] = int(frame.episode_id)
    arrays["semantic_source_env_id"][env_id, view_id] = int(frame.env_id)
    arrays["semantic_source_view_id"][env_id, view_id] = int(frame.view_id)
    arrays["semantic_episode_version"][env_id, view_id] = int(frame.episode_id)
    arrays["semantic_timestamp_ns"][env_id, view_id] = now_ns
    arrays["vlm_inference_start_ns"][env_id, view_id] = int(inference_start_ns)
    arrays["vlm_inference_end_ns"][env_id, view_id] = int(inference_end_ns)
    arrays["vlm_checkpoint_hash"][env_id, view_id] = int(checkpoint_hash)
    arrays["vlm_preprocess_hash"][env_id, view_id] = int(preprocess_hash)
    arrays["semantic_valid"][env_id, view_id] = 1
    arrays["semantic_version"][env_id, view_id] = version + 2
    arrays["stats"][STAT_COMPLETED] += 1
    arrays["stats"][STAT_SEMANTIC_PUBLISH] += 1
    arrays["completed_per_env"][env_id, view_id] += 1
    return generation


def _fallback_payload(spec: VLMSharedSpec, batch: int, *, mode: str, offset: int = 0, squeeze_batch: bool = True):
    if mode == "uniform":
        semantic = np.full((batch, spec.grid_size, spec.grid_size), 1.0 / max(spec.token_count, 1), dtype=np.float32)
    else:
        semantic = np.ones((batch, spec.grid_size, spec.grid_size), dtype=np.float32)
    features = np.zeros((batch, spec.token_count, spec.feature_dim), dtype=np.float32)
    if batch == 1 and squeeze_batch:
        return {"semantic": semantic[0], "features": features[0]}
    return {"semantic": semantic, "features": features}


def _create_or_attach(name: str, size: int, *, create: bool) -> shared_memory.SharedMemory:
    if create:
        try:
            old = shared_memory.SharedMemory(name=name, create=False)
            _unregister_unowned_shared_memory(old)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    block = shared_memory.SharedMemory(name=name, create=False)
    _unregister_unowned_shared_memory(block)
    return block


def _unregister_unowned_shared_memory(block: shared_memory.SharedMemory) -> None:
    try:
        resource_tracker.unregister(block._name, "shared_memory")
    except Exception:
        pass


def _resolve_dim(spec: VLMSharedSpec, dim: Any) -> int:
    if isinstance(dim, str):
        return int(getattr(spec, dim))
    return int(dim)


def _shared_spec_values(spec: VLMSharedSpec) -> np.ndarray:
    return np.asarray(
        [
            spec.env_num,
            spec.clip_len,
            spec.height,
            spec.width,
            spec.grid_size,
            spec.feature_dim,
            spec.view_num,
            PROTOCOL_VERSION,
        ],
        dtype=np.int64,
    )


def _shared_spec_dict(values: np.ndarray) -> dict[str, int]:
    names = (
        "env_num",
        "clip_len",
        "height",
        "width",
        "grid_size",
        "feature_dim",
        "view_num",
        "protocol_version",
    )
    return {name: int(value) for name, value in zip(names, values.tolist())}


def _to_numpy_image(image) -> np.ndarray:
    if torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected images [B,H,W,3] or [B,1,H,W,3], got {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_numpy_bool(value, batch: int) -> np.ndarray:
    if value is None:
        return np.zeros((batch,), dtype=bool)
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.reshape(batch, -1)[:, 0].astype(bool)


def _to_numpy_int(
    value,
    batch: int,
    *,
    default: np.ndarray | list[int] | None = None,
    required: bool = False,
) -> np.ndarray:
    if value is None:
        if required:
            raise ValueError("Explicit VLM stream identity is required by protocol v5.")
        if default is None:
            return np.zeros((batch,), dtype=np.int64)
        return np.asarray(default, dtype=np.int64).reshape(batch)
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.reshape(batch, -1)[:, 0].astype(np.int64)


def _validate_view_id(view_id: int, spec: VLMSharedSpec) -> int:
    value = int(view_id)
    if value < 0 or value >= int(spec.view_num):
        raise ValueError(f"view_id={value} is outside [0,{int(spec.view_num) - 1}]")
    return value


def _validate_stream_batch(spec: VLMSharedSpec, env_ids: np.ndarray, view_ids: np.ndarray) -> None:
    if len(env_ids) != len(view_ids):
        raise ValueError(f"env/view batch length mismatch: {len(env_ids)} != {len(view_ids)}")
    streams = []
    for env_id_raw, view_id_raw in zip(env_ids, view_ids, strict=True):
        env_id = int(env_id_raw)
        if env_id < 0 or env_id >= int(spec.env_num):
            raise ValueError(f"env_id={env_id} is outside [0,{int(spec.env_num) - 1}]")
        view_id = _validate_view_id(view_id_raw, spec)
        streams.append((env_id, view_id))
    if len(streams) != len(set(streams)):
        raise ValueError(f"duplicate (env_id, view_id) streams in one VLM batch: {streams}")


def _resize_or_check_image(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[:2] == (height, width):
        return frame
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((int(width), int(height)), Image.BILINEAR), dtype=np.uint8)


def write_heartbeat(shared: VLMSharedMemory) -> None:
    shared.arrays["heartbeat_ns"][0] = int(time.time_ns())


def heartbeat_age_sec(shared: VLMSharedMemory) -> float:
    raw = int(shared.arrays["heartbeat_ns"][0])
    if raw <= 0:
        return math.inf
    return max(0.0, (time.time_ns() - raw) / 1e9)


def _oldest_pending_wait_ms(shared: VLMSharedMemory) -> float:
    pending = shared.arrays["request_pending"].astype(bool)
    if not bool(pending.any()):
        return 0.0
    times = shared.arrays["request_timestamp_ns"][pending].astype(np.int64)
    times = times[times > 0]
    if times.size == 0:
        return 0.0
    return max(0.0, (time.time_ns() - int(times.min())) / 1e6)


def _service_uptime_sec(shared: VLMSharedMemory) -> float:
    raw = int(shared.arrays["service_start_ns"][0])
    if raw <= 0:
        return 0.0
    return max(0.0, (time.time_ns() - raw) / 1e9)


def ensure_ready_file(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
