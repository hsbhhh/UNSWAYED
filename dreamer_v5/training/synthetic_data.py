from __future__ import annotations

import numpy as np
import torch

from ..pipeline.replay import _stack_batch
from ..pipeline.temporal_record import TemporalRecordBuilder


def make_synthetic_sequence(
    *,
    batch_size: int,
    sequence_length: int,
    action_dim: int,
    image_size: int,
    patch_count: int,
    feature_dim: int,
    seed: int = 0,
    num_views: int = 2,
    schema_version: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    rng = np.random.default_rng(int(seed))
    builder = TemporalRecordBuilder(
        env_num=batch_size,
        action_dim=action_dim,
        counterfactual_views=int(num_views),
        schema_version=schema_version,
    )
    records = []
    for env_id in range(batch_size):
        builder.reset_env(env_id)
        for step in range(sequence_length):
            action = rng.normal(0.0, 0.25, size=(action_dim,)).astype(np.float32)
            fg = _synthetic_agent_image(rng, image_size=image_size, env_id=env_id, step=step)
            bg_shape = fg[..., :3].shape
            backgrounds = [
                rng.uniform(0.0, 1.0, size=bg_shape).astype(np.float32)
                for _ in range(int(num_views))
            ]
            alpha = fg[..., 3:4]
            raw_views = [fg[..., :3] * alpha + background * (1.0 - alpha) for background in backgrounds]
            raw_a = raw_views[0]
            scores = _synthetic_patch_scores(patch_count, high=(step + env_id) % patch_count)
            features = _synthetic_patch_features(rng, patch_count, feature_dim, scores)
            score_views = np.stack(
                [
                    np.clip(scores * (0.55 + 0.15 * view_id) + 0.03 * view_id, 0.0, 1.0)
                    for view_id in range(int(num_views))
                ]
            ).astype(np.float32)
            feature_views = np.stack(
                [
                    features + rng.normal(0.0, 0.02, size=features.shape).astype(np.float32)
                    for _ in range(int(num_views))
                ]
            )
            common = dict(
                raw_a=raw_a.transpose(2, 0, 1),
                outgoing_action=action,
                paired_view_valid=True,
                vlm_pair_valid=True,
                metadata={
                    "step_id": int(step),
                    "camera_id": 0,
                    "physics_state_hash": f"synthetic:{env_id}:{step}",
                    "view_ids_b": tuple(
                        f"B{view_id + 1}:synthetic_{view_id}:{step}"
                        for view_id in range(int(num_views))
                    ),
                    "background_ids_b": tuple(
                        f"synthetic_{view_id}:{step}"
                        for view_id in range(int(num_views))
                    ),
                },
            )
            generic_k_view = str(builder.schema_version) == "v4_kview_v2"
            if int(num_views) == 2 and not generic_k_view:
                rec = builder.build_current_record(
                    env_id,
                    raw_b1=raw_views[0].transpose(2, 0, 1),
                    raw_b2=raw_views[1].transpose(2, 0, 1),
                    patch_scores_b1=score_views[0],
                    patch_scores_b2=score_views[1],
                    patch_features_b1=feature_views[0],
                    patch_features_b2=feature_views[1],
                    **common,
                )
            else:
                rec = builder.build_current_record(
                    env_id,
                    raw_b_views=np.stack(raw_views).astype(np.float32),
                    patch_scores=score_views,
                    patch_features=feature_views,
                    **common,
                )
            records.append(rec)
            builder.advance_after_env_step(
                env_id,
                outgoing_action=action,
                reward=float(scores.mean()),
                is_last=False,
                is_terminal=False,
            )
    sequences = [
        records[index * sequence_length : (index + 1) * sequence_length]
        for index in range(batch_size)
    ]
    arrays = _stack_batch(sequences)
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    batch["sequence_valid"] = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    return batch, records


def _synthetic_agent_image(rng: np.random.Generator, *, image_size: int, env_id: int, step: int) -> np.ndarray:
    img = np.zeros((image_size, image_size, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cx = image_size * (0.25 + 0.5 * ((step + env_id) % 7) / 6.0)
    cy = image_size * (0.35 + 0.2 * (env_id % 3))
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (image_size * 0.10) ** 2
    img[..., 0] = 0.9
    img[..., 1] = 0.85
    img[..., 2] = 0.1
    img[..., 3] = mask.astype(np.float32)
    return img


def _synthetic_patch_scores(patch_count: int, *, high: int) -> np.ndarray:
    scores = np.full(int(patch_count), 0.02, dtype=np.float32)
    scores[int(high) % int(patch_count)] = 0.98
    return scores


def _synthetic_patch_features(
    rng: np.random.Generator,
    patch_count: int,
    feature_dim: int,
    scores: np.ndarray,
) -> np.ndarray:
    base = rng.normal(0.0, 0.2, size=(int(patch_count), int(feature_dim))).astype(np.float32)
    base += scores[:, None].astype(np.float32)
    return base
