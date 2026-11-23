from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


TEMPORAL_SCHEMA_VERSION = "v4_temporal_v1"
ROBUST_FOUR_VIEW_SCHEMA_VERSION = "v4_robust_4view_v1"
COUNTERFACTUAL_SCHEMA_VERSION = "v4_counterfactual_k6_v2"
K_VIEW_SCHEMA_VERSION = "v4_kview_v2"
TEMPORAL_CONTRACT = "obs_t_with_incoming_action"


@dataclass
class TemporalRecordState:
    env_id: int
    action_dim: int
    episode_id: int = 0
    observation_index: int = 0
    is_first: bool = True
    incoming_action: np.ndarray = field(init=False)
    incoming_reward: float = 0.0
    incoming_is_last: bool = False
    incoming_is_terminal: bool = False

    def __post_init__(self) -> None:
        self.incoming_action = np.zeros(int(self.action_dim), dtype=np.float32)

    def reset_episode(self) -> None:
        self.episode_id += 1
        self.observation_index = 0
        self.is_first = True
        self.incoming_action[...] = 0.0
        self.incoming_reward = 0.0
        self.incoming_is_last = False
        self.incoming_is_terminal = False


class TemporalRecordBuilder:
    """Build replay rows with obs_t aligned to action_{t-1}.

    The model-facing ``action`` field is an alias of ``incoming_action``. The
    optional ``outgoing_action`` field is debug evidence only and must not be
    used by RSSM.observe.
    """

    def __init__(
        self,
        *,
        env_num: int,
        action_dim: int,
        counterfactual_views: int = 2,
        view_ids: tuple[str, ...] | list[str] | None = None,
        schema_version: str | None = None,
        temporal_protocol_version: str = "v5_independent_view_ring",
        pairing_contract: str = "same_physical_state_same_camera_background_only",
        background_stream_policy: str = "continuous_per_env_view",
    ):
        self.env_num = int(env_num)
        self.action_dim = int(action_dim)
        self.counterfactual_views = int(counterfactual_views)
        if self.counterfactual_views not in {2, 4, 6}:
            raise ValueError("K-view replay supports num_views in {2,4,6}")
        self.view_ids = tuple(view_ids or [f"B{index + 1}" for index in range(self.counterfactual_views)])
        self.schema_version = str(
            schema_version
            or (
                COUNTERFACTUAL_SCHEMA_VERSION
                if self.counterfactual_views == 6
                else ROBUST_FOUR_VIEW_SCHEMA_VERSION
                if self.counterfactual_views == 4
                else TEMPORAL_SCHEMA_VERSION
            )
        )
        self.temporal_protocol_version = str(temporal_protocol_version)
        self.pairing_contract = str(pairing_contract)
        self.background_stream_policy = str(background_stream_policy)
        if len(self.view_ids) != self.counterfactual_views:
            raise ValueError("view_ids length must equal counterfactual_views")
        self.states = [TemporalRecordState(env_id=i, action_dim=action_dim) for i in range(self.env_num)]

    def reset_env(self, env_id: int) -> None:
        self.states[int(env_id)].reset_episode()

    def build_current_record(
        self,
        env_id: int,
        *,
        raw_a: Any,
        raw_b1: Any | None = None,
        raw_b2: Any | None = None,
        raw_b_views: Any | None = None,
        outgoing_action: Any | None = None,
        patch_scores_b1: Any | None = None,
        patch_scores_b2: Any | None = None,
        patch_features_b1: Any | None = None,
        patch_features_b2: Any | None = None,
        patch_scores: Any | None = None,
        patch_features: Any | None = None,
        paired_view_valid: bool = True,
        vlm_pair_valid: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.states[int(env_id)]
        incoming_action = state.incoming_action.astype(np.float32, copy=True)
        if state.is_first and not np.allclose(incoming_action, 0.0):
            raise AssertionError("UNSWAYED temporal contract violation: is_first row must store zero incoming action.")
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "temporal_contract": TEMPORAL_CONTRACT,
            "counterfactual_K": self.counterfactual_views,
            "counterfactual_view_ids": self.view_ids,
            "temporal_protocol_version": self.temporal_protocol_version,
            "pairing_contract": self.pairing_contract,
            "background_stream_policy": self.background_stream_policy,
            "env_id": int(env_id),
            "episode_id": int(state.episode_id),
            "observation_index": int(state.observation_index),
            "incoming_action_index": -1 if state.is_first else int(state.observation_index - 1),
            "outgoing_action_index": int(state.observation_index),
            "raw_a": raw_a,
            "action": incoming_action,
            "incoming_action": incoming_action.copy(),
            "reward": np.float32(state.incoming_reward),
            "continue": np.float32(0.0 if state.incoming_is_terminal else 1.0),
            "is_first": bool(state.is_first),
            "is_last": bool(state.incoming_is_last),
            "is_terminal": bool(state.incoming_is_terminal),
            "paired_view_valid": bool(paired_view_valid),
            "vlm_pair_valid": bool(vlm_pair_valid),
        }
        generic_k_view = self.schema_version == K_VIEW_SCHEMA_VERSION
        if generic_k_view or self.counterfactual_views in {4, 6}:
            if raw_b_views is None:
                raise ValueError(f"K={self.counterfactual_views} temporal records require raw_b_views")
            views = np.asarray(raw_b_views)
            if views.shape[0] != self.counterfactual_views:
                raise ValueError(
                    f"raw_b_views must start with K={self.counterfactual_views}, got {views.shape}"
                )
            record["raw_b_views"] = views
            record["multi_view_valid"] = bool(paired_view_valid)
            if not generic_k_view and self.counterfactual_views == 4:
                record["four_view_valid"] = bool(paired_view_valid)
            elif not generic_k_view:
                record["six_view_valid"] = bool(paired_view_valid)
            record["vlm_views_valid"] = bool(vlm_pair_valid)
        else:
            record["raw_b1"] = raw_b1
            record["raw_b2"] = raw_b2
        if outgoing_action is not None:
            record["outgoing_action"] = np.asarray(outgoing_action, dtype=np.float32).copy()
        if patch_scores_b1 is not None:
            record["patch_scores_b1"] = patch_scores_b1
        if patch_scores_b2 is not None:
            record["patch_scores_b2"] = patch_scores_b2
        if patch_features_b1 is not None:
            record["patch_features_b1"] = patch_features_b1
        if patch_features_b2 is not None:
            record["patch_features_b2"] = patch_features_b2
        if patch_scores is not None:
            record["patch_scores"] = patch_scores
        if patch_features is not None:
            record["patch_features"] = patch_features
        if metadata:
            record.update(metadata)
        return record

    def advance_after_env_step(
        self,
        env_id: int,
        *,
        outgoing_action: Any,
        reward: float,
        is_last: bool = False,
        is_terminal: bool = False,
        next_is_first: bool = False,
    ) -> None:
        state = self.states[int(env_id)]
        if next_is_first:
            state.reset_episode()
            return
        action = np.asarray(outgoing_action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"outgoing_action must have shape {(self.action_dim,)}, got {action.shape}")
        state.incoming_action = action.copy()
        state.incoming_reward = float(reward)
        state.incoming_is_last = bool(is_last)
        state.incoming_is_terminal = bool(is_terminal)
        state.is_first = False
        state.observation_index += 1


def summarize_records(records: Iterable[dict[str, Any]], *, atol: float = 1.0e-6) -> dict[str, float]:
    rows = list(records)
    first_count = 0
    first_zero = 0
    nonfirst_count = 0
    prev_match = 0
    violations = 0
    cross_env_leaks = 0
    previous_by_stream: dict[tuple[int, int], np.ndarray] = {}
    last_outgoing_by_env: dict[int, np.ndarray] = {}
    for row in rows:
        env_id = int(row["env_id"])
        episode_id = int(row["episode_id"])
        stream = (env_id, episode_id)
        incoming = np.asarray(row["action"], dtype=np.float32)
        is_first = bool(row["is_first"])
        if is_first:
            first_count += 1
            if np.allclose(incoming, 0.0, atol=atol):
                first_zero += 1
            else:
                violations += 1
            if env_id in last_outgoing_by_env and np.allclose(incoming, last_outgoing_by_env[env_id], atol=atol):
                cross_env_leaks += 1
        else:
            nonfirst_count += 1
            expected = previous_by_stream.get(stream)
            if expected is None:
                violations += 1
            elif np.allclose(incoming, expected, atol=atol):
                prev_match += 1
            else:
                violations += 1
        if "incoming_action" in row and not np.allclose(incoming, np.asarray(row["incoming_action"], dtype=np.float32), atol=atol):
            violations += 1
        if "outgoing_action" in row:
            outgoing = np.asarray(row["outgoing_action"], dtype=np.float32)
            previous_by_stream[stream] = outgoing.copy()
            last_outgoing_by_env[env_id] = outgoing.copy()
    return {
        "temporal/record_count": float(len(rows)),
        "temporal/is_first_count": float(first_count),
        "temporal/nonfirst_count": float(nonfirst_count),
        "temporal/is_first_action_zero_fraction": _safe_ratio(first_zero, first_count),
        "temporal/nonfirst_prev_action_match_fraction": _safe_ratio(prev_match, nonfirst_count),
        "temporal/action_alignment_violation_count": float(violations),
        "temporal/cross_env_action_leak_count": float(cross_env_leaks),
    }


def stack_records(records: list[dict[str, Any]], *, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
    if len(records) != batch_size * sequence_length:
        raise ValueError("records must contain exactly batch_size * sequence_length rows")
    out: dict[str, list[Any]] = {}
    for row in records:
        for key, value in row.items():
            if key in {
                "schema_version",
                "temporal_contract",
                "counterfactual_view_ids",
                "temporal_protocol_version",
                "pairing_contract",
                "background_stream_policy",
            }:
                continue
            if value is None:
                continue
            out.setdefault(key, []).append(value)
    stacked: dict[str, np.ndarray] = {}
    for key, values in out.items():
        try:
            stacked[key] = np.asarray(values).reshape(batch_size, sequence_length, *np.asarray(values[0]).shape)
        except Exception:
            continue
    return stacked


def _safe_ratio(num: int, den: int) -> float:
    return 1.0 if den == 0 else float(num) / float(den)
