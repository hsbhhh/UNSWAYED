from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .temporal_record import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    K_VIEW_SCHEMA_VERSION,
    ROBUST_FOUR_VIEW_SCHEMA_VERSION,
    TEMPORAL_CONTRACT,
    TEMPORAL_SCHEMA_VERSION,
)


REQUIRED_RECORD_KEYS = (
    "raw_a",
    "raw_b1",
    "raw_b2",
    "action",
    "reward",
    "continue",
    "is_first",
    "is_last",
    "is_terminal",
    "paired_view_valid",
)

VLM_RECORD_KEYS = (
    "patch_scores_b1",
    "patch_scores_b2",
    "patch_features_b1",
    "patch_features_b2",
    "vlm_pair_valid",
)

K6_REQUIRED_RECORD_KEYS = (
    "raw_a",
    "raw_b_views",
    "action",
    "reward",
    "continue",
    "is_first",
    "is_last",
    "is_terminal",
    "paired_view_valid",
    "six_view_valid",
)

K6_VLM_RECORD_KEYS = (
    "patch_scores",
    "patch_features",
    "vlm_views_valid",
)

MULTI_VIEW_REQUIRED_RECORD_KEYS = (
    "raw_a",
    "raw_b_views",
    "action",
    "reward",
    "continue",
    "is_first",
    "is_last",
    "is_terminal",
    "paired_view_valid",
    "multi_view_valid",
)

MULTI_VIEW_VLM_RECORD_KEYS = (
    "patch_scores",
    "patch_features",
    "vlm_views_valid",
)


@dataclass(frozen=True)
class ReplaySchema:
    schema_version: str = TEMPORAL_SCHEMA_VERSION
    temporal_contract: str = TEMPORAL_CONTRACT
    counterfactual_K: int = 2
    view_ids: tuple[str, ...] = ("B1", "B2")
    temporal_protocol_version: str = "v5_independent_view_ring"
    pairing_contract: str = "same_physical_state_same_camera_background_only"
    background_stream_policy: str = "continuous_per_env_view"
    action_semantics: str = "incoming_action"
    reward_semantics: str = "reward_from_incoming_action"
    continue_semantics: str = "1_minus_is_terminal"

    def to_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["view_ids"] = list(self.view_ids)
        return out

    @classmethod
    def from_config(cls, config) -> "ReplaySchema":
        counterfactual = config.get("counterfactual", {})
        count = int(counterfactual.get("views", 2))
        return cls(
            schema_version=str(config.replay.schema_version),
            temporal_contract=str(config.replay.temporal_contract),
            counterfactual_K=count,
            view_ids=tuple(map(str, counterfactual.get("view_ids", [f"B{i + 1}" for i in range(count)]))),
            temporal_protocol_version=str(counterfactual.get("temporal_protocol_version", "v5_independent_view_ring")),
            pairing_contract=str(counterfactual.get("pairing_contract", "same_physical_state_same_camera_background_only")),
            background_stream_policy=str(counterfactual.get("background_stream_policy", "continuous_per_env_view")),
        )


def validate_record(
    record: dict[str, Any],
    *,
    require_vlm: bool = False,
    expected_schema: ReplaySchema | None = None,
) -> None:
    counterfactual_k = int(record.get("counterfactual_K", 2))
    record_version = str(record.get("schema_version"))
    if record_version == K_VIEW_SCHEMA_VERSION:
        required = MULTI_VIEW_REQUIRED_RECORD_KEYS
        vlm_keys = MULTI_VIEW_VLM_RECORD_KEYS
        expected_version = K_VIEW_SCHEMA_VERSION
    elif counterfactual_k == 4:
        required = MULTI_VIEW_REQUIRED_RECORD_KEYS
        vlm_keys = MULTI_VIEW_VLM_RECORD_KEYS
        expected_version = ROBUST_FOUR_VIEW_SCHEMA_VERSION
    elif counterfactual_k == 6:
        required = K6_REQUIRED_RECORD_KEYS
        vlm_keys = K6_VLM_RECORD_KEYS
        expected_version = COUNTERFACTUAL_SCHEMA_VERSION
    elif counterfactual_k == 2:
        required = REQUIRED_RECORD_KEYS
        vlm_keys = VLM_RECORD_KEYS
        expected_version = TEMPORAL_SCHEMA_VERSION
    else:
        raise ValueError(f"unsupported counterfactual_K={counterfactual_k}")
    missing = [key for key in required if key not in record]
    if require_vlm:
        missing.extend([key for key in vlm_keys if key not in record])
    if missing:
        raise KeyError(f"V4 replay record is missing required keys: {missing}")
    if record.get("schema_version") != expected_version:
        raise ValueError(f"unexpected schema_version={record.get('schema_version')!r}")
    if expected_schema is not None:
        if record.get("schema_version") != expected_schema.schema_version:
            raise ValueError(
                f"record schema does not match runtime: {record.get('schema_version')!r} != "
                f"{expected_schema.schema_version!r}"
            )
        if counterfactual_k != int(expected_schema.counterfactual_K):
            raise ValueError(
                f"record K does not match runtime: {counterfactual_k} != {expected_schema.counterfactual_K}"
            )
    if record.get("temporal_contract") != TEMPORAL_CONTRACT:
        raise ValueError(f"unexpected temporal_contract={record.get('temporal_contract')!r}")
    if bool(record["is_first"]) and not np.allclose(np.asarray(record["action"], dtype=np.float32), 0.0):
        raise ValueError("is_first record must have zero incoming action")
    if record_version == K_VIEW_SCHEMA_VERSION or counterfactual_k in {4, 6}:
        raw = np.asarray(record["raw_b_views"])
        if raw.ndim != 4 or raw.shape[0] != counterfactual_k or raw.shape[-1] != 3:
            raise ValueError(f"raw_b_views must be [{counterfactual_k},H,W,3], got {raw.shape}")
        if require_vlm:
            scores = np.asarray(record["patch_scores"])
            features = np.asarray(record["patch_features"])
            if scores.ndim != 2 or scores.shape[0] != counterfactual_k:
                raise ValueError(f"patch_scores must be [{counterfactual_k},N], got {scores.shape}")
            if features.ndim != 3 or features.shape[:2] != scores.shape:
                raise ValueError(f"patch_features must be [{counterfactual_k},N,D], got {features.shape}")
