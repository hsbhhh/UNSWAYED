from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PairSample:
    pair_id: str
    pair_type: str
    anchor_path: Path
    paired_path: Path
    task_mask_path: Path | None
    metadata_path: Path
    episode_id: str | None = None
    physics_trajectory_id: str | None = None
    davis_video_id: str | None = None
    intervention_type: str | None = None
    future_effect: float | None = None
    success_changed: bool | None = None
    split: str | None = None

    @property
    def group_key(self) -> str:
        for value in (self.physics_trajectory_id, self.episode_id, self.davis_video_id, self.intervention_type):
            if value:
                return str(value)
        return self.pair_id


def read_metadata(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def background_sample(pair_dir: Path, *, split: str | None = None) -> PairSample:
    pair_dir = Path(pair_dir)
    meta_path = pair_dir / "metadata.json"
    meta = read_metadata(meta_path)
    pair_id = str(meta.get("pair_id") or pair_dir.name)
    mask_path = pair_dir / "task_masks.npz"
    return PairSample(
        pair_id=pair_id,
        pair_type="background",
        anchor_path=pair_dir / "clean_frames.npz",
        paired_path=pair_dir / "distracted_frames.npz",
        task_mask_path=mask_path if mask_path.exists() else None,
        metadata_path=meta_path,
        episode_id=_optional_str(meta.get("episode_id")),
        physics_trajectory_id=_optional_str(meta.get("physics_trajectory_id")),
        davis_video_id=_optional_str(meta.get("davis_video_id")),
        split=split,
    )


def interaction_sample(pair_dir: Path, *, split: str | None = None) -> PairSample:
    pair_dir = Path(pair_dir)
    meta_path = pair_dir / "metadata.json"
    meta = read_metadata(meta_path)
    pair_id = str(meta.get("pair_id") or pair_dir.name)
    return PairSample(
        pair_id=pair_id,
        pair_type="interaction",
        anchor_path=pair_dir / "anchor_frames.npz",
        paired_path=pair_dir / "intervention_frames.npz",
        task_mask_path=None,
        metadata_path=meta_path,
        episode_id=_optional_str(meta.get("episode_id")),
        physics_trajectory_id=_optional_str(meta.get("physics_trajectory_id")),
        intervention_type=_optional_str(meta.get("intervention_type")),
        future_effect=_optional_float(meta.get("future_effect")),
        success_changed=_optional_bool(meta.get("success_changed")),
        split=split,
    )


def sample_to_json(sample: PairSample) -> dict[str, Any]:
    data = dict(sample.__dict__)
    for key in ("anchor_path", "paired_path", "task_mask_path", "metadata_path"):
        data[key] = None if data[key] is None else str(data[key])
    return data


def sample_from_json(row: Mapping[str, Any]) -> PairSample:
    data = dict(row)
    for key in ("anchor_path", "paired_path", "task_mask_path", "metadata_path"):
        data[key] = None if data.get(key) is None else Path(data[key])
    return PairSample(**data)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)

