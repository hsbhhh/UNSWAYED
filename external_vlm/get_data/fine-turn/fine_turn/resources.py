from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from fine_turn.config import dump_yaml


FINE_TURN_ROOT = Path(__file__).resolve().parents[1]

ENV_KEYS = {
    "get_data_root": "GET_DATA_ROOT",
    "data_root": "BALL_IN_CUP_VLM_DATA_ROOT",
    "internvideo_root": "INTERNVIDEO_ROOT",
    "internvideo2_root": "INTERNVIDEO2_ROOT",
    "weights_root": "INTERNVIDEO2_WEIGHTS_ROOT",
    "clip_checkpoint": "INTERNVIDEO2_CLIP_CHECKPOINT",
    "stage2_checkpoint": "INTERNVIDEO2_STAGE2_CHECKPOINT",
    "internvl_checkpoint": "INTERNVL_CHECKPOINT",
    "output_root": "VLM_FINETUNE_OUTPUT_ROOT",
}


@dataclass(frozen=True)
class FineTuneResources:
    get_data_root: Path
    data_root: Path | None
    background_root: Path | None
    interaction_root: Path | None
    internvideo_root: Path | None
    internvideo2_root: Path | None
    weights_root: Path | None
    clip_checkpoint: Path | None
    stage2_checkpoint: Path | None
    internvl_checkpoint: Path | None
    output_root: Path

    def to_serializable(self) -> dict[str, str | None]:
        return {key: None if value is None else str(value) for key, value in asdict(self).items()}

    def require_data(self) -> None:
        if self.data_root is None or not self.data_root.exists():
            raise FileNotFoundError("SKIPPED: server dataset not available")

    def has_internvideo2_checkpoint(self) -> bool:
        return self.clip_checkpoint is not None and self.clip_checkpoint.exists()


def add_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--get-data-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--background-root", type=Path, default=None)
    parser.add_argument("--interaction-root", type=Path, default=None)
    parser.add_argument("--internvideo-root", type=Path, default=None)
    parser.add_argument("--internvideo2-root", type=Path, default=None)
    parser.add_argument("--weights-root", type=Path, default=None)
    parser.add_argument("--clip-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-checkpoint", type=Path, default=None)
    parser.add_argument("--internvl-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)


def resolve_resources(
    args: argparse.Namespace | None,
    config: Mapping[str, Any] | None,
    *,
    write_resolved: bool = True,
) -> FineTuneResources:
    config = config or {}
    resources_cfg = config.get("resources", {}) or {}
    if not isinstance(resources_cfg, Mapping):
        raise ValueError("resources section must be a mapping")
    args_map = vars(args) if args is not None else {}

    get_data_root = _resolve_one("get_data_root", args_map, resources_cfg)
    if get_data_root is None:
        get_data_root = FINE_TURN_ROOT.parent.resolve()

    data_root = _resolve_one("data_root", args_map, resources_cfg, base=get_data_root)
    background_root = _resolve_one("background_root", args_map, resources_cfg, base=get_data_root)
    interaction_root = _resolve_one("interaction_root", args_map, resources_cfg, base=get_data_root)
    if data_root is not None:
        background_root = background_root or (data_root / "pairs" / "background")
        interaction_root = interaction_root or (data_root / "pairs" / "interaction")

    internvideo_root = _resolve_one("internvideo_root", args_map, resources_cfg, base=get_data_root)
    internvideo2_root = _resolve_one("internvideo2_root", args_map, resources_cfg, base=get_data_root)
    if internvideo2_root is None and internvideo_root is not None:
        candidate = internvideo_root / "InternVideo2"
        if candidate.exists():
            internvideo2_root = candidate.resolve()

    output_root = _resolve_one("output_root", args_map, resources_cfg, base=FINE_TURN_ROOT)
    if output_root is None:
        output_root = FINE_TURN_ROOT / "outputs"

    resolved = FineTuneResources(
        get_data_root=get_data_root.resolve(),
        data_root=_resolve_optional(data_root),
        background_root=_resolve_optional(background_root),
        interaction_root=_resolve_optional(interaction_root),
        internvideo_root=_resolve_optional(internvideo_root),
        internvideo2_root=_resolve_optional(internvideo2_root),
        weights_root=_resolve_optional(_resolve_one("weights_root", args_map, resources_cfg, base=get_data_root)),
        clip_checkpoint=_resolve_optional(_resolve_one("clip_checkpoint", args_map, resources_cfg, base=get_data_root)),
        stage2_checkpoint=_resolve_optional(_resolve_one("stage2_checkpoint", args_map, resources_cfg, base=get_data_root)),
        internvl_checkpoint=_resolve_optional(_resolve_one("internvl_checkpoint", args_map, resources_cfg, base=get_data_root)),
        output_root=output_root.resolve(),
    )
    if write_resolved:
        dump_yaml(FINE_TURN_ROOT / "config_resolved.yaml", {"resources": resolved.to_serializable()})
    return resolved


def _resolve_one(
    name: str,
    args_map: Mapping[str, Any],
    resource_config: Mapping[str, Any],
    *,
    base: Path | None = None,
) -> Path | None:
    for raw in (args_map.get(name), os.environ.get(ENV_KEYS.get(name, "")), resource_config.get(name)):
        if raw is None or raw == "":
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (base or FINE_TURN_ROOT) / path
        return path.resolve()
    return None


def _resolve_optional(path: Path | None) -> Path | None:
    return None if path is None else Path(path).expanduser().resolve()

