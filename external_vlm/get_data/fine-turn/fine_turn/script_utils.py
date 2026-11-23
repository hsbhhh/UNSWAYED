from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fine_turn.config import load_yaml
from fine_turn.datasets.background_pairs import BackgroundPairDataset, discover_background_pairs
from fine_turn.datasets.interaction_pairs import InteractionPairDataset, discover_interaction_pairs
from fine_turn.resources import add_resource_args, resolve_resources


def load_config_resources(argv_description: str):
    parser = argparse.ArgumentParser(description=argv_description)
    add_resource_args(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.mock:
        config.setdefault("model", {})["mock"] = True
    resources = resolve_resources(args, config, write_resolved=True)
    return args, config, resources


def dataset_roots(resources) -> tuple[Path | None, Path | None]:
    return resources.background_root, resources.interaction_root


def build_pair_datasets(resources, config: dict[str, Any]):
    model_cfg = config.get("model", {})
    num_frames = int(model_cfg.get("num_frames") or 8)
    image_size = int(model_cfg.get("image_size") or 224)
    background = BackgroundPairDataset(resources.background_root, num_frames=num_frames, image_size=image_size) if resources.background_root else None
    interaction = InteractionPairDataset(resources.interaction_root, num_frames=num_frames, image_size=image_size) if resources.interaction_root else None
    return background, interaction


def inspect_pair_counts(resources) -> dict[str, Any]:
    bg_root, int_root = dataset_roots(resources)
    bg = [] if bg_root is None else discover_background_pairs(bg_root)
    interaction = [] if int_root is None else discover_interaction_pairs(int_root)
    return {
        "background_pairs": len(bg),
        "interaction_pairs": len(interaction),
        "background_root": None if bg_root is None else str(bg_root),
        "interaction_root": None if int_root is None else str(int_root),
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))

