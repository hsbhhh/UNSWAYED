from __future__ import annotations

from typing import Sequence


def pair_collate(batch: Sequence[dict]) -> dict:
    import torch

    masks = [item.get("task_mask") for item in batch]
    return {
        "samples": [item["sample"] for item in batch],
        "anchor": torch.stack([item["anchor"] for item in batch], dim=0),
        "paired": torch.stack([item["paired"] for item in batch], dim=0),
        "task_mask": None if any(mask is None for mask in masks) else torch.stack(masks, dim=0),
        "transform_spec": [item["transform_spec"] for item in batch],
    }

