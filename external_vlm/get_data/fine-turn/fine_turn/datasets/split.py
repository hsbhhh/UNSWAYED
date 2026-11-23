from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fine_turn.datasets.metadata_adapter import PairSample, sample_from_json, sample_to_json


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


def assign_group_split(group_key: str, ratios: SplitRatios = SplitRatios()) -> str:
    total = ratios.train + ratios.validation + ratios.test
    if total <= 0:
        raise ValueError("Split ratios must be positive")
    value = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    train_cut = ratios.train / total
    validation_cut = (ratios.train + ratios.validation) / total
    if value < train_cut:
        return "train"
    if value < validation_cut:
        return "validation"
    return "test"


def assign_splits(samples: Iterable[PairSample], ratios: SplitRatios = SplitRatios()) -> list[PairSample]:
    assigned = []
    group_to_split: dict[str, str] = {}
    for sample in samples:
        split = group_to_split.setdefault(sample.group_key, assign_group_split(sample.group_key, ratios))
        assigned.append(PairSample(**{**sample.__dict__, "split": split}))
    return assigned


def write_splits(samples: Iterable[PairSample], split_root: Path) -> None:
    split_root = Path(split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "validation": [], "test": []}
    for sample in samples:
        buckets.setdefault(sample.split or "train", []).append(sample)
    for split, rows in buckets.items():
        with (split_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in rows:
                handle.write(json.dumps(sample_to_json(sample), sort_keys=True) + "\n")


def read_split(path: Path) -> list[PairSample]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(sample_from_json(json.loads(line)))
    return samples

