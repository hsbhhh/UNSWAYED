from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fine_turn.datasets.metadata_adapter import PairSample


@dataclass(frozen=True)
class MatchedTriplet:
    anchor_key: str
    background: PairSample
    interaction: PairSample


def anchor_match_key(sample: PairSample) -> str | None:
    if sample.physics_trajectory_id:
        return sample.physics_trajectory_id
    if sample.episode_id:
        return sample.episode_id
    return None


def match_triplets(background: Iterable[PairSample], interaction: Iterable[PairSample]) -> list[MatchedTriplet]:
    bg_by_key: dict[str, PairSample] = {}
    for sample in background:
        key = anchor_match_key(sample)
        if key and key not in bg_by_key:
            bg_by_key[key] = sample
    triplets = []
    for sample in interaction:
        key = anchor_match_key(sample)
        if key and key in bg_by_key:
            triplets.append(MatchedTriplet(key, bg_by_key[key], sample))
    return triplets

