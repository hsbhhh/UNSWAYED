from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fine_turn.losses.background_consistency import background_consistency_loss
from fine_turn.losses.interaction_ranking import interaction_relation_loss, triplet_ranking_loss
from fine_turn.losses.region_supervision import region_loss


@dataclass
class TotalLoss:
    background_weight: float = 1.0
    background_map_weight: float = 0.5
    interaction_weight: float = 0.25
    interaction_rank_weight: float = 0.5
    region_weight: float = 2.0
    dice_weight: float = 0.5
    rank_margin: float = 0.2
    interaction_margin: float = 0.2

    @classmethod
    def from_config(cls, config: Mapping) -> "TotalLoss":
        return cls(**{key: float(config.get(key, getattr(cls, key))) for key in cls.__dataclass_fields__})

    def background(self, anchor_z, paired_z, anchor_map=None, paired_map=None, mask=None):
        loss, metrics = background_consistency_loss(
            anchor_z,
            paired_z,
            anchor_map,
            paired_map,
            map_weight=self.background_map_weight,
        )
        if mask is not None and anchor_map is not None:
            region, region_metrics = region_loss(anchor_map, mask, dice_weight=self.dice_weight)
            metrics.update(region_metrics)
            loss = loss * self.background_weight + region * self.region_weight
        else:
            loss = loss * self.background_weight
        return loss, metrics

    def interaction(self, anchor_rel, paired_rel, future_effect=None):
        loss, metrics = interaction_relation_loss(
            anchor_rel,
            paired_rel,
            future_effect,
            margin=self.interaction_margin,
        )
        return loss * self.interaction_weight, metrics

    def triplet(self, anchor_z, bg_z, int_z, future_effect=None):
        loss, metrics = triplet_ranking_loss(anchor_z, bg_z, int_z, future_effect, margin=self.rank_margin)
        return loss * self.interaction_rank_weight, metrics

