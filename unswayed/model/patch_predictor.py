from __future__ import annotations

import torch
from torch import nn


class BeliefPatchFeatureHead(nn.Module):
    """Lightweight A-branch head for K=0 VLM patch feature prediction."""

    def __init__(self, belief_dim: int, *, patch_count: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.patch_count = int(patch_count)
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(int(belief_dim)),
            nn.Linear(int(belief_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.patch_count * self.feature_dim),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        batch, time = belief.shape[:2]
        out = self.net(belief)
        return out.reshape(batch, time, self.patch_count, self.feature_dim)
