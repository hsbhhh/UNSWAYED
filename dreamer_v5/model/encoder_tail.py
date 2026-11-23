from __future__ import annotations

import torch
from torch import nn


class DreamerEncoderTail(nn.Module):
    """Small V4-owned encoder tail for A-branch observations."""

    def __init__(self, in_channels: int = 64, out_dim: int = 256):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, stem_feature: torch.Tensor) -> torch.Tensor:
        if stem_feature.ndim == 5:
            b, t = stem_feature.shape[:2]
            flat = stem_feature.reshape(b * t, *stem_feature.shape[-3:])
            out = self.net(flat)
            return out.reshape(b, t, -1)
        return self.net(stem_feature)
