from __future__ import annotations

import torch
from torch import nn


class CanonicalStem(nn.Module):
    """Small visual stem for normalized online observations.

    It is intentionally separate from the frozen teacher. The EMA target stem is
    initialized from this module by `EMAUpdater`. Input is `[0,1]`; the first
    convolution receives the R2-Dreamer-aligned centered range `[-0.5,0.5]`.
    """

    def __init__(self, in_channels: int = 3, channels: int = 64):
        super().__init__()
        self.out_channels = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 4, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        flat, shape = _flatten_bt(image)
        feat = self.net(flat.float() - 0.5)
        return feat.reshape(*shape, *feat.shape[-3:])


def _flatten_bt(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if image.ndim == 5:
        batch_shape = image.shape[:2]
        return image.reshape(-1, *image.shape[-3:]), tuple(batch_shape)
    if image.ndim == 4:
        return image, tuple()
    raise ValueError(f"expected [B,T,C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
