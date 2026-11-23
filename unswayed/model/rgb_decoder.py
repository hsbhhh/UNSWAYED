from __future__ import annotations

import torch
from torch import nn


class TaskRGBDecoder(nn.Module):
    def __init__(self, belief_dim: int, image_size: int = 64):
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Linear(belief_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128 * 8 * 8),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        flat = belief.reshape(-1, belief.shape[-1])
        hid = self.net(flat).reshape(flat.shape[0], 128, 8, 8)
        image = self.deconv(hid)
        image = image[..., : self.image_size, : self.image_size]
        return image.reshape(*belief.shape[:-1], *image.shape[-3:])
