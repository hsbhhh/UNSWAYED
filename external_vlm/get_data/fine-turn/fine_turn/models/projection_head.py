from __future__ import annotations

import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, hidden_dim: int | None = None):
        super().__init__()
        output_dim = int(output_dim or input_dim)
        hidden_dim = int(hidden_dim or max(input_dim, output_dim))
        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        import torch.nn.functional as F

        return F.normalize(self.module(x), dim=-1)
