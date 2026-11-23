from __future__ import annotations

import torch


class RelationHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = int(output_dim or input_dim)
        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(),
            torch.nn.Linear(input_dim, output_dim),
        )

    def forward(self, global_embedding, temporal_tokens=None):
        import torch.nn.functional as F

        if temporal_tokens is not None:
            x = global_embedding + temporal_tokens.mean(dim=1)
        else:
            x = global_embedding
        return F.normalize(self.module(x), dim=-1)
