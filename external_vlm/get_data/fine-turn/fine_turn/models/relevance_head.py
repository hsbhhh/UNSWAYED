from __future__ import annotations

import torch


class RelevanceHead(torch.nn.Module):
    def __init__(self, token_dim: int, text_dim: int | None = None, hidden_dim: int | None = None):
        super().__init__()
        text_dim = int(text_dim or token_dim)
        hidden_dim = int(hidden_dim or token_dim)
        self.token_proj = torch.nn.Linear(token_dim, hidden_dim)
        self.text_proj = torch.nn.Linear(text_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens, text_embedding=None, *, token_grid_shape=None):
        hidden = self.token_proj(patch_tokens)
        if text_embedding is not None:
            text = self.text_proj(text_embedding).unsqueeze(1)
            hidden = hidden + text
        scores = self.out(torch.tanh(hidden)).squeeze(-1)
        if token_grid_shape is None:
            return scores
        bsz = scores.shape[0]
        return scores.reshape(bsz, *tuple(int(x) for x in token_grid_shape))
