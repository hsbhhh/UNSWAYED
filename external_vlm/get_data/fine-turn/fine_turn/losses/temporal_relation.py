from __future__ import annotations


def temporal_smoothness_loss(temporal_tokens):
    import torch

    if temporal_tokens is None or temporal_tokens.shape[1] < 2:
        return torch.zeros(())
    return torch.mean((temporal_tokens[:, 1:] - temporal_tokens[:, :-1]) ** 2)

