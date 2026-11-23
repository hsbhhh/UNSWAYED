from __future__ import annotations


def normalize_future_effect(values):
    import torch

    if values is None:
        return None
    values = torch.as_tensor(values, dtype=torch.float32)
    return values.abs() / values.abs().clamp_min(1e-6).max().clamp_min(1.0)


def interaction_relation_loss(anchor_rel, paired_rel, future_effect=None, *, margin: float = 0.2):
    import torch
    import torch.nn.functional as F

    distance = 1.0 - F.cosine_similarity(anchor_rel, paired_rel, dim=-1)
    weights = normalize_future_effect(future_effect)
    if weights is None:
        weights = torch.ones_like(distance)
    weights = weights.to(device=distance.device, dtype=distance.dtype)
    loss = weights * F.relu(float(margin) - distance)
    return loss.mean(), {"interaction_distance": distance.detach().mean()}


def triplet_ranking_loss(anchor_z, bg_z, int_z, future_effect=None, *, margin: float = 0.2):
    import torch
    import torch.nn.functional as F

    d_bg = 1.0 - F.cosine_similarity(anchor_z, bg_z, dim=-1)
    d_int = 1.0 - F.cosine_similarity(anchor_z, int_z, dim=-1)
    weights = normalize_future_effect(future_effect)
    if weights is None:
        weights = torch.ones_like(d_bg)
    weights = weights.to(device=d_bg.device, dtype=d_bg.dtype)
    loss = weights * F.relu(float(margin) + d_bg - d_int)
    return loss.mean(), {"triplet_d_bg": d_bg.detach().mean(), "triplet_d_int": d_int.detach().mean()}

