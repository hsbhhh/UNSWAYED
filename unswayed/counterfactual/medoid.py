from __future__ import annotations

import torch


def select_counterfactual_medoid(
    raw_b_views: torch.Tensor,
    patch_features: torch.Tensor,
    center: torch.Tensor,
    patch_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one complete real B view independently for every `[B,T]` row."""

    features = torch.as_tensor(patch_features).detach().float()
    center = torch.as_tensor(center, device=features.device).detach().float()
    budget = torch.as_tensor(patch_budget, device=features.device).detach().float()
    costs = ((features - center.unsqueeze(2)).square().sum(dim=-1) * budget.unsqueeze(2)).sum(dim=-1)
    selected = costs.argmin(dim=2)
    images = _to_btkchw(raw_b_views).detach()
    gather_index = selected[..., None, None, None, None].expand(
        *selected.shape, 1, *images.shape[-3:]
    )
    target = images.gather(2, gather_index).squeeze(2)
    return target, selected.detach(), costs.detach()


def _to_btkchw(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 6:
        raise ValueError(f"raw_b_views must be [B,T,K,H,W,C] or [B,T,K,C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 1, 2, 5, 3, 4).contiguous()
    elif tensor.shape[3] != 3:
        raise ValueError(f"raw_b_views has no RGB channel dimension: {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and float(tensor.detach().amax().cpu()) > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)
