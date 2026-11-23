from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PSABResult:
    gradients: list[torch.Tensor | None]
    dreamer_norm: torch.Tensor
    auxiliary_norm: torch.Tensor
    cosine: torch.Tensor
    conflict: torch.Tensor
    safe_aux_norm: torch.Tensor
    dreamer_safe_dot: torch.Tensor
    alpha: torch.Tensor
    budgeted_aux_norm: torch.Tensor
    merged_norm: torch.Tensor


def merge_primary_safe_gradients(
    dreamer: list[torch.Tensor | None],
    auxiliary: list[torch.Tensor | None],
    *,
    epsilon: float,
) -> PSABResult:
    if len(dreamer) != len(auxiliary):
        raise ValueError("Dreamer and auxiliary gradient lists must have equal length")
    reference = next((item for item in dreamer + auxiliary if item is not None), torch.zeros(()))
    device = reference.device
    dot = torch.zeros((), device=device)
    dreamer_sq = torch.zeros((), device=device)
    auxiliary_sq = torch.zeros((), device=device)
    for primary, aux in zip(dreamer, auxiliary, strict=True):
        if primary is not None:
            dreamer_sq = dreamer_sq + primary.detach().float().square().sum()
        if aux is not None:
            auxiliary_sq = auxiliary_sq + aux.detach().float().square().sum()
        if primary is not None and aux is not None:
            dot = dot + (primary.detach().float() * aux.detach().float()).sum()
    dreamer_norm = dreamer_sq.sqrt()
    auxiliary_norm = auxiliary_sq.sqrt()
    cosine = dot / (dreamer_norm * auxiliary_norm + float(epsilon))
    conflict = dot < 0
    coefficient = torch.where(conflict, dot / (dreamer_sq + float(epsilon)), torch.zeros_like(dot))
    safe = []
    safe_sq = torch.zeros((), device=device)
    safe_dot = torch.zeros((), device=device)
    for primary, aux in zip(dreamer, auxiliary, strict=True):
        if aux is None:
            safe.append(None)
            continue
        value = aux.detach().clone()
        if primary is not None:
            value = value - coefficient.to(value.dtype) * primary.detach()
        safe.append(value)
        safe_sq = safe_sq + value.float().square().sum()
        if primary is not None:
            safe_dot = safe_dot + (primary.detach().float() * value.float()).sum()
    safe_norm = safe_sq.sqrt()
    alpha = torch.minimum(
        torch.ones((), device=device),
        dreamer_norm / (safe_norm + float(epsilon)),
    )
    merged = []
    merged_sq = torch.zeros((), device=device)
    for primary, safe_aux in zip(dreamer, safe, strict=True):
        if primary is None and safe_aux is None:
            merged.append(None)
            continue
        if primary is None:
            value = alpha.to(safe_aux.dtype) * safe_aux
        elif safe_aux is None:
            value = primary.detach().clone()
        else:
            value = primary.detach() + alpha.to(safe_aux.dtype) * safe_aux
        merged.append(value)
        merged_sq = merged_sq + value.float().square().sum()
    return PSABResult(
        gradients=merged,
        dreamer_norm=dreamer_norm.detach(),
        auxiliary_norm=auxiliary_norm.detach(),
        cosine=cosine.detach(),
        conflict=conflict.detach(),
        safe_aux_norm=safe_norm.detach(),
        dreamer_safe_dot=safe_dot.detach(),
        alpha=alpha.detach(),
        budgeted_aux_norm=(alpha * safe_norm).detach(),
        merged_norm=merged_sq.sqrt().detach(),
    )
