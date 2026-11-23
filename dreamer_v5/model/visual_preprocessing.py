from __future__ import annotations

import torch


def preprocess_observation(value: torch.Tensor, *, name: str = "observation", eps: float = 1.0e-6) -> torch.Tensor:
    """Convert an RGB observation to formal encoder input [B,T,C,H,W] in [0,1]."""

    tensor = _to_btchw(value, name=name)
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32) / 255.0
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be uint8 [0,255] or floating point [0,1], got {tensor.dtype}")
    tensor = tensor.to(dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if tensor.numel():
        observed_min = float(tensor.detach().amin().cpu())
        observed_max = float(tensor.detach().amax().cpu())
        if observed_min < -float(eps) or observed_max > 1.0 + float(eps):
            raise ValueError(
                f"floating {name} must already be normalized to [0,1]; "
                f"observed range=[{observed_min:.6g},{observed_max:.6g}]"
            )
    return tensor


def observation_storage_range(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.as_tensor(value)
    if not tensor.numel():
        zero = torch.zeros((), dtype=torch.float32, device=tensor.device)
        return zero, zero
    numeric = tensor.detach().to(dtype=torch.float32)
    return numeric.amin(), numeric.amax()


def _to_btchw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 5 and tensor.shape[2] == 3:
        return tensor
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3).contiguous()
    if tensor.ndim == 4 and tensor.shape[1] == 3:
        return tensor[:, None]
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        return tensor.permute(0, 3, 1, 2).contiguous()[:, None]
    raise ValueError(
        f"expected {name} as [B,T,C,H,W], [B,T,H,W,C], [B,C,H,W], or [B,H,W,C], "
        f"got {tuple(tensor.shape)}"
    )
