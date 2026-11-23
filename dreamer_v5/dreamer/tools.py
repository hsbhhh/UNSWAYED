from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init


def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.int32)


def weight_init_(m: nn.Module, fan_type: str = "in") -> None:
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return
    weight = getattr(m, "weight", None)
    if weight is None or int(weight.numel()) == 0:
        return
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)
    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


def rpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    for _ in range(int(pad)):
        x = x.unsqueeze(-1)
    return x


def tensorstats(tensor: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    value = tensor.detach().float()
    return {
        f"{prefix}_mean": value.mean(),
        f"{prefix}_std": value.std(),
        f"{prefix}_min": value.amin(),
        f"{prefix}_max": value.amax(),
    }


def compute_global_norm(tensors) -> torch.Tensor:
    flat = [tensor.reshape(-1) for tensor in tensors if tensor is not None]
    if not flat:
        return torch.tensor(0.0)
    return torch.linalg.norm(torch.cat(flat), ord=2)
