from __future__ import annotations

from functools import partial

import torch
from torch import nn

from . import distributions as dists
from .tools import weight_init_


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer used by the copied DreamerV3 RSSM."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class MLP(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, layers: int = 2, act: str = "SiLU"):
        super().__init__()
        activation = getattr(nn, str(act))
        parts: list[nn.Module] = []
        dim = int(inp_dim)
        for _ in range(int(layers)):
            parts.extend([nn.Linear(dim, int(hidden_dim)), nn.RMSNorm(int(hidden_dim), eps=1e-4), activation()])
            dim = int(hidden_dim)
        self.layers = nn.Sequential(*parts)
        self.out_dim = dim
        self.apply(weight_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        outscale: float = 0.01,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ):
        super().__init__()
        self.mlp = MLP(inp_dim, hidden_dim, layers=hidden_layers)
        self.last = nn.Linear(self.mlp.out_dim, int(action_dim) * 2)
        self.hidden_layers = int(hidden_layers)
        self.outscale = float(outscale)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.last.apply(weight_init_)
        with torch.no_grad():
            self.last.weight.mul_(self.outscale)

    def forward(self, x: torch.Tensor):
        return dists.bounded_normal(self.raw_parameters(x), self.min_std, self.max_std)

    def raw_parameters(self, x: torch.Tensor) -> torch.Tensor:
        return self.last(self.mlp(x))


class ValueHead(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        bin_num: int = 255,
        outscale: float = 0.0,
    ):
        super().__init__()
        self.mlp = MLP(inp_dim, hidden_dim, layers=hidden_layers)
        self.last = nn.Linear(self.mlp.out_dim, int(bin_num))
        self.hidden_layers = int(hidden_layers)
        self.bin_num = int(bin_num)
        self.outscale = float(outscale)
        self._dist = partial(dists.symexp_twohot, bin_num=self.bin_num)
        self.last.apply(weight_init_)
        with torch.no_grad():
            self.last.weight.mul_(self.outscale)

    def forward(self, x: torch.Tensor):
        return self._dist(self.logits(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.last(self.mlp(x))


class ReturnEMA(nn.Module):
    def __init__(self, device: torch.device | str, alpha: float = 1.0e-2):
        super().__init__()
        self.alpha = float(alpha)
        self.register_buffer("range", torch.tensor([0.05, 0.95], dtype=torch.float32, device=torch.device(device)))
        self.register_buffer("ema_vals", torch.zeros(2, dtype=torch.float32, device=torch.device(device)))

    def forward(self, x: torch.Tensor, *, update: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        should_update = self.training if update is None else bool(update)
        if should_update:
            x_quantile = torch.quantile(torch.flatten(x.detach().float()), self.range)
            self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1.0 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
