from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Tie-stable empirical rank in `[0,1]` along the patch dimension."""

    values = torch.as_tensor(values).float()
    count = int(values.shape[-1])
    if count <= 1:
        return torch.ones_like(values)
    lhs = values.unsqueeze(-1)
    rhs = values.unsqueeze(-2)
    less = (lhs > rhs).float().sum(dim=-1)
    equal_other = (lhs == rhs).float().sum(dim=-1) - 1.0
    return ((less + 0.5 * equal_other) / float(count - 1)).clamp(0.0, 1.0)


class TaskRelevanceEstimator(nn.Module):
    """Detached Dreamer-gradient sensitivity with checkpointed spatial EMA."""

    def __init__(self, patch_count: int):
        super().__init__()
        self.patch_count = int(patch_count)
        self.grid_size = int(round(self.patch_count ** 0.5))
        if self.grid_size * self.grid_size != self.patch_count:
            raise ValueError("TaskRelevanceEstimator requires a square patch grid")
        self.register_buffer("ema", torch.ones(self.patch_count, dtype=torch.float32))
        self.register_buffer("ema_valid", torch.zeros((), dtype=torch.bool))
        self.register_buffer("update_counter", torch.zeros((), dtype=torch.long))

    def forward(
        self,
        primary_loss: torch.Tensor,
        spatial_embedding: torch.Tensor,
        *,
        mode: str,
        update_period: int,
        ema_momentum: float,
    ) -> torch.Tensor:
        spatial = torch.as_tensor(spatial_embedding)
        if spatial.ndim != 5:
            raise ValueError(f"spatial embedding must be [B,T,C,H,W], got {tuple(spatial.shape)}")
        batch, time = spatial.shape[:2]
        due = str(mode) == "exact" or int(self.update_counter.item()) % max(1, int(update_period)) == 0
        can_differentiate = bool(torch.is_grad_enabled() and primary_loss.requires_grad and spatial.requires_grad)
        if due and can_differentiate:
            gradient = torch.autograd.grad(
                primary_loss,
                spatial,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            sensitivity = gradient.detach().float().square().sum(dim=2).sqrt()
            sensitivity = F.adaptive_avg_pool2d(
                sensitivity.reshape(batch * time, 1, *sensitivity.shape[-2:]),
                (self.grid_size, self.grid_size),
            ).reshape(batch, time, self.patch_count)
            with torch.no_grad():
                current_mean = sensitivity.mean(dim=(0, 1))
                if not bool(self.ema_valid):
                    self.ema.copy_(current_mean)
                    self.ema_valid.fill_(True)
                    smoothed = sensitivity
                else:
                    previous = self.ema.clone()
                    momentum = float(ema_momentum)
                    self.ema.mul_(momentum).add_(current_mean, alpha=1.0 - momentum)
                    smoothed = momentum * previous.view(1, 1, -1) + (1.0 - momentum) * sensitivity
        else:
            smoothed = self.ema.view(1, 1, -1).expand(batch, time, -1)
        # Debug/evaluation calls are read-only. Only formal training advances
        # the checkpointed periodic-update schedule.
        if self.training:
            self.update_counter.add_(1)
        return smoothed.detach()


class NormalizedPatchBudget(nn.Module):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, stability: torch.Tensor, task_relevance: torch.Tensor) -> torch.Tensor:
        stability = torch.as_tensor(stability).detach().float().clamp(0.0, 1.0)
        relevance = torch.as_tensor(task_relevance, device=stability.device).detach().float().clamp(0.0, 1.0)
        unnormalized = stability * relevance + self.epsilon
        budget = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        if not bool(torch.isfinite(budget).all()):
            raise FloatingPointError("normalized patch budget contains non-finite values")
        return budget.detach()
