from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LaProp(Optimizer):
    """LaProp optimizer migrated into UNSWAYED runtime."""

    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1.0 - beta1) * group["lr"]
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1.0 - beta2)
                bias_correction1 = state["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0
                step_size = 1.0 / bias_correction1
                denom = exp_avg_sq.div(state["exp_avg_lr_2"]).sqrt_().add_(group["eps"])
                step_grad = grad / denom
                exp_avg.mul_(beta1).add_(step_grad, alpha=(1.0 - beta1) * group["lr"])
                param.data.add_(exp_avg, alpha=-step_size)
                if group["weight_decay"]:
                    param.data.add_(param.data, alpha=-float(group["weight_decay"]))
        return loss


def clip_grad_agc_(parameters: Tensor | Iterable[Tensor], clip: float, pmin: float) -> None:
    params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    for param in params:
        if param.grad is None:
            continue
        pnorm = torch.norm(param.detach(), p=2)
        gnorm = torch.norm(param.grad.detach(), p=2)
        upper = float(clip) * torch.maximum(torch.as_tensor(float(pmin), device=param.device), pnorm)
        scale = 1.0 / torch.maximum(torch.as_tensor(1.0, device=param.device), gnorm / upper.clamp_min(1e-12))
        param.grad.detach().mul_(scale)


@dataclass(frozen=True)
class OptimizerGroups:
    world: list[torch.nn.Parameter]
    behavior: list[torch.nn.Parameter]


def build_world_optimizer(model, config) -> LaProp:
    return LaProp(
        model.world_parameters(),
        lr=float(config.training.lr),
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        eps=float(config.optimizer.eps),
        weight_decay=float(config.optimizer.weight_decay),
    )


def build_behavior_optimizer(behavior, config) -> LaProp:
    return LaProp(
        behavior.trainable_parameters(),
        lr=float(config.behavior.lr),
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        eps=float(config.optimizer.eps),
        weight_decay=float(config.optimizer.weight_decay),
    )


def build_warmup_scheduler(optimizer: Optimizer, config) -> LambdaLR:
    warmup = int(config.optimizer.warmup_updates)

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup)) if warmup > 0 else 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def assert_parameter_isolation(world_params, behavior_params) -> None:
    world_ids = {id(param) for param in world_params}
    behavior_ids = {id(param) for param in behavior_params}
    overlap = world_ids & behavior_ids
    if overlap:
        raise RuntimeError(f"world and behavior optimizers share {len(overlap)} parameters")
