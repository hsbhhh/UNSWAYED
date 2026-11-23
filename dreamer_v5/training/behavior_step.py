from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dreamer.optim import clip_grad_agc_
from ..dreamer.tools import compute_global_norm


@dataclass
class BehaviorStepResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def behavior_train_step(
    behavior,
    world_model,
    world_output,
    batch: dict[str, torch.Tensor],
    optimizer,
    config,
    *,
    update_index: int = 0,
) -> BehaviorStepResult:
    if not bool(config.behavior.enabled):
        zero = torch.zeros((), device=world_output.deter.device)
        return BehaviorStepResult(zero, {"behavior/skipped": zero.detach()})
    behavior.train()
    previous_diagnostic_state = bool(
        getattr(behavior, "boundary_diagnostic_enabled", False)
    )
    diagnostic_origin = int(getattr(behavior, "boundary_diagnostic_origin", 0))
    behavior.boundary_diagnostic_enabled = (
        int(update_index) + 1 - diagnostic_origin
    ) % 100 == 0
    try:
        result = behavior.compute_losses(
            world_model,
            world_output.stoch.detach(),
            world_output.deter.detach(),
            batch,
        )
    finally:
        behavior.boundary_diagnostic_enabled = previous_diagnostic_state
    params = list(behavior.trainable_parameters())
    optimizer.zero_grad(set_to_none=True)
    _require_finite_behavior_objective(result)
    snapshots = _snapshot_behavior(behavior, config, update_index)
    result.total_loss.backward()
    metrics = dict(result.metrics)
    _require_finite_metrics(metrics, prefix="behavior metric")
    global_norm_before_agc = compute_global_norm([p.grad for p in params]).to(
        result.total_loss.device
    )
    _require_finite_gradient_norm(
        params,
        global_norm_before_agc,
        stage="behavior gradients before AGC",
    )
    metrics["grad/behavior_global_norm_before_agc"] = global_norm_before_agc.detach()
    branch_norms_before_agc = _actor_branch_gradient_norms(behavior)
    for name, value in branch_norms_before_agc.items():
        metrics[f"grad/{name}_before_agc"] = value.detach()
    if float(config.optimizer.agc) > 0.0:
        clip_grad_agc_(params, float(config.optimizer.agc), float(config.optimizer.pmin))
    global_norm_after_agc = compute_global_norm([p.grad for p in params]).to(
        result.total_loss.device
    )
    _require_finite_gradient_norm(
        params,
        global_norm_after_agc,
        stage="behavior gradients after AGC",
    )
    metrics["grad/behavior_global_norm_after_agc"] = global_norm_after_agc.detach()
    branch_norms_after_agc = _actor_branch_gradient_norms(behavior)
    for name, value in branch_norms_after_agc.items():
        metrics[f"grad/{name}_after_agc"] = value.detach()
        metrics[f"grad/{name}_agc_ratio"] = (
            value / branch_norms_before_agc[name].clamp_min(1.0e-30)
        ).detach()
    optimizer.step()
    metrics.update(_behavior_update_metrics(behavior, snapshots))
    _require_finite_metrics(metrics, prefix="post-step behavior metric")
    behavior.update_slow_value()
    return BehaviorStepResult(result.total_loss.detach(), metrics)


def _actor_branch_gradient_norms(behavior) -> dict[str, torch.Tensor]:
    """Read-only actor branch gradient norms for E1 behavior instrumentation."""
    last = behavior.actor.last
    action_dim = int(last.out_features // 2)
    candidates = [
        parameter
        for name, parameter in behavior.actor.named_parameters()
        if name.startswith("mlp.") and name.endswith("weight") and parameter.ndim == 2
    ]
    if not candidates:
        raise RuntimeError("actor penultimate linear layer not found")

    def norm(tensors: list[torch.Tensor | None]) -> torch.Tensor:
        reference = next(behavior.parameters())
        total = torch.zeros((), device=reference.device, dtype=torch.float32)
        for tensor in tensors:
            if tensor is not None:
                total = total + tensor.detach().float().square().sum()
        return total.sqrt()

    weight_grad = last.weight.grad
    bias_grad = last.bias.grad
    return {
        "mean_head": norm([
            None if weight_grad is None else weight_grad[:action_dim],
            None if bias_grad is None else bias_grad[:action_dim],
        ]),
        "std_head": norm([
            None if weight_grad is None else weight_grad[action_dim : action_dim * 2],
            None if bias_grad is None else bias_grad[action_dim : action_dim * 2],
        ]),
        "last_layer": norm([weight_grad, bias_grad]),
        "penultimate": norm([candidates[-1].grad]),
    }


def _require_finite_behavior_objective(result) -> None:
    for attribute in ("actor_loss", "value_loss", "repval_loss", "total_loss"):
        value = getattr(result, attribute)
        _require_finite_tensor(f"behavior {attribute}", value)


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    detached = value.detach()
    if not (detached.is_floating_point() or detached.is_complex()):
        return
    finite = torch.isfinite(detached)
    if bool(finite.all()):
        return
    nonfinite = int((~finite).sum().cpu())
    raise FloatingPointError(
        f"non-finite {name}: {nonfinite}/{detached.numel()} values; "
        "optimizer.step was not called"
    )


def _require_finite_metrics(metrics: dict[str, torch.Tensor], *, prefix: str) -> None:
    for name, value in metrics.items():
        _require_finite_tensor(f"{prefix} {name}", value)


def _require_finite_gradient_norm(
    parameters: list[torch.nn.Parameter],
    global_norm: torch.Tensor,
    *,
    stage: str,
) -> None:
    if bool(torch.isfinite(global_norm.detach()).all()):
        return
    offenders = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient.detach()).all()):
            offenders.append(index)
    suffix = (
        f"; non-finite parameter-gradient indices={offenders[:16]}"
        if offenders
        else "; all elements were finite but their global norm overflowed"
    )
    raise FloatingPointError(
        f"non-finite {stage} global norm{suffix}; optimizer.step was not called"
    )


def _snapshot_behavior(behavior, config, update_index: int):
    counterfactual = config.get("counterfactual", {})
    capc = config.get("capc", {})
    if not (bool(counterfactual.get("enabled", False)) or bool(capc.get("enabled", False))):
        return None
    if bool(capc.get("enabled", False)):
        interval = int(capc.get("logging", {}).get("expensive_interval", 2000))
    else:
        interval = int(counterfactual.diagnostics.parameter_update_interval)
    index = int(update_index)
    period = max(1, int(interval))
    if not (index == 0 or (index + 1) % period == 0):
        return None
    return {
        "actor": [(param, param.detach().clone()) for param in behavior.actor.parameters()],
        "critic": [(param, param.detach().clone()) for param in behavior.value.parameters()],
    }


def _behavior_update_metrics(behavior, snapshots):
    if snapshots is None:
        return {}
    device = next(behavior.parameters()).device
    metrics = {}
    for name, rows in snapshots.items():
        delta_sq = torch.zeros((), device=device)
        parameter_sq = torch.zeros((), device=device)
        for param, before in rows:
            delta_sq = delta_sq + (param.detach() - before).float().square().sum()
            parameter_sq = parameter_sq + before.float().square().sum()
        metrics[f"update_ratio/{name}"] = (
            delta_sq.sqrt() / (parameter_sq.sqrt() + 1.0e-12)
        ).detach()
        metrics[f"update_norm/{name}"] = delta_sq.sqrt().detach()
        metrics[f"parameter_norm/{name}"] = parameter_sq.sqrt().detach()
    return metrics
