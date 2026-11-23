from __future__ import annotations

from dataclasses import dataclass

import torch

from ..counterfactual.psab import merge_primary_safe_gradients
from ..dreamer.optim import clip_grad_agc_
from ..dreamer.tools import compute_global_norm
from .losses import compute_world_model_loss_components


@dataclass
class WorldModelStepResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    output: object


def world_model_train_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer,
    config,
    *,
    update_index: int = 0,
) -> WorldModelStepResult:
    model.train()
    components = compute_world_model_loss_components(model, batch, update_index=update_index)
    metrics = dict(components.metrics)
    optimizer.zero_grad(set_to_none=True)
    grad_params = list(model.world_parameters())
    _require_finite_world_objective(components, config)
    _require_finite_metrics(metrics, prefix="world-model metric")
    if components.formal_counterfactual:
        _assign_counterfactual_gradients(model, components, config, metrics)
    else:
        components.total.backward()
    global_norm_before_agc = compute_global_norm(
        [param.grad for param in grad_params]
    ).to(components.total.device)
    _require_finite_gradient_norm(
        grad_params,
        global_norm_before_agc,
        stage="world-model gradients before AGC",
    )
    metrics["grad/world_model_global_norm_before_agc"] = global_norm_before_agc.detach()
    if float(config.optimizer.agc) > 0.0:
        clip_grad_agc_(grad_params, float(config.optimizer.agc), float(config.optimizer.pmin))
    global_norm_after_agc = compute_global_norm(
        [param.grad for param in grad_params]
    ).to(components.total.device)
    _require_finite_gradient_norm(
        grad_params,
        global_norm_after_agc,
        stage="world-model gradients after AGC",
    )
    metrics["grad/world_model_global_norm_after_agc"] = global_norm_after_agc.detach()
    optimizer.step()
    _require_finite_metrics(metrics, prefix="post-step world-model metric")
    return WorldModelStepResult(
        loss=components.total.detach(),
        metrics=metrics,
        output=components.output,
    )


def _require_finite_world_objective(components, config) -> None:
    """Fail closed before backward/optimizer state can consume invalid values."""

    critical_losses = {
        "world primary loss": components.primary,
        "VGC/CVGC loss": components.cvgc,
        "MSIR loss": components.msir,
        "native world loss": components.native,
        "CAPC raw loss": components.capc_raw,
        "CAPC weighted loss": components.capc,
        "world total loss": components.total,
    }
    for name, value in critical_losses.items():
        _require_finite_tensor(name, value)

    if not bool(config.get("capc", {}).get("enabled", False)):
        return
    target = components.capc_artifacts.get("q_bar")
    if target is None:
        raise FloatingPointError(
            "CAPC is enabled but its q_bar target is missing; optimizer.step was not called"
        )
    _require_finite_tensor("CAPC q_bar target", target)
    if bool(target.requires_grad):
        raise FloatingPointError(
            "CAPC q_bar target is not stop-gradient; optimizer.step was not called"
        )


def _require_finite_metrics(metrics: dict[str, torch.Tensor], *, prefix: str) -> None:
    for name, value in metrics.items():
        _require_finite_tensor(f"{prefix} {name}", value)


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


def _assign_counterfactual_gradients(model, components, config, metrics: dict[str, torch.Tensor]) -> None:
    all_params = list(model.world_parameters())
    primary_grads = torch.autograd.grad(
        components.primary,
        all_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    primary_by_id = {id(param): grad for param, grad in zip(all_params, primary_grads, strict=True)}

    visual = list(model.visual_encoder_parameters())
    posterior = list(model.rssm_observation_parameters())
    predictor = list(model.patch_predictor_parameters())
    decoder = list(model.decoder_parameters()) if bool(config.model.msir.training_enabled) else []
    aux_params = _unique_parameters([*visual, *posterior, *predictor, *decoder])
    auxiliary_loss = components.cvgc + components.msir
    auxiliary_grads = torch.autograd.grad(
        auxiliary_loss,
        aux_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    auxiliary_by_id = {id(param): grad for param, grad in zip(aux_params, auxiliary_grads, strict=True)}

    epsilon = float(config.counterfactual.psab.epsilon)
    shared_ids: set[int] = set()
    for name, params in (("encoder", visual), ("rssm_posterior", posterior)):
        params = _unique_parameters(params)
        shared_ids.update(id(param) for param in params)
        result = merge_primary_safe_gradients(
            [primary_by_id.get(id(param)) for param in params],
            [auxiliary_by_id.get(id(param)) for param in params],
            epsilon=epsilon,
        )
        for param, gradient in zip(params, result.gradients, strict=True):
            param.grad = None if gradient is None else gradient.to(dtype=param.dtype)
        prefix = f"psab/{name}"
        metrics[f"{prefix}/dreamer_grad_norm"] = result.dreamer_norm
        metrics[f"{prefix}/aux_grad_norm"] = result.auxiliary_norm
        metrics[f"{prefix}/aux_dreamer_ratio"] = result.auxiliary_norm / (result.dreamer_norm + epsilon)
        metrics[f"{prefix}/cosine"] = result.cosine
        metrics[f"{prefix}/conflict_fraction"] = result.conflict.float()
        metrics[f"{prefix}/safe_aux_grad_norm"] = result.safe_aux_norm
        metrics[f"{prefix}/dreamer_safe_aux_dot"] = result.dreamer_safe_dot
        metrics[f"{prefix}/alpha"] = result.alpha
        metrics[f"{prefix}/budgeted_aux_grad_norm"] = result.budgeted_aux_norm
        metrics[f"{prefix}/post_psab_grad_norm"] = result.merged_norm

    exclusive_ids = {id(param) for param in [*predictor, *decoder]}
    for param in all_params:
        param_id = id(param)
        if param_id in shared_ids:
            continue
        gradient = auxiliary_by_id.get(param_id) if param_id in exclusive_ids else primary_by_id.get(param_id)
        param.grad = None if gradient is None else gradient.detach().clone().to(dtype=param.dtype)

    device = components.total.device
    metrics["grad/raw_dreamer_global_norm"] = compute_global_norm(primary_grads).to(device).detach()
    metrics["grad/raw_auxiliary_allowed_global_norm"] = compute_global_norm(auxiliary_grads).to(device).detach()
    metrics["grad/post_psab_global_norm"] = compute_global_norm(
        [param.grad for param in all_params]
    ).to(device).detach()
    metrics["grad/rssm_prior_direct_aux_grad_norm"] = torch.zeros((), device=device)
    metrics["grad/rssm_recurrent_direct_aux_grad_norm"] = torch.zeros((), device=device)
    metrics["grad/reward_head_direct_aux_grad_norm"] = torch.zeros((), device=device)
    metrics["grad/continue_head_direct_aux_grad_norm"] = torch.zeros((), device=device)
    metrics["grad/actor_direct_aux_grad_norm"] = torch.zeros((), device=device)
    metrics["grad/critic_direct_aux_grad_norm"] = torch.zeros((), device=device)


def _unique_parameters(parameters) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    out = []
    for param in parameters:
        if id(param) not in seen and param.requires_grad:
            seen.add(id(param))
            out.append(param)
    return out
