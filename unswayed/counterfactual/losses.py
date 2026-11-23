from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..semantic.semantic_lowrank_gaussian import laplacian_pyramid, prepare_reconstruction_image
from .medoid import select_counterfactual_medoid
from .nuisance import apply_soft_nuisance_metric, geometric_median, nuisance_statistics
from .relevance import NormalizedPatchBudget, percentile_rank


def counterfactual_vgc_loss(
    output,
    config,
    batch: dict[str, torch.Tensor],
    task_sensitivity: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = config.counterfactual
    robust_cfg = cfg.robust_vgc
    prediction = torch.as_tensor(output.vgc_patch_prediction).float()
    features = _to_btknd(_require(batch, "patch_features").to(prediction.device)).detach().float()
    expected = (
        prediction.shape[0],
        prediction.shape[1],
        int(cfg.views),
        prediction.shape[2],
        prediction.shape[3],
    )
    if tuple(features.shape) != expected:
        raise ValueError(f"K-view feature contract mismatch: expected={expected}, got={tuple(features.shape)}")
    valid = _valid_mask(batch, prediction.shape[:2], prediction.device)
    features = torch.where(
        valid[..., None, None, None], features, torch.zeros_like(features)
    )
    if not bool(torch.isfinite(features[valid]).all()):
        raise FloatingPointError("valid K-view patch features contain NaN or Inf")
    center_cfg = cfg.center
    if str(center_cfg.mode) == "geometric_median":
        center = geometric_median(
            features,
            epsilon=float(center_cfg.epsilon),
            max_iters=int(center_cfg.max_iters),
            convergence_tol=float(center_cfg.convergence_tol),
        )
    elif str(center_cfg.mode) == "arithmetic_mean":
        center = features.mean(dim=2)
    else:
        raise ValueError(f"unsupported counterfactual center mode={center_cfg.mode!r}")
    center = center.detach()
    nuisance_cfg = cfg.nuisance
    nuisance = nuisance_statistics(
        features,
        center,
        epsilon=float(nuisance_cfg.epsilon),
        eigen_epsilon=float(nuisance_cfg.eigen_epsilon),
        rank_tolerance=float(nuisance_cfg.rank_tolerance),
        linalg_chunk_size=int(nuisance_cfg.linalg_chunk_size),
    )
    dispersion = (features - center.unsqueeze(2)).square().sum(dim=-1).median(dim=2).values.detach()
    stability = (1.0 - percentile_rank(dispersion)).detach()
    task_rank = percentile_rank(torch.as_tensor(task_sensitivity, device=prediction.device).detach()).detach()
    budget = NormalizedPatchBudget(float(cfg.budget.epsilon))(stability, task_rank)
    residual = prediction - center
    q = apply_soft_nuisance_metric(
        residual,
        nuisance.deviations.detach(),
        nuisance.scale.detach(),
        epsilon=float(nuisance_cfg.epsilon),
        linalg_chunk_size=int(nuisance_cfg.linalg_chunk_size),
    )
    rho_epsilon = float(robust_cfg.charbonnier_epsilon)
    robust = torch.sqrt(q + rho_epsilon**2) - rho_epsilon
    per_sample = (budget * robust).sum(dim=-1)
    raw_loss = _masked_mean(per_sample, valid)
    loss = raw_loss * float(robust_cfg.weight)
    influence = budget * (0.5 / torch.sqrt(q.detach() + rho_epsilon**2))
    eigenvalues = nuisance.eigenvalues.detach()
    energy_total = eigenvalues.sum(dim=-1).clamp_min(float(nuisance_cfg.epsilon))
    descending = torch.flip(eigenvalues, dims=(-1,))
    metrics = {
        "cvgc/loss": loss.detach(),
        "cvgc/raw_loss": raw_loss.detach(),
        "cvgc/valid_fraction": valid.float().mean().detach(),
        "cvgc/semantic_precision_enabled": torch.zeros((), device=prediction.device),
        "cvgc/tau_used_in_loss": torch.zeros((), device=prediction.device),
        "cvgc/target_stop_grad": torch.tensor(float(not center.requires_grad), device=prediction.device),
        "cvgc/q_p50": _quantile(q.detach(), 0.50),
        "cvgc/q_p95": _quantile(q.detach(), 0.95),
        "cvgc/q_p99": _quantile(q.detach(), 0.99),
        "cvgc/loss_p50": _quantile(robust.detach(), 0.50),
        "cvgc/loss_p95": _quantile(robust.detach(), 0.95),
        "cvgc/loss_p99": _quantile(robust.detach(), 0.99),
        "cvgc/per_patch_gradient_influence_p50": _quantile(influence, 0.50),
        "cvgc/per_patch_gradient_influence_p95": _quantile(influence, 0.95),
        "cvgc/per_patch_gradient_influence_p99": _quantile(influence, 0.99),
        "counterfactual/stability_mean": stability.mean(),
        "counterfactual/stability_p50": _quantile(stability, 0.50),
        "counterfactual/stability_p95": _quantile(stability, 0.95),
        "counterfactual/stability_p99": _quantile(stability, 0.99),
        "counterfactual/task_relevance_mean": task_rank.mean(),
        "counterfactual/task_relevance_p50": _quantile(task_rank, 0.50),
        "counterfactual/task_relevance_p95": _quantile(task_rank, 0.95),
        "counterfactual/task_relevance_p99": _quantile(task_rank, 0.99),
        "counterfactual/budget_mean": budget.mean(),
        "counterfactual/budget_p50": _quantile(budget, 0.50),
        "counterfactual/budget_p95": _quantile(budget, 0.95),
        "counterfactual/budget_p99": _quantile(budget, 0.99),
        "counterfactual/budget_sum_error_max": (budget.sum(dim=-1) - 1.0).abs().max(),
        "nuisance/effective_rank_mean": nuisance.effective_rank.float().mean().detach(),
        "nuisance/effective_rank_max": nuisance.effective_rank.max().detach(),
        "nuisance/operator_gain_min": nuisance.gains.min().detach(),
        "nuisance/operator_gain_max": torch.ones((), device=prediction.device),
        "nuisance/linalg_chunk_size": torch.tensor(
            float(nuisance.linalg_chunk_size), device=prediction.device
        ),
        "nuisance/eigensolver_split_retries": torch.tensor(
            float(nuisance.eigensolver_split_retries), device=prediction.device
        ),
        "nuisance/mean_attenuation": (1.0 - nuisance.gains).mean().detach(),
        "nuisance/p95_attenuation": _quantile(1.0 - nuisance.gains, 0.95),
        "nuisance/rank1_energy": (descending[..., :1].sum(dim=-1) / energy_total).mean(),
        "nuisance/rank3_energy": (descending[..., :3].sum(dim=-1) / energy_total).mean(),
        "nuisance/rank5_energy": (descending[..., :5].sum(dim=-1) / energy_total).mean(),
    }
    for index in range(min(5, int(descending.shape[-1]))):
        metrics[f"nuisance/singular_value_{index + 1}"] = torch.sqrt(
            descending[..., index] * float(cfg.views)
        ).mean()
    metrics.update(_oracle_budget_metrics(batch, budget, prediction.device))
    artifacts = {
        "center": center,
        "deviations": nuisance.deviations.detach(),
        "dispersion": dispersion,
        "stability": stability,
        "task_relevance": task_rank,
        "budget": budget,
        "q": q.detach(),
        "robust": robust.detach(),
        "effective_rank": nuisance.effective_rank.detach(),
        "eigenvalues": eigenvalues,
        "operator_gains": nuisance.gains.detach(),
    }
    return loss, metrics, artifacts


def task_normalized_medoid_msir(
    output,
    config,
    batch: dict[str, torch.Tensor],
    artifacts: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = config.model.msir
    prediction = prepare_reconstruction_image(output.rgb_reconstruction)
    if not bool(cfg.enabled) or not bool(cfg.training_enabled):
        zero = prediction.sum() * 0.0
        return zero, {
            "msir/loss": zero.detach(),
            "msir/enabled": torch.zeros((), device=prediction.device),
            "msir/decoder_training_enabled": torch.zeros((), device=prediction.device),
        }, {}
    if str(cfg.target_mode) != "counterfactual_medoid":
        raise ValueError(f"formal K=6 MSIR requires counterfactual_medoid, got {cfg.target_mode!r}")
    raw_views = _require(batch, "raw_b_views").to(prediction.device)
    features = _to_btknd(_require(batch, "patch_features").to(prediction.device)).detach()
    valid = _valid_mask(batch, prediction.shape[:2], prediction.device)
    features = torch.where(
        valid[..., None, None, None], features, torch.zeros_like(features)
    )
    target, selected, costs = select_counterfactual_medoid(
        raw_views,
        features,
        artifacts["center"],
        artifacts["budget"],
    )
    target = target.detach()
    pred_levels = laplacian_pyramid(prediction, levels=int(cfg.pyramid_levels))
    target_levels = [level.detach() for level in laplacian_pyramid(target, levels=int(cfg.pyramid_levels))]
    epsilon = float(cfg.eps)
    charbonnier_epsilon = float(cfg.get("charbonnier_epsilon", 1.0e-3))
    level_losses = []
    metrics: dict[str, torch.Tensor] = {
        "msir/enabled": torch.ones((), device=prediction.device),
        "msir/decoder_training_enabled": torch.ones((), device=prediction.device),
        "msir/medoid_real_view": torch.ones((), device=prediction.device),
        "msir/selected_view_mean": selected.float().mean(),
        "msir/medoid_cost_mean": costs.gather(2, selected.unsqueeze(-1)).mean(),
    }
    patch_budget = artifacts["budget"].detach()
    grid = int(config.external_vlm.grid_size)
    patch_map = patch_budget.reshape(-1, 1, grid, grid)
    for index, (pred_level, target_level) in enumerate(zip(pred_levels, target_levels, strict=True)):
        weight = F.interpolate(
            patch_map,
            size=pred_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*prediction.shape[:2], 1, *pred_level.shape[-2:]).detach()
        pixel_loss = (
            torch.sqrt((pred_level - target_level).square() + charbonnier_epsilon**2)
            - charbonnier_epsilon
        ).mean(dim=2, keepdim=True)
        mass = weight.sum(dim=(-1, -2, -3)).clamp_min(epsilon)
        per_sample = (pixel_loss * weight).sum(dim=(-1, -2, -3)) / mass
        level_loss = _masked_mean(per_sample, valid)
        level_losses.append(level_loss * float(cfg.level_weights[index]))
        metrics[f"msir/level{index}_loss"] = level_loss.detach()
        metrics[f"msir/level{index}_weight_mass"] = mass.mean().detach()
    total = sum(level_losses) * float(cfg.weight)
    metrics["msir/loss"] = total.detach()
    metrics["msir/valid_fraction"] = valid.float().mean().detach()
    return total, metrics, {
        "medoid_target": target,
        "selected_view_id": selected.detach(),
        "medoid_costs": costs.detach(),
    }


def _to_btknd(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 5:
        return tensor
    if tensor.ndim == 4:
        return tensor[:, None]
    raise ValueError(f"patch_features must be [B,T,K,N,D] or [B,K,N,D], got {tuple(tensor.shape)}")


def _valid_mask(batch: dict[str, Any], shape, device) -> torch.Tensor:
    valid = torch.ones(tuple(shape), dtype=torch.bool, device=device)
    for key, default in (("sequence_valid", True), ("six_view_valid", False), ("vlm_views_valid", False)):
        value = batch.get(key)
        if value is None:
            valid &= bool(default)
            continue
        tensor = torch.as_tensor(value, device=device).bool()
        if tensor.ndim == 1:
            tensor = tensor[:, None].expand(tuple(shape))
        valid &= tensor.reshape(tuple(shape))
    return valid


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(values.device).float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    flat = torch.as_tensor(value).detach().float().reshape(-1)
    return torch.quantile(flat, float(q)) if flat.numel() else torch.zeros((), device=value.device)


def _require(mapping: dict[str, Any], key: str) -> torch.Tensor:
    if key not in mapping or mapping[key] is None:
        raise KeyError(f"counterfactual E4/E5 requires batch['{key}']")
    return mapping[key]


def _oracle_budget_metrics(
    batch: dict[str, Any], budget: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    """Foreground masks are diagnostics only and never enter the loss path."""

    raw_mask = batch.get("task_foreground_mask")
    if raw_mask is None:
        return {"counterfactual/oracle_mask_available": torch.zeros((), device=device)}
    mask = torch.as_tensor(raw_mask, device=device).float()
    if mask.ndim == 5 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim == 5 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 4:
        return {"counterfactual/oracle_mask_available": torch.zeros((), device=device)}
    grid = int(round(budget.shape[-1] ** 0.5))
    pooled = F.adaptive_avg_pool2d(
        mask.reshape(-1, 1, *mask.shape[-2:]), (grid, grid)
    ).reshape(*budget.shape).detach()
    foreground = (pooled >= 0.5).float()
    foreground_mass = (budget.detach() * foreground).sum(dim=-1)
    background_mass = (budget.detach() * (1.0 - foreground)).sum(dim=-1)
    return {
        "counterfactual/oracle_mask_available": torch.ones((), device=device),
        "counterfactual/foreground_budget_mass": foreground_mass.mean(),
        "counterfactual/background_budget_mass": background_mass.mean(),
        "counterfactual/foreground_background_budget_ratio": (
            foreground_mass / background_mass.clamp_min(1.0e-8)
        ).mean(),
    }
