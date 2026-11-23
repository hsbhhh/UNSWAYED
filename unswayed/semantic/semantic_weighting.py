from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticWeightingResult:
    raw_scores: torch.Tensor
    z_scores: torch.Tensor
    per_view_weights: torch.Tensor
    fused_weights: torch.Tensor
    metrics: dict[str, torch.Tensor]


def semantic_weighting_from_batch(
    batch: dict,
    config: Any,
    *,
    device: torch.device | str,
) -> SemanticWeightingResult:
    """Build detached semantic weights with shape [B,T,P]."""

    # The production K-view contract always preserves the view dimension.
    # B1/B2 compatibility is isolated in the legacy loss path and must not be
    # silently selected by robust semantic weighting.
    scores = _to_btkp(_require(batch, "patch_scores"), device=device)
    cfg = config.model.semantic_weighting
    mode = str(cfg.mode)
    if mode == "robust_mad_sigmoid":
        result = robust_mad_semantic_weights(scores, cfg)
    elif mode == "standard_softmax":
        result = standard_softmax_semantic_weights(scores, cfg)
    else:
        raise ValueError(f"unsupported semantic_weighting.mode={mode!r}")
    oracle_metrics = _oracle_task_region_metrics(
        batch,
        result.fused_weights,
        result.raw_scores,
        result.per_view_weights,
        device=torch.device(device),
    )
    if not oracle_metrics:
        return result
    return SemanticWeightingResult(
        raw_scores=result.raw_scores,
        z_scores=result.z_scores,
        per_view_weights=result.per_view_weights,
        fused_weights=result.fused_weights,
        metrics={**result.metrics, **oracle_metrics},
    )


def robust_mad_semantic_weights(scores: torch.Tensor, cfg: Any) -> SemanticWeightingResult:
    """Median/MAD -> sigmoid -> view mean -> semantic-mass normalization.

    Input is [B,T,K,P]. All returned tensors are detached so task-conditioned
    VLM outputs cannot create a trainable score-adaptation path.
    """

    raw = torch.as_tensor(scores).detach().float()
    if raw.ndim != 4:
        raise ValueError(f"semantic scores must be [B,T,K,P], got {tuple(raw.shape)}")
    if raw.shape[2] != int(cfg.num_views):
        raise ValueError(
            f"semantic score view count must equal num_views={int(cfg.num_views)}, got {raw.shape[2]}"
        )
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("raw VLM patch scores contain NaN or Inf")
    if str(cfg.mode) != "robust_mad_sigmoid":
        raise ValueError(f"robust weighting requires mode=robust_mad_sigmoid, got {cfg.mode!r}")
    if str(cfg.aggregate) != "mean":
        raise ValueError("robust semantic weighting supports aggregate=mean only")
    if not bool(cfg.mass_normalize):
        raise ValueError("robust semantic weighting requires mass_normalize=true")

    eps = float(cfg.eps)
    floor = max(eps, float(cfg.get("denominator_floor", eps)))
    median = raw.median(dim=-1, keepdim=True).values
    mad = (raw - median).abs().median(dim=-1, keepdim=True).values
    denominator = (float(cfg.mad_scale) * mad + eps).clamp_min(floor)
    z = (raw - median) / denominator
    z_clip = float(cfg.get("z_clip", 12.0))
    z = z.clamp(min=-z_clip, max=z_clip)
    per_view = torch.sigmoid(float(cfg.beta) * z)
    fused = per_view.mean(dim=2)
    patch_count = int(fused.shape[-1])
    fused_sum = fused.sum(dim=-1, keepdim=True)
    weights = float(patch_count) * fused / (fused_sum + eps)
    if not bool(torch.isfinite(z).all() and torch.isfinite(per_view).all() and torch.isfinite(weights).all()):
        raise FloatingPointError("robust semantic weighting produced NaN or Inf")

    metrics = _weighting_metrics(raw, median, mad, z, per_view, weights)
    metrics["semantic_weight/num_views"] = torch.tensor(float(raw.shape[2]), device=raw.device)
    metrics["semantic_weight/patch_count"] = torch.tensor(float(patch_count), device=raw.device)
    metrics["semantic_weight/beta"] = torch.tensor(float(cfg.beta), device=raw.device)
    metrics["semantic_weight/mad_scale"] = torch.tensor(float(cfg.mad_scale), device=raw.device)
    metrics["semantic_weight/denominator_floor"] = torch.tensor(float(floor), device=raw.device)
    metrics["semantic_weight/weight_mean_error"] = (
        weights.mean(dim=-1) - float(cfg.expected_mean_weight)
    ).abs().max()
    return SemanticWeightingResult(
        raw_scores=raw.detach(),
        z_scores=z.detach(),
        per_view_weights=per_view.detach(),
        fused_weights=weights.detach(),
        metrics={key: value.detach() for key, value in metrics.items()},
    )


def standard_softmax_semantic_weights(scores: torch.Tensor, cfg: Any) -> SemanticWeightingResult:
    """Mean/std -> K-view z consensus -> temperature softmax semantic weights."""

    raw = torch.as_tensor(scores).detach().float()
    if raw.ndim != 4:
        raise ValueError(f"semantic scores must be [B,T,K,P], got {tuple(raw.shape)}")
    if raw.shape[2] != int(cfg.num_views):
        raise ValueError(
            f"semantic score view count must equal num_views={int(cfg.num_views)}, got {raw.shape[2]}"
        )
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("raw VLM patch scores contain NaN or Inf")
    if str(cfg.mode) != "standard_softmax":
        raise ValueError(f"standard softmax weighting requires mode=standard_softmax, got {cfg.mode!r}")
    if str(cfg.aggregate) != "mean":
        raise ValueError("standard softmax semantic weighting supports aggregate=mean only")
    if not bool(cfg.mass_normalize):
        raise ValueError("standard softmax semantic weighting requires mass_normalize=true")

    eps = float(cfg.eps)
    temperature = float(cfg.temperature)
    if eps <= 0.0:
        raise ValueError("semantic weighting eps must be positive")
    if temperature <= 0.0:
        raise ValueError("semantic weighting temperature must be positive")

    mean = raw.mean(dim=-1, keepdim=True)
    std = raw.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    z = (raw - mean) / (std + eps)
    z_bar = z.mean(dim=2)
    patch_count = int(z_bar.shape[-1])
    weights = float(patch_count) * torch.softmax(z_bar / temperature, dim=-1)
    per_view_weights = float(patch_count) * torch.softmax(z / temperature, dim=-1)
    if not bool(
        torch.isfinite(std).all()
        and torch.isfinite(z).all()
        and torch.isfinite(z_bar).all()
        and torch.isfinite(per_view_weights).all()
        and torch.isfinite(weights).all()
    ):
        raise FloatingPointError("standard softmax semantic weighting produced NaN or Inf")

    metrics = _standard_softmax_metrics(raw, mean, std, z, z_bar, per_view_weights, weights)
    metrics["semantic_weight/num_views"] = torch.tensor(float(raw.shape[2]), device=raw.device)
    metrics["semantic_weight/patch_count"] = torch.tensor(float(patch_count), device=raw.device)
    metrics["semantic_weight/temperature"] = torch.tensor(temperature, device=raw.device)
    metrics["semantic_weight/eps"] = torch.tensor(eps, device=raw.device)
    metrics["semantic_weight/mode_standard_softmax"] = torch.tensor(1.0, device=raw.device)
    metrics["semantic_weight/no_mad_path_used"] = torch.tensor(1.0, device=raw.device)
    metrics["semantic_weight/z_hard_clip_used"] = torch.tensor(0.0, device=raw.device)
    metrics["semantic_weight/weight_mean_error"] = (
        weights.mean(dim=-1) - float(cfg.expected_mean_weight)
    ).abs().max()
    return SemanticWeightingResult(
        raw_scores=raw.detach(),
        z_scores=z.detach(),
        per_view_weights=per_view_weights.detach(),
        fused_weights=weights.detach(),
        metrics={key: value.detach() for key, value in metrics.items()},
    )


def _weighting_metrics(
    raw: torch.Tensor,
    median: torch.Tensor,
    mad: torch.Tensor,
    z: torch.Tensor,
    per_view: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for view_id in range(int(raw.shape[2])):
        view = raw[:, :, view_id]
        view_z = z[:, :, view_id]
        metrics[f"semantic_weight/score_mean_view{view_id}"] = view.mean()
        metrics[f"semantic_weight/score_std_view{view_id}"] = view.std(unbiased=False)
        metrics[f"semantic_weight/score_min_view{view_id}"] = view.min()
        metrics[f"semantic_weight/score_max_view{view_id}"] = view.max()
        metrics[f"semantic_weight/median_score_view{view_id}"] = median[:, :, view_id].mean()
        metrics[f"semantic_weight/mad_view{view_id}"] = mad[:, :, view_id].mean()
        metrics[f"semantic_weight/z_mean_view{view_id}"] = view_z.mean()
        metrics[f"semantic_weight/z_std_view{view_id}"] = view_z.std(unbiased=False)
        metrics[f"semantic_weight/z_p95_view{view_id}"] = _quantile(view_z, 0.95)
        metrics[f"semantic_weight/z_p99_view{view_id}"] = _quantile(view_z, 0.99)
    raw_across_view_mean = raw.mean(dim=2)
    raw_across_view_std = raw.std(dim=2, unbiased=False)
    semantic_across_view_mean = per_view.mean(dim=2)
    semantic_across_view_std = per_view.std(dim=2, unbiased=False)
    metrics.update(
        {
            "semantic_weight/weight_mean": weights.mean(),
            "semantic_weight/weight_std": weights.std(unbiased=False),
            "semantic_weight/weight_min": weights.min(),
            "semantic_weight/weight_max": weights.max(),
            "semantic_weight/weight_p50": _quantile(weights, 0.50),
            "semantic_weight/weight_p90": _quantile(weights, 0.90),
            "semantic_weight/weight_p95": _quantile(weights, 0.95),
            "semantic_weight/weight_p99": _quantile(weights, 0.99),
            "semantic_weight/same_patch_raw_score_std_across_views": raw_across_view_std.mean(),
            "semantic_weight/same_patch_weight_std_across_views": semantic_across_view_std.mean(),
            "semantic_weight/raw_score_cross_view_std_mean": raw_across_view_std.mean(),
            "semantic_weight/raw_score_cross_view_std_p95": _quantile(raw_across_view_std, 0.95),
            "semantic_weight/normalized_z_cross_view_std_mean": z.std(dim=2, unbiased=False).mean(),
            "semantic_weight/semantic_a_cross_view_std_mean": semantic_across_view_std.mean(),
            "semantic_weight/semantic_a_cross_view_std_p95": _quantile(semantic_across_view_std, 0.95),
            "semantic_weight/same_patch_raw_score_cv_across_views": (
                raw_across_view_std / raw_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/same_patch_weight_cv_across_views": (
                semantic_across_view_std / semantic_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/weights_detached": torch.tensor(
                float(not weights.requires_grad), device=weights.device
            ),
        }
    )
    return metrics


def _standard_softmax_metrics(
    raw: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    z: torch.Tensor,
    z_bar: torch.Tensor,
    per_view_weights: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for view_id in range(int(raw.shape[2])):
        view = raw[:, :, view_id]
        view_z = z[:, :, view_id]
        metrics[f"semantic_weight/score_mean_view{view_id}"] = view.mean()
        metrics[f"semantic_weight/score_std_view{view_id}"] = view.std(unbiased=False)
        metrics[f"semantic_weight/score_min_view{view_id}"] = view.min()
        metrics[f"semantic_weight/score_max_view{view_id}"] = view.max()
        metrics[f"semantic_weight/standard_mean_view{view_id}"] = mean[:, :, view_id].mean()
        metrics[f"semantic_weight/standard_std_view{view_id}"] = std[:, :, view_id].mean()
        metrics[f"semantic_weight/z_mean_view{view_id}"] = view_z.mean()
        metrics[f"semantic_weight/z_std_view{view_id}"] = view_z.std(unbiased=False)
        metrics[f"semantic_weight/z_min_view{view_id}"] = view_z.min()
        metrics[f"semantic_weight/z_max_view{view_id}"] = view_z.max()
        metrics[f"semantic_weight/z_p50_view{view_id}"] = _quantile(view_z, 0.50)
        metrics[f"semantic_weight/z_p90_view{view_id}"] = _quantile(view_z, 0.90)
        metrics[f"semantic_weight/z_p95_view{view_id}"] = _quantile(view_z, 0.95)
        metrics[f"semantic_weight/z_p99_view{view_id}"] = _quantile(view_z, 0.99)
    raw_across_view_mean = raw.mean(dim=2)
    raw_across_view_std = raw.std(dim=2, unbiased=False)
    z_across_view_mean = z.mean(dim=2)
    z_across_view_std = z.std(dim=2, unbiased=False)
    weight_across_view_mean = per_view_weights.mean(dim=2)
    weight_across_view_std = per_view_weights.std(dim=2, unbiased=False)
    metrics.update(
        {
            "semantic_weight/z_bar_mean": z_bar.mean(),
            "semantic_weight/z_bar_std": z_bar.std(unbiased=False),
            "semantic_weight/z_bar_min": z_bar.min(),
            "semantic_weight/z_bar_max": z_bar.max(),
            "semantic_weight/z_bar_p95": _quantile(z_bar, 0.95),
            "semantic_weight/z_bar_p99": _quantile(z_bar, 0.99),
            "semantic_weight/weight_mean": weights.mean(),
            "semantic_weight/weight_std": weights.std(unbiased=False),
            "semantic_weight/weight_min": weights.min(),
            "semantic_weight/weight_max": weights.max(),
            "semantic_weight/weight_p50": _quantile(weights, 0.50),
            "semantic_weight/weight_p90": _quantile(weights, 0.90),
            "semantic_weight/weight_p95": _quantile(weights, 0.95),
            "semantic_weight/weight_p99": _quantile(weights, 0.99),
            "semantic_weight/same_patch_raw_score_std_across_views": raw_across_view_std.mean(),
            "semantic_weight/same_patch_weight_std_across_views": weight_across_view_std.mean(),
            "semantic_weight/raw_score_cross_view_std_mean": raw_across_view_std.mean(),
            "semantic_weight/raw_score_cross_view_std_p95": _quantile(raw_across_view_std, 0.95),
            "semantic_weight/standardized_z_cross_view_std_mean": z_across_view_std.mean(),
            "semantic_weight/normalized_z_cross_view_std_mean": z_across_view_std.mean(),
            "semantic_weight/semantic_a_cross_view_std_mean": weight_across_view_std.mean(),
            "semantic_weight/semantic_a_cross_view_std_p95": _quantile(weight_across_view_std, 0.95),
            "semantic_weight/raw_score_cross_view_cv": (
                raw_across_view_std / raw_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/standardized_z_cross_view_cv": (
                z_across_view_std / z_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/same_patch_raw_score_cv_across_views": (
                raw_across_view_std / raw_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/same_patch_weight_cv_across_views": (
                weight_across_view_std / weight_across_view_mean.abs().clamp_min(1.0e-6)
            ).mean(),
            "semantic_weight/weights_detached": torch.tensor(
                float(not weights.requires_grad), device=weights.device
            ),
        }
    )
    return metrics


def _to_btkp(value: Any, *, device: torch.device | str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device).float()
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor[:, None]
    raise ValueError(f"expected K-view scores [B,T,K,P] or [B,K,P], got {tuple(tensor.shape)}")


def _to_btp(value: Any, *, device: torch.device | str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device).float()
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2:
        return tensor[:, None]
    raise ValueError(f"expected scores [B,T,P] or [B,P], got {tuple(tensor.shape)}")


def _quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(value.detach().float().reshape(-1), float(q))


def _oracle_task_region_metrics(
    batch: dict,
    weights: torch.Tensor,
    raw_scores: torch.Tensor,
    per_view_weights: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Measure task-region separation without feeding the oracle mask into losses."""

    zero = torch.zeros((), device=device)
    raw_mask = batch.get("task_foreground_mask")
    if raw_mask is None:
        return {"semantic_weight/oracle_mask_available": zero}
    mask = torch.as_tensor(raw_mask, device=device).float().detach()
    if mask.ndim == 5 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim == 5 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3 and weights.shape[0] == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4 or tuple(mask.shape[:2]) != tuple(weights.shape[:2]):
        return {"semantic_weight/oracle_mask_available": zero}

    grid = int(round(int(weights.shape[-1]) ** 0.5))
    if grid * grid != int(weights.shape[-1]):
        return {"semantic_weight/oracle_mask_available": zero}
    pooled = F.adaptive_avg_pool2d(
        mask.reshape(-1, 1, *mask.shape[-2:]),
        (grid, grid),
    ).reshape_as(weights).clamp(0.0, 1.0)

    valid = torch.ones(weights.shape[:2], dtype=torch.bool, device=device)
    if batch.get("foreground_mask_valid") is not None:
        candidate = torch.as_tensor(batch["foreground_mask_valid"], device=device).bool()
        if candidate.ndim == 1 and weights.shape[0] == 1:
            candidate = candidate.unsqueeze(0)
        if tuple(candidate.shape) == tuple(valid.shape):
            valid = candidate
    valid_patch = valid.unsqueeze(-1).float()
    foreground_mass = pooled * valid_patch
    background_mass = (1.0 - pooled) * valid_patch
    foreground_denominator = foreground_mass.sum()
    background_denominator = background_mass.sum()
    available = (foreground_denominator > 1.0e-6) & (background_denominator > 1.0e-6)
    if not bool(available):
        return {
            "semantic_weight/oracle_mask_available": zero,
            "semantic_weight/oracle_valid_frame_fraction": valid.float().mean(),
        }

    detached_weights = weights.detach()
    foreground_mean = (detached_weights * foreground_mass).sum() / foreground_denominator
    background_mean = (detached_weights * background_mass).sum() / background_denominator
    weighted_foreground_sum = (detached_weights * foreground_mass).sum()
    weighted_background_sum = (detached_weights * background_mass).sum()
    weighted_total = (weighted_foreground_sum + weighted_background_sum).clamp_min(1.0e-8)
    task_semantic_mass = weighted_foreground_sum / weighted_total
    background_semantic_mass = weighted_background_sum / weighted_total
    task_patch_fraction = foreground_denominator / (
        foreground_denominator + background_denominator
    ).clamp_min(1.0e-8)
    task_mass_gain = task_semantic_mass / task_patch_fraction.clamp_min(1.0e-8)
    mean_raw_score = raw_scores.detach().mean(dim=2)
    foreground_raw_score = (mean_raw_score * foreground_mass).sum() / foreground_denominator
    background_raw_score = (mean_raw_score * background_mass).sum() / background_denominator
    raw_cv = raw_scores.detach().std(dim=2, unbiased=False) / mean_raw_score.abs().clamp_min(1.0e-6)
    semantic_mean = per_view_weights.detach().mean(dim=2)
    semantic_cv = per_view_weights.detach().std(dim=2, unbiased=False) / semantic_mean.abs().clamp_min(1.0e-6)
    task_raw_cv = (raw_cv * foreground_mass).sum() / foreground_denominator
    background_raw_cv = (raw_cv * background_mass).sum() / background_denominator
    task_semantic_cv = (semantic_cv * foreground_mass).sum() / foreground_denominator
    background_semantic_cv = (semantic_cv * background_mass).sum() / background_denominator
    return {
        "semantic_weight/oracle_mask_available": torch.ones((), device=device),
        "semantic_weight/oracle_valid_frame_fraction": valid.float().mean(),
        "semantic_weight/task_patch_weight_mean": foreground_mean,
        "semantic_weight/background_patch_weight_mean": background_mean,
        "semantic_weight/task_background_weight_advantage": foreground_mean - background_mean,
        "semantic_weight/task_background_weight_ratio": foreground_mean
        / background_mean.clamp_min(1.0e-8),
        "semantic_weight/task_semantic_mass": task_semantic_mass,
        "semantic_weight/background_semantic_mass": background_semantic_mass,
        "semantic_weight/task_patch_fraction": task_patch_fraction,
        "semantic_weight/task_mass_gain_over_uniform": task_mass_gain,
        "semantic_weight/task_patch_raw_score_mean": foreground_raw_score,
        "semantic_weight/background_patch_raw_score_mean": background_raw_score,
        "semantic_weight/task_background_raw_score_advantage": foreground_raw_score
        - background_raw_score,
        "semantic_weight/task_patch_raw_score_cv": task_raw_cv,
        # The final fused w has no view dimension; these requested CV metrics
        # therefore measure the per-view normalized semantic response a before
        # mean aggregation and mass normalization.
        "semantic_weight/task_patch_final_weight_cv": task_semantic_cv,
        "semantic_weight/final_task_weight_cv": task_semantic_cv,
        "semantic_weight/background_patch_raw_score_cv": background_raw_cv,
        "semantic_weight/background_patch_final_weight_cv": background_semantic_cv,
        "semantic_weight/final_background_weight_cv": background_semantic_cv,
        "semantic_weight/task_patch_semantic_a_cv": task_semantic_cv,
        "semantic_weight/background_patch_semantic_a_cv": background_semantic_cv,
    }


def _require(mapping: dict, key: str):
    if key not in mapping or mapping[key] is None:
        raise KeyError(f"semantic weighting requires batch['{key}']")
    return mapping[key]
