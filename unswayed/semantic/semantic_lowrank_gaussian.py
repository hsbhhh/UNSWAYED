from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .score_calibration import (
    calibrate_semantic_scores,
    calibrate_semantic_score_pair,
    score_calibration_stats,
    semantic_variance_from_scores,
)
from .semantic_weighting import SemanticWeightingResult


def normalize_semantic_score(scores: torch.Tensor, config: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return calibrate_semantic_scores(scores, config)


def lowrank_mahalanobis_distance(
    error: torch.Tensor,
    delta: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    err = error.float()
    diff = delta.float()
    var = sigma2.float().clamp_min(float(eps))
    while var.ndim < err.ndim:
        var = var.unsqueeze(-1)
    err2 = err.square().sum(dim=-1)
    delta2 = diff.square().sum(dim=-1)
    edot = (err * diff).sum(dim=-1)
    var_scalar = var.squeeze(-1)
    correction = 0.25 * edot.square() / (var_scalar * (var_scalar + 0.25 * delta2).clamp_min(float(eps)))
    maha = err2 / var_scalar - correction
    return maha.clamp_min(0.0)


def multiview_lowrank_mahalanobis_distance(
    error: torch.Tensor,
    factors: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Woodbury Mahalanobis for Sigma = sigma2 I + U U^T.

    ``error`` is [...,D] and ``factors`` is [...,D,K]. K-view callers use
    centered view values divided by sqrt(K-1), so U U^T is the unbiased
    empirical covariance requested by the production contract.
    """

    err = torch.as_tensor(error).float()
    u = torch.as_tensor(factors).float()
    if u.shape[:-2] != err.shape[:-1] or u.shape[-2] != err.shape[-1]:
        raise ValueError(
            f"multiview factors must be [...,D,K] for error [...,D], got error={tuple(err.shape)} "
            f"factors={tuple(u.shape)}"
        )
    var = torch.as_tensor(sigma2, device=err.device).float().clamp_min(float(eps))
    while var.ndim < err.ndim:
        var = var.unsqueeze(-1)
    var_scalar = var.squeeze(-1)
    gram = torch.einsum("...dk,...dl->...kl", u, u)
    eye = torch.eye(int(u.shape[-1]), device=err.device, dtype=err.dtype)
    system = eye + gram / var_scalar[..., None, None]
    rhs = torch.einsum("...d,...dk->...k", err, u) / var_scalar[..., None]
    solved = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    correction = (rhs * solved).sum(dim=-1)
    maha = err.square().sum(dim=-1) / var_scalar - correction
    return maha.clamp_min(0.0)


def semantic_feature_consensus_loss(
    output,
    config,
    batch: dict,
    *,
    scale: bool = True,
    semantic_weights: SemanticWeightingResult | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weighting_cfg = config.model.get("semantic_weighting", {"enabled": False})
    if not bool(weighting_cfg.get("enabled", False)):
        return _legacy_semantic_feature_consensus_loss(output, config, batch, scale=scale)
    if semantic_weights is None:
        raise RuntimeError("robust VGC requires the shared SemanticWeightingResult")
    cfg = config.model.vlm_gaussian_consensus
    if not cfg.enabled:
        return {}, {}
    pred = getattr(output, "vgc_patch_prediction", None)
    if pred is None:
        raise RuntimeError("VGC enabled but model did not produce vgc_patch_prediction")
    features = _multiview_features(batch, pred.device).detach().float()
    pred = pred.float()
    if pred.shape != features.shape[:2] + features.shape[3:]:
        raise ValueError(
            "VGC prediction and K-view VLM patch feature shapes differ: "
            f"pred={tuple(pred.shape)} targets={tuple(features.shape)}"
        )
    if bool(cfg.feature_normalize):
        pred = F.normalize(pred, dim=-1)
        features = F.normalize(features, dim=-1)
    view_count = int(features.shape[2])
    expected_view_count = int(config.model.num_views)
    if view_count != expected_view_count:
        raise ValueError(
            f"VGC runtime view count must equal model.num_views={expected_view_count}, got {view_count}"
        )
    if view_count < 2:
        raise ValueError("K-view empirical covariance requires at least two views")
    mu = features.mean(dim=2).detach()
    centered = (features - mu.unsqueeze(2)).detach()
    factors = centered.permute(0, 1, 3, 4, 2).contiguous() / math.sqrt(float(view_count - 1))
    weights = semantic_weights.fused_weights.to(pred.device)
    if weights.shape != pred.shape[:3]:
        raise ValueError(f"semantic weights must match [B,T,P]={tuple(pred.shape[:3])}, got {tuple(weights.shape)}")
    sigma2 = torch.full_like(weights, _robust_base_variance(cfg))
    residual = pred - mu
    maha = multiview_lowrank_mahalanobis_distance(residual, factors, sigma2, eps=float(cfg.eps))
    valid = _vgc_valid_mask(batch, maha.shape[:2], pred.device)
    feature_dim = int(pred.shape[-1])
    effective_weight = _effective_vgc_weight(cfg, feature_dim, device=pred.device)
    raw_loss = _masked_weighted_mean(maha, weights, valid)
    scaled_loss = raw_loss * effective_weight
    loss = scaled_loss if bool(scale) else raw_loss
    effective_precision = maha / residual.detach().square().sum(dim=-1).clamp_min(float(cfg.eps))
    cov_stats = _multiview_covariance_stats(factors, sigma2, effective_precision)
    metrics = {
        "vgc/loss": loss.detach(),
        "vgc/raw": raw_loss.detach(),
        "vgc/mahalanobis_mean": _masked_mean(maha, valid).detach(),
        "vgc/weighted_mahalanobis_mean": raw_loss.detach(),
        "vgc/mahalanobis_p50": _quantile(maha, 0.50),
        "vgc/mahalanobis_p95": _quantile(maha, 0.95),
        "vgc/mahalanobis_p99": _quantile(maha, 0.99),
        "vgc/per_dimension": (raw_loss / max(feature_dim, 1)).detach(),
        "vgc/effective_weight": torch.tensor(float(effective_weight), device=pred.device),
        "vgc/scaled": scaled_loss.detach(),
        "vgc/feature_dim": torch.tensor(float(feature_dim), device=pred.device),
        "vgc/view_count": torch.tensor(float(view_count), device=pred.device),
        "vgc/valid_count": _valid_count(maha, valid).detach(),
        "vgc/base_variance": sigma2.detach().mean(),
        "vgc/score_covariance_decoupled": torch.tensor(1.0, device=pred.device),
        "vgc/semantic_score_used_by_covariance": torch.tensor(0.0, device=pred.device),
        "vgc/semantic_weight_used_by_covariance": torch.tensor(0.0, device=pred.device),
        "vgc/semantic_weight_precision_corr": _pearson_correlation(weights, effective_precision),
        "vgc/covariance_denominator": torch.tensor(float(view_count - 1), device=pred.device),
        "vgc/tau_used_in_loss": torch.tensor(0.0, device=pred.device),
        "vgc/view_deviation_norm_mean": centered.detach().norm(dim=-1).mean(),
        "vgc/target_stop_grad": torch.tensor(
            float(not mu.requires_grad and not factors.requires_grad and not weights.requires_grad),
            device=pred.device,
        ),
        "vgc/prediction_horizon": torch.tensor(0.0, device=pred.device),
    }
    metrics.update({f"vgc/{key}": value.detach() for key, value in cov_stats.items()})
    metrics.update(semantic_weights.metrics)
    return {"vgc/feature_k0": loss}, metrics


def _legacy_semantic_feature_consensus_loss(
    output,
    config,
    batch: dict,
    *,
    scale: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = config.model.vlm_gaussian_consensus
    if not cfg.enabled:
        return {}, {}
    pred = getattr(output, "vgc_patch_prediction", None)
    if pred is None:
        raise RuntimeError("VGC enabled but model did not produce vgc_patch_prediction")
    patch_scores_b1 = _require(batch, "patch_scores_b1").to(pred.device)
    patch_scores_b2 = _require(batch, "patch_scores_b2").to(pred.device)
    patch_features_b1 = _require(batch, "patch_features_b1").to(pred.device)
    patch_features_b2 = _require(batch, "patch_features_b2").to(pred.device)
    patch_scores_b1 = _to_btn(patch_scores_b1).detach()
    patch_scores_b2 = _to_btn(patch_scores_b2).detach()
    patch_features_b1 = _to_btnd(patch_features_b1).detach().float()
    patch_features_b2 = _to_btnd(patch_features_b2).detach().float()
    pred = pred.float()
    if pred.shape != patch_features_b1.shape:
        raise ValueError(
            "VGC prediction and VLM patch feature shapes differ: "
            f"pred={tuple(pred.shape)} target={tuple(patch_features_b1.shape)}"
        )
    if bool(cfg.feature_normalize):
        pred = F.normalize(pred, dim=-1)
        patch_features_b1 = F.normalize(patch_features_b1, dim=-1)
        patch_features_b2 = F.normalize(patch_features_b2, dim=-1)
    score_terms = calibrate_semantic_score_pair(
        patch_scores_b1,
        patch_scores_b2,
        config.model.semantic_score_calibration,
    )
    score_avg_raw = score_terms["raw_score_mean"]
    score_avg = score_terms["calibrated_score_mean"]
    mu = (0.5 * (patch_features_b1 + patch_features_b2)).detach()
    delta = (patch_features_b1 - patch_features_b2).detach()
    sigma2 = semantic_variance_from_scores(
        score_avg,
        semantic_var_min=float(cfg.sigma_min) ** 2,
        semantic_var_span=float(cfg.sigma_vlm) ** 2,
        semantic_var_max=float(cfg.sigma_max) ** 2,
        semantic_temperature=float(cfg.tau_s),
    )
    maha = lowrank_mahalanobis_distance(pred - mu, delta, sigma2, eps=float(cfg.eps))
    valid = _vgc_valid_mask(batch, maha.shape[:2], pred.device)
    feature_dim = int(pred.shape[-1])
    effective_weight = _effective_vgc_weight(cfg, feature_dim, device=pred.device)
    raw_loss = _masked_mean(maha, valid)
    scaled_loss = raw_loss * effective_weight
    loss = scaled_loss if bool(scale) else raw_loss
    metrics = {
        "vgc/loss": loss.detach(),
        "vgc/raw": raw_loss.detach(),
        "vgc/mahalanobis_mean": raw_loss.detach(),
        "vgc/per_dimension": (raw_loss / max(feature_dim, 1)).detach(),
        "vgc/effective_weight": torch.tensor(float(effective_weight), device=pred.device),
        "vgc/scaled": scaled_loss.detach(),
        "vgc/feature_dim": torch.tensor(float(feature_dim), device=pred.device),
        "vgc/weight_mode_inverse_feature_dim": torch.tensor(
            float(str(getattr(cfg, "weight_mode", "configured")) == "inverse_feature_dim"),
            device=pred.device,
        ),
        "vgc/valid_count": _valid_count(maha, valid).detach(),
        "vgc/sigma2_mean": sigma2.detach().mean(),
        "vgc/semantic_var_min": sigma2.detach().min(),
        "vgc/semantic_var_mean": sigma2.detach().mean(),
        "vgc/semantic_var_max": sigma2.detach().max(),
        "vgc/precision_min": (1.0 / sigma2.detach().clamp_min(float(cfg.eps))).min(),
        "vgc/precision_mean": (1.0 / sigma2.detach().clamp_min(float(cfg.eps))).mean(),
        "vgc/precision_max": (1.0 / sigma2.detach().clamp_min(float(cfg.eps))).max(),
        "vgc/delta_norm_mean": delta.detach().norm(dim=-1).mean(),
        "vgc/target_stop_grad": torch.tensor(float(not mu.requires_grad and not delta.requires_grad and not sigma2.requires_grad), device=pred.device),
        "vgc/prediction_horizon": torch.tensor(0.0, device=pred.device),
    }
    metrics.update({f"vgc/{key}": value.detach() for key, value in score_calibration_stats(score_avg_raw, score_avg).items()})
    return {"vgc/feature_k0": loss}, metrics


def _effective_vgc_weight(cfg: Any, feature_dim: int, *, device) -> float:
    mode = str(getattr(cfg, "weight_mode", "configured"))
    configured = float(getattr(cfg, "weight", 1.0))
    if mode == "configured":
        return configured
    if mode == "inverse_feature_dim":
        expected = 1.0 / max(int(feature_dim), 1)
        if abs(configured - expected) > 1.0e-8:
            raise RuntimeError(
                "VGC weight_mode=inverse_feature_dim requires configured weight to equal "
                f"1/feature_dim to prevent ambiguous or double scaling: weight={configured} "
                f"feature_dim={feature_dim} expected={expected}"
            )
        return expected
    raise ValueError(f"unsupported VGC weight_mode={mode!r}")


def msir_loss(
    output,
    config,
    batch: dict,
    *,
    semantic_weights: SemanticWeightingResult | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weighting_cfg = config.model.get("semantic_weighting", {"enabled": False})
    if not bool(weighting_cfg.get("enabled", False)):
        return _legacy_msir_loss(output, config, batch)
    if semantic_weights is None:
        raise RuntimeError("robust MSIR requires the shared SemanticWeightingResult")
    cfg = config.model.msir
    if not cfg.enabled:
        return {}, {}
    pred = getattr(output, "rgb_reconstruction", None)
    if pred is None:
        raise RuntimeError("MSIR enabled but model did not produce rgb_reconstruction")
    pred = prepare_reconstruction_image(pred)
    raw_views = _multiview_rgb(batch, pred.device)
    view_count = int(raw_views.shape[2])
    expected_view_count = int(config.model.num_views)
    if view_count != expected_view_count:
        raise ValueError(
            f"MSIR runtime view count must equal model.num_views={expected_view_count}, got {view_count}"
        )
    if view_count < 2:
        raise ValueError("K-view MSIR empirical covariance requires at least two views")
    raw_views = prepare_k_view_reconstruction_images(raw_views).detach()
    weights = semantic_weights.fused_weights.to(pred.device)
    if weights.shape[:2] != pred.shape[:2]:
        raise ValueError(f"MSIR semantic weights must match [B,T], got {tuple(weights.shape)}")
    rho = _patch_scores_to_map(weights, pred.shape[-2:], config.external_vlm.grid_size)
    pred_levels = laplacian_pyramid(pred, levels=int(cfg.pyramid_levels))
    view_levels = [level.detach() for level in k_view_laplacian_pyramid(raw_views, levels=int(cfg.pyramid_levels))]
    valid = _msir_valid_mask(batch, pred.shape[:2], pred.device)
    valid_count = _valid_count(torch.ones(pred.shape[:2], device=pred.device), valid)
    if float(valid_count.detach().cpu()) <= 0.0:
        zero = pred.sum() * 0.0
        return {"msir/pyramid_mahalanobis": zero}, {
            "msir/loss": zero.detach(),
            "msir/valid_count": zero.detach(),
            "msir/skipped_no_valid_pair": torch.tensor(1.0, device=pred.device),
            "msir/target_stop_grad": torch.tensor(1.0, device=pred.device),
        }
    losses = []
    metrics: dict[str, torch.Tensor] = {}
    all_effective_precision = []
    all_cov_max_upper_bound = []
    all_mahalanobis = []
    all_weight_precision_corr = []
    for index, (pred_level, view_level) in enumerate(zip(pred_levels, view_levels, strict=True)):
        weight_level = F.interpolate(
            rho.reshape(-1, 1, *rho.shape[-2:]),
            size=pred_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*pred.shape[:2], *pred_level.shape[-2:]).detach()
        sigma2 = torch.full(
            (*pred.shape[:2], *pred_level.shape[-2:]),
            _robust_base_variance(cfg),
            dtype=pred_level.dtype,
            device=pred_level.device,
        )
        mu = view_level.mean(dim=2).detach()
        centered = (view_level - mu.unsqueeze(2)).detach()
        factors = centered.permute(0, 1, 4, 5, 3, 2).contiguous() / math.sqrt(float(view_count - 1))
        error = (pred_level - mu).permute(0, 1, 3, 4, 2).contiguous()
        maha_map = multiview_lowrank_mahalanobis_distance(error, factors, sigma2, eps=float(cfg.eps))
        level_loss = _masked_weighted_spatial_mean(maha_map, weight_level, valid)
        level_weight = float(cfg.level_weights[index])
        losses.append(level_loss * level_weight)
        residual2 = error.detach().square().sum(dim=-1).clamp_min(float(cfg.eps))
        effective_precision = maha_map.detach() / residual2
        # Trace is a cheap, safe upper bound for the largest covariance
        # eigenvalue and avoids a multi-million-matrix eigensolve per update.
        cov_max_upper_bound = sigma2.detach() + factors.detach().square().sum(dim=(-2, -1))
        all_effective_precision.append(effective_precision.reshape(-1))
        all_cov_max_upper_bound.append(cov_max_upper_bound.reshape(-1))
        all_mahalanobis.append(maha_map.detach().reshape(-1))
        level_corr = _pearson_correlation(weight_level, effective_precision)
        all_weight_precision_corr.append(level_corr)
        metrics[f"msir/level{index}_loss"] = level_loss.detach()
        metrics[f"msir/level{index}_base_variance"] = sigma2.detach().mean()
        metrics[f"msir/level{index}_view_deviation_norm_mean"] = centered.detach().square().sum(dim=3).sqrt().mean()
        metrics[f"msir/level{index}_weight_mean"] = weight_level.detach().mean()
        metrics[f"msir/level{index}_weight_precision_corr"] = level_corr
        metrics[f"msir/level{index}_height"] = torch.tensor(float(pred_level.shape[-2]), device=pred.device)
        metrics[f"msir/level{index}_width"] = torch.tensor(float(pred_level.shape[-1]), device=pred.device)
    total = sum(losses) * float(cfg.weight)
    precision = torch.cat(all_effective_precision)
    cov_max_upper_bound = torch.cat(all_cov_max_upper_bound)
    mahalanobis = torch.cat(all_mahalanobis)
    metrics.update(
        {
            "msir/loss": total.detach(),
            "msir/valid_count": valid_count.detach(),
            "msir/skipped_no_valid_pair": torch.tensor(0.0, device=pred.device),
            "msir/target_stop_grad": torch.tensor(1.0, device=pred.device),
            "msir/shared_semantic_weight": torch.tensor(1.0, device=pred.device),
            "msir/score_covariance_decoupled": torch.tensor(1.0, device=pred.device),
            "msir/semantic_score_used_by_covariance": torch.tensor(0.0, device=pred.device),
            "msir/semantic_weight_used_by_covariance": torch.tensor(0.0, device=pred.device),
            "msir/semantic_weight_precision_corr": torch.stack(all_weight_precision_corr).mean(),
            "msir/tau_used_in_loss": torch.tensor(0.0, device=pred.device),
            "msir/cov_eigen_min": torch.tensor(_robust_base_variance(cfg), device=pred.device),
            "msir/cov_eigen_max_upper_bound": cov_max_upper_bound.max(),
            "msir/precision_p50": _quantile(precision, 0.50),
            "msir/precision_p95": _quantile(precision, 0.95),
            "msir/precision_p99": _quantile(precision, 0.99),
            "msir/mahalanobis_p50": _quantile(mahalanobis, 0.50),
            "msir/mahalanobis_p95": _quantile(mahalanobis, 0.95),
            "msir/mahalanobis_p99": _quantile(mahalanobis, 0.99),
            "msir/weight_mean": weights.detach().mean(),
            "msir/view_count": torch.tensor(float(view_count), device=pred.device),
            "msir/covariance_denominator": torch.tensor(float(view_count - 1), device=pred.device),
            "msir/mean_channels": torch.tensor(float(pred.shape[2]), device=pred.device),
        }
    )
    return {"msir/pyramid_mahalanobis": total}, metrics


def _legacy_msir_loss(output, config, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = config.model.msir
    if not cfg.enabled:
        return {}, {}
    pred = getattr(output, "rgb_reconstruction", None)
    if pred is None:
        raise RuntimeError("MSIR enabled but model did not produce rgb_reconstruction")
    pred = prepare_reconstruction_image(pred)
    if "raw_b1" not in batch or "raw_b2" not in batch:
        raise KeyError(
            "MSIR is enabled but replay samples do not contain raw_b1/raw_b2. "
            "Recollect replay with the paired B1/B2 image schema or disable MSIR."
        )
    raw_b1 = prepare_reconstruction_image(_require(batch, "raw_b1").to(pred.device)).detach()
    raw_b2 = prepare_reconstruction_image(_require(batch, "raw_b2").to(pred.device)).detach()
    patch_scores_b1 = _to_btn(_require(batch, "patch_scores_b1").to(pred.device)).detach()
    patch_scores_b2 = _to_btn(_require(batch, "patch_scores_b2").to(pred.device)).detach()
    score_terms = calibrate_semantic_score_pair(
        patch_scores_b1,
        patch_scores_b2,
        config.model.semantic_score_calibration,
    )
    score_avg = score_terms["calibrated_score_mean"]
    rho = _patch_scores_to_map(score_avg, pred.shape[-2:], config.external_vlm.grid_size)
    pred_levels = laplacian_pyramid(pred, levels=int(cfg.pyramid_levels))
    b1_levels = [level.detach() for level in laplacian_pyramid(raw_b1, levels=int(cfg.pyramid_levels))]
    b2_levels = [level.detach() for level in laplacian_pyramid(raw_b2, levels=int(cfg.pyramid_levels))]
    valid = _msir_valid_mask(batch, pred.shape[:2], pred.device)
    valid_count = _valid_count(torch.ones(pred.shape[:2], device=pred.device), valid)
    if float(valid_count.detach().cpu()) <= 0.0:
        zero = pred.sum() * 0.0
        metrics = {
            "msir/loss": zero.detach(),
            "msir/valid_count": zero.detach(),
            "msir/skipped_no_valid_pair": torch.tensor(1.0, device=pred.device),
            "msir/target_stop_grad": torch.tensor(1.0, device=pred.device),
        }
        return {"msir/pyramid_mahalanobis": zero}, metrics
    losses = []
    metrics: dict[str, torch.Tensor] = {}
    for index, (pred_level, b1_level, b2_level) in enumerate(zip(pred_levels, b1_levels, b2_levels, strict=True)):
        rho_level = F.interpolate(
            rho.reshape(-1, 1, *rho.shape[-2:]),
            size=pred_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*pred.shape[:2], 1, *pred_level.shape[-2:]).detach()
        sigma2 = semantic_variance_from_scores(
            rho_level,
            semantic_var_min=float(cfg.sigma_min) ** 2,
            semantic_var_span=float(cfg.sigma_vlm) ** 2,
            semantic_var_max=float(cfg.sigma_max) ** 2,
            semantic_temperature=float(cfg.tau_s),
        )
        mu = (0.5 * (b1_level + b2_level)).detach()
        delta = (b1_level - b2_level).detach()
        maha = _pixel_mahalanobis(pred_level - mu, delta, sigma2, eps=float(cfg.eps))
        level_loss = _masked_mean(maha, valid)
        weight = float(cfg.level_weights[index])
        losses.append(level_loss * weight)
        metrics[f"msir/level{index}_loss"] = level_loss.detach()
        metrics[f"msir/level{index}_sigma2_mean"] = sigma2.detach().mean()
        metrics[f"msir/level{index}_delta_norm_mean"] = delta.detach().square().sum(dim=2).sqrt().mean()
    total = sum(losses) * float(cfg.weight)
    metrics.update(
        {
            f"msir/{key}": value.detach()
            for key, value in score_calibration_stats(score_terms["raw_score_mean"], score_terms["calibrated_score_mean"]).items()
        }
    )
    metrics["msir/loss"] = total.detach()
    metrics["msir/valid_count"] = valid_count.detach()
    metrics["msir/skipped_no_valid_pair"] = torch.tensor(0.0, device=pred.device)
    metrics["msir/target_stop_grad"] = torch.tensor(1.0, device=pred.device)
    return {"msir/pyramid_mahalanobis": total}, metrics


def prepare_reconstruction_image(value: torch.Tensor) -> torch.Tensor:
    """Convert replay/decoder RGB tensors to the decoder pixel range [0, 1]."""

    tensor = _btchw(value).float()
    if tensor.numel() and float(tensor.detach().amax().cpu()) > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def prepare_k_view_reconstruction_images(value: torch.Tensor) -> torch.Tensor:
    """Convert explicit [B,T,K,C,H,W] K-view RGB to decoder range [0,1]."""

    tensor = torch.as_tensor(value).float()
    if tensor.ndim != 6 or tensor.shape[3] != 3:
        raise ValueError(f"expected K-view RGB [B,T,K,3,H,W], got {tuple(tensor.shape)}")
    if tensor.numel() and float(tensor.detach().amax().cpu()) > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def laplacian_pyramid(images: torch.Tensor, *, levels: int = 3) -> list[torch.Tensor]:
    if int(levels) != 3:
        raise ValueError("semantic MSIR requires exactly 3 pyramid levels")
    x = _btchw(images).float()
    prefix = x.shape[:2]
    current = x.reshape(-1, *x.shape[2:])
    out: list[torch.Tensor] = []
    for _ in range(int(levels) - 1):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
        out.append((current - up).reshape(*prefix, *current.shape[1:]))
        current = down
    out.append(current.reshape(*prefix, *current.shape[1:]))
    return out


def k_view_laplacian_pyramid(images: torch.Tensor, *, levels: int = 3) -> list[torch.Tensor]:
    """Apply the unchanged Laplacian decomposition independently to every view."""

    views = prepare_k_view_reconstruction_images(images)
    batch, time, count, channels, height, width = views.shape
    flattened = views.reshape(batch, time * count, channels, height, width)
    flattened_levels = laplacian_pyramid(flattened, levels=levels)
    return [
        level.reshape(batch, time, count, channels, *level.shape[-2:])
        for level in flattened_levels
    ]


def reconstruct_laplacian_pyramid(levels: list[torch.Tensor]) -> torch.Tensor:
    if len(levels) != 3:
        raise ValueError("semantic MSIR requires exactly 3 pyramid levels")
    current = levels[-1]
    for residual in reversed(levels[:-1]):
        up = F.interpolate(
            current.reshape(-1, *current.shape[2:]),
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(*residual.shape)
        current = up + residual
    return current


def _pixel_mahalanobis(error: torch.Tensor, delta: torch.Tensor, sigma2: torch.Tensor, *, eps: float) -> torch.Tensor:
    return _pixel_mahalanobis_map(error, delta, sigma2, eps=eps).mean(dim=(-1, -2))


def _pixel_mahalanobis_map(
    error: torch.Tensor,
    delta: torch.Tensor,
    sigma2: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    err = error.permute(0, 1, 3, 4, 2).contiguous()
    diff = delta.permute(0, 1, 3, 4, 2).contiguous()
    var = sigma2.permute(0, 1, 3, 4, 2).contiguous()
    return lowrank_mahalanobis_distance(err, diff, var, eps=eps)


def _patch_scores_to_map(scores: torch.Tensor, size: tuple[int, int], grid_size: int) -> torch.Tensor:
    batch, time, count = scores.shape
    grid = int(grid_size)
    if grid * grid != int(count):
        root = int(math.sqrt(int(count)))
        if root * root != int(count):
            raise ValueError(f"patch score count must be square, got {count}")
        grid = root
    patch_map = scores.reshape(batch * time, 1, grid, grid)
    up = F.interpolate(patch_map, size=size, mode="bilinear", align_corners=False)
    return up.reshape(batch, time, 1, *size)


def _to_btn(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2:
        return tensor[:, None]
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    raise ValueError(f"expected patch scores [B,T,N] or [B,N], got {tuple(tensor.shape)}")


def _to_btnd(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor[:, None]
    raise ValueError(f"expected patch features [B,T,N,D] or [B,N,D], got {tuple(tensor.shape)}")


def _multiview_features(batch: dict, device) -> torch.Tensor:
    tensor = torch.as_tensor(_require(batch, "patch_features"), device=device).float()
    if tensor.ndim == 5:
        return tensor
    if tensor.ndim == 4:
        return tensor[:, None]
    raise ValueError(f"expected K-view features [B,T,K,P,D] or [B,K,P,D], got {tuple(tensor.shape)}")


def _multiview_rgb(batch: dict, device) -> torch.Tensor:
    tensor = torch.as_tensor(_require(batch, "raw_b_views"), device=device)
    if tensor.ndim == 6 and tensor.shape[3] == 3:
        return tensor.float()
    if tensor.ndim == 6 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 2, 5, 3, 4).contiguous().float()
    if tensor.ndim == 5 and tensor.shape[2] == 3:
        return tensor[:, None].float()
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3).contiguous()[:, None].float()
    raise ValueError(f"expected K-view RGB [B,T,K,H,W,3] or [B,T,K,3,H,W], got {tuple(tensor.shape)}")


def _btchw(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 5 and tensor.shape[2] == 3:
        return tensor.float()
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3).contiguous().float()
    if tensor.ndim == 4 and tensor.shape[1] == 3:
        return tensor[:, None].float()
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        return tensor.permute(0, 3, 1, 2).contiguous()[:, None].float()
    raise ValueError(f"expected RGB tensor, got {tuple(tensor.shape)}")


def _vgc_valid_mask(batch: dict, shape: torch.Size | tuple[int, int], device) -> torch.Tensor:
    paired = _bool_mask_from_batch(batch, "paired_view_valid", shape, device, default=True)
    vlm = _bool_mask_from_batch(batch, "vlm_pair_valid", shape, device, default=False)
    sequence = _bool_mask_from_batch(batch, "sequence_valid", shape, device, default=True)
    return paired & vlm & sequence


def _msir_valid_mask(batch: dict, shape: torch.Size | tuple[int, int], device) -> torch.Tensor:
    paired = _bool_mask_from_batch(batch, "paired_view_valid", shape, device, default=True)
    vlm = _bool_mask_from_batch(batch, "vlm_pair_valid", shape, device, default=False)
    sequence = _bool_mask_from_batch(batch, "sequence_valid", shape, device, default=True)
    return paired & vlm & sequence


def _bool_mask_from_batch(batch: dict, key: str, shape: torch.Size | tuple[int, int], device, *, default: bool) -> torch.Tensor:
    valid = batch.get(key)
    if valid is None:
        return torch.full(tuple(shape), bool(default), dtype=torch.bool, device=device)
    out = torch.as_tensor(valid, device=device).bool()
    if out.ndim == 1:
        out = out[:, None].expand(tuple(shape))
    if out.shape != tuple(shape):
        out = out.reshape(tuple(shape))
    return out


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    mask = valid.to(values.device).float().expand_as(values)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"weighted patch loss shape mismatch: values={tuple(values.shape)} weights={tuple(weights.shape)}")
    frame_mask = valid.to(values.device).float().unsqueeze(-1)
    numerator = (values * weights * frame_mask).sum()
    denominator = (frame_mask.sum() * float(values.shape[-1])).clamp_min(1.0)
    return numerator / denominator


def _masked_weighted_spatial_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"weighted spatial loss shape mismatch: values={tuple(values.shape)} weights={tuple(weights.shape)}")
    frame_mask = valid.to(values.device).float().unsqueeze(-1).unsqueeze(-1)
    numerator = (values * weights * frame_mask).sum()
    denominator = (frame_mask.sum() * float(values.shape[-2] * values.shape[-1])).clamp_min(1.0)
    return numerator / denominator


def _robust_base_variance(cfg: Any) -> float:
    # In robust mode the legacy sigma_vlm span becomes a fixed configured
    # isotropic component. Neither raw nor calibrated VLM score is consulted.
    return float(cfg.sigma_min) ** 2 + float(cfg.base_variance)


def _multiview_covariance_stats(
    factors: torch.Tensor,
    sigma2: torch.Tensor,
    effective_precision: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gram = torch.einsum("...dk,...dl->...kl", factors.detach(), factors.detach())
    # cuSOLVER's batched symmetric eigensolver can fail internally when the
    # formal [B=16,T=64,P=64] batch is flattened into 65,536 matrices at once.
    # This is diagnostics-only and chunking preserves the exact eigenvalues.
    flat_gram = gram.float().reshape(-1, gram.shape[-2], gram.shape[-1])
    flat_sigma2 = sigma2.detach().float().reshape(-1)
    chunk_size = 4096
    cov_max = torch.full((), -torch.inf, device=gram.device, dtype=torch.float32)
    for start in range(0, int(flat_gram.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(flat_gram.shape[0]))
        eigenvalues = torch.linalg.eigvalsh(flat_gram[start:stop])
        chunk_cov_max = (flat_sigma2[start:stop] + eigenvalues[:, -1]).max()
        cov_max = torch.maximum(cov_max, chunk_cov_max)
    precision = effective_precision.detach().reshape(-1)
    return {
        "cov_eigen_min": sigma2.detach().min(),
        "cov_eigen_max": cov_max,
        "precision_p50": _quantile(precision, 0.50),
        "precision_p95": _quantile(precision, 0.95),
        "precision_p99": _quantile(precision, 0.99),
    }


def _quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(torch.as_tensor(value).detach().float().reshape(-1), float(q))


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(left).detach().float().reshape(-1)
    y = torch.as_tensor(right).detach().float().reshape(-1)
    if x.numel() != y.numel():
        raise ValueError(f"correlation inputs must have equal size, got {x.numel()} and {y.numel()}")
    x = x - x.mean()
    y = y - y.mean()
    denominator = (x.square().sum() * y.square().sum()).sqrt()
    if not bool(torch.isfinite(denominator)) or float(denominator.cpu()) <= 1.0e-12:
        return torch.zeros((), device=x.device)
    return (x * y).sum() / denominator


def _valid_count(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    return valid.to(values.device).float().expand_as(values).sum()


def _require(mapping: dict, key: str):
    if key not in mapping or mapping[key] is None:
        raise KeyError(f"semantic low-rank Gaussian/MSIR requires batch['{key}']")
    return mapping[key]
