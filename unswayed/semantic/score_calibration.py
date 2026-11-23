from __future__ import annotations

from typing import Any

import torch


def calibrate_semantic_scores(scores: torch.Tensor, config: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = _plain(config)
    mode = str(cfg.get("mode", "identity_probability"))
    if mode != "identity_probability":
        raise ValueError(f"UNSWAYED only supports semantic score calibration mode 'identity_probability', got {mode!r}.")
    raw = scores.float()
    expected_min = float(cfg.get("expected_min", 0.0))
    expected_max = float(cfg.get("expected_max", 1.0))
    fail = bool(cfg.get("fail_on_domain_violation", True))
    raw_min = raw.detach().amin()
    raw_max = raw.detach().amax()
    if fail and (float(raw_min) < expected_min - 1e-6 or float(raw_max) > expected_max + 1e-6):
        raise ValueError(
            "UNSWAYED semantic score contract violation: expected probabilities "
            f"in [{expected_min}, {expected_max}], got min={float(raw_min):.6g}, max={float(raw_max):.6g}."
        )
    calibrated = raw.clamp(expected_min, expected_max) if bool(cfg.get("clamp", True)) else raw
    return calibrated, score_calibration_stats(raw, calibrated)


def calibrate_semantic_score_pair(
    scores_b1: torch.Tensor,
    scores_b2: torch.Tensor,
    config: Any,
    **_: Any,
) -> dict[str, torch.Tensor]:
    raw_mean = 0.5 * (scores_b1.float() + scores_b2.float())
    calibrated, stats = calibrate_semantic_scores(raw_mean, config)
    return {
        "raw_score_mean": raw_mean.detach(),
        "calibrated_score_mean": calibrated.detach(),
        **stats,
    }


def score_calibration_stats(raw: torch.Tensor, calibrated: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = (calibrated.float() - raw.float()).abs()
    return {
        "score/raw_mean": raw.detach().float().mean(),
        "score/calibrated_mean": calibrated.detach().float().mean(),
        "score/raw_min": raw.detach().float().amin(),
        "score/raw_max": raw.detach().float().amax(),
        "score/calibrated_min": calibrated.detach().float().amin(),
        "score/calibrated_max": calibrated.detach().float().amax(),
        "score/mean_abs_calibration_change": diff.detach().mean(),
        "score/apply_extra_sigmoid": torch.tensor(0.0, device=raw.device),
        "score/per_frame_minmax": torch.tensor(0.0, device=raw.device),
    }


def semantic_variance_from_scores(
    scores: torch.Tensor,
    *,
    semantic_var_min: float,
    semantic_var_span: float,
    semantic_var_max: float,
    semantic_temperature: float,
) -> torch.Tensor:
    temperature = max(float(semantic_temperature), 1e-6)
    var = float(semantic_var_min) + float(semantic_var_span) * torch.exp(-scores.float() / temperature)
    return var.clamp(max=float(semantic_var_max))


def calibration_report_from_config(config: Any) -> dict[str, Any]:
    cfg = _plain(config)
    return {
        "mode": str(cfg.get("mode", "identity_probability")),
        "expected_domain": str(cfg.get("expected_domain", "probability")),
        "apply_extra_sigmoid": bool(cfg.get("apply_extra_sigmoid", False)),
        "per_frame_minmax": bool(cfg.get("per_frame_minmax", False)),
        "contract": "probability_[0,1]_identity",
    }


def _plain(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {str(k): v for k, v in vars(value).items()}
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return {}

