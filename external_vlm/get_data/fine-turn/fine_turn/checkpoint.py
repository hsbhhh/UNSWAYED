from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload.setdefault("rng_state", capture_rng_state())
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(str(path), map_location=map_location, weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


def extract_state_dict(payload: Mapping[str, Any], *, preferred_keys: tuple[str, ...] = ("task_relevance", "model", "model_state", "state_dict")) -> Mapping[str, Any]:
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def state_dict_match_report(model_state: Mapping[str, Any], checkpoint_state: Mapping[str, Any]) -> dict[str, Any]:
    matched = []
    missing = []
    unexpected = []
    shape_mismatch = []
    matched_numel = 0
    total_numel = 0
    checkpoint_keys = set(checkpoint_state.keys())
    for key, tensor in model_state.items():
        if torch.is_tensor(tensor):
            total_numel += int(tensor.numel())
        if key not in checkpoint_state:
            missing.append(key)
            continue
        value = checkpoint_state[key]
        if torch.is_tensor(tensor) and torch.is_tensor(value) and tuple(tensor.shape) != tuple(value.shape):
            shape_mismatch.append({"key": key, "model": list(tensor.shape), "checkpoint": list(value.shape)})
            continue
        matched.append(key)
        if torch.is_tensor(tensor):
            matched_numel += int(tensor.numel())
    for key in sorted(checkpoint_keys - set(model_state.keys())):
        unexpected.append(key)
    coverage = float(matched_numel / total_numel) if total_numel else 0.0
    return {
        "matched_total": len(matched),
        "missing_total": len(missing),
        "unexpected_total": len(unexpected),
        "shape_mismatch_total": len(shape_mismatch),
        "matched_numel": matched_numel,
        "total_numel": total_numel,
        "coverage": coverage,
        "matched_keys": matched[:50],
        "missing_keys": missing[:50],
        "unexpected_keys": unexpected[:50],
        "shape_mismatch": shape_mismatch[:50],
        "model_prefix_counts": _prefix_counts(model_state),
        "checkpoint_prefix_counts": _prefix_counts(checkpoint_state),
    }


def load_matching_state_dict(module, checkpoint_state: Mapping[str, Any], *, strict: bool = False, min_coverage: float | None = None) -> dict[str, Any]:
    model_state = module.state_dict()
    filtered = {}
    for key, value in checkpoint_state.items():
        if key in model_state and torch.is_tensor(value) and tuple(model_state[key].shape) == tuple(value.shape):
            filtered[key] = value
    result = module.load_state_dict(filtered, strict=False)
    report = state_dict_match_report(model_state, checkpoint_state)
    report["load_missing_total"] = len(getattr(result, "missing_keys", []) or [])
    report["load_unexpected_total"] = len(getattr(result, "unexpected_keys", []) or [])
    if strict and (report["missing_total"] or report["unexpected_total"] or report["shape_mismatch_total"]):
        raise RuntimeError(f"Checkpoint strict load failed: {report}")
    if min_coverage is not None and report["coverage"] < float(min_coverage):
        raise RuntimeError(f"Checkpoint coverage {report['coverage']:.4f} is below required {float(min_coverage):.4f}: {report}")
    return report


def _prefix_counts(state: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in state.keys():
        prefix = str(key).split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))
