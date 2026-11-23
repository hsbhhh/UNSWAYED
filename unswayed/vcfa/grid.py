from __future__ import annotations

import torch
import torch.nn.functional as F


def resolve_vcfa_grid(config: dict, default_grid: tuple[int, int]) -> tuple[int, int]:
    vcfa = merged_vcfa_config(config)
    if vcfa.get("vlm_grid_size") is not None:
        size = int(vcfa["vlm_grid_size"])
        return size, size
    h = vcfa.get("vlm_grid_h", vcfa.get("grid_h"))
    w = vcfa.get("vlm_grid_w", vcfa.get("grid_w"))
    if h is not None or w is not None:
        return int(h or w), int(w or h)
    return tuple(int(x) for x in default_grid)


def merged_vcfa_config(config: dict) -> dict:
    merged = dict(config.get("vcfa", {}) or {})
    model_vcfa = (config.get("model", {}) or {}).get("vcfa", {}) or {}
    merged.update(dict(model_vcfa))
    return merged


def normalize_grid_shape(grid: tuple[int, ...] | list[int] | None, token_count: int | None = None) -> tuple[int, int]:
    if grid is not None:
        values = tuple(int(x) for x in grid)
        if len(values) == 3:
            return values[-2], values[-1]
        if len(values) == 2:
            return values
    if token_count is None:
        raise ValueError("grid or token_count is required.")
    side = int(round(float(token_count) ** 0.5))
    if side * side != int(token_count):
        raise ValueError(f"Cannot infer square grid from token_count={token_count}")
    return side, side


def align_relevance_grid(
    q_sem: torch.Tensor,
    *,
    source_grid: tuple[int, int] | tuple[int, int, int] | list[int] | None,
    target_grid: tuple[int, int],
) -> torch.Tensor:
    if q_sem.ndim == 4:
        flat = q_sem.reshape(q_sem.shape[0] * q_sem.shape[1], 1, q_sem.shape[2], q_sem.shape[3])
        bsz, time = q_sem.shape[:2]
    elif q_sem.ndim == 3:
        bsz, time, token_count = q_sem.shape
        h, w = normalize_grid_shape(source_grid, token_count)
        flat = q_sem.reshape(bsz * time, 1, h, w)
    else:
        raise ValueError(f"Expected q_sem [B,T,K] or [B,T,H,W], got {tuple(q_sem.shape)}")
    target_grid = tuple(int(x) for x in target_grid)
    if tuple(flat.shape[-2:]) != target_grid:
        flat = F.interpolate(flat.float(), size=target_grid, mode="bilinear", align_corners=False)
    return flat.reshape(bsz, time, target_grid[0] * target_grid[1]).clamp(0.0, 1.0)


def align_patch_features_grid(
    patch_features: torch.Tensor,
    *,
    source_grid: tuple[int, int] | tuple[int, int, int] | list[int] | None,
    target_grid: tuple[int, int],
) -> torch.Tensor:
    if patch_features.ndim != 4:
        raise ValueError(f"Expected patch_features [B,T,K,D], got {tuple(patch_features.shape)}")
    bsz, time, token_count, dim = patch_features.shape
    source_h, source_w = normalize_grid_shape(source_grid, token_count)
    x = patch_features.float().reshape(bsz * time, source_h, source_w, dim).permute(0, 3, 1, 2)
    target_grid = tuple(int(v) for v in target_grid)
    if (source_h, source_w) != target_grid:
        x = F.interpolate(x, size=target_grid, mode="area")
    return x.permute(0, 2, 3, 1).reshape(bsz, time, target_grid[0] * target_grid[1], dim)


def compress_feature_dim(features: torch.Tensor, dim: int) -> torch.Tensor:
    dim = int(dim)
    if features.shape[-1] == dim:
        return features.float()
    flat = features.float().reshape(-1, 1, features.shape[-1])
    pooled = F.adaptive_avg_pool1d(flat, dim)
    return pooled.reshape(*features.shape[:-1], dim)


def grid_centers(grid_shape: tuple[int, int], *, device, dtype=torch.float32) -> torch.Tensor:
    h, w = tuple(int(x) for x in grid_shape)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype),
        torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).reshape(h * w, 2)
