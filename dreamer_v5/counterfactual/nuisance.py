from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_LINALG_CHUNK_SIZE = 16_384


@dataclass
class NuisanceStatistics:
    deviations: torch.Tensor
    gram: torch.Tensor
    eigenvalues: torch.Tensor
    scale: torch.Tensor
    effective_rank: torch.Tensor
    gains: torch.Tensor
    linalg_chunk_size: int
    eigensolver_split_retries: int


def geometric_median(
    features: torch.Tensor,
    *,
    epsilon: float,
    max_iters: int,
    convergence_tol: float,
) -> torch.Tensor:
    """Deterministic Weiszfeld center over K for `[B,T,K,N,D]` features."""

    points = torch.as_tensor(features).float().movedim(2, -2)
    if points.ndim != 5:
        raise ValueError(f"features must be [B,T,K,N,D], got {tuple(features.shape)}")
    center = points.mean(dim=-2)
    eps = float(epsilon)
    for _ in range(max(1, int(max_iters))):
        distance = (points - center.unsqueeze(-2)).norm(dim=-1).clamp_min(eps)
        weight = distance.reciprocal()
        updated = (points * weight.unsqueeze(-1)).sum(dim=-2) / weight.sum(dim=-1, keepdim=True).clamp_min(eps)
        shift = (updated - center).norm(dim=-1)
        center = updated
        if bool(torch.all(shift <= float(convergence_tol))):
            break
    if not bool(torch.isfinite(center).all()):
        raise FloatingPointError("geometric median produced non-finite values")
    return center


def nuisance_statistics(
    features: torch.Tensor,
    center: torch.Tensor,
    *,
    epsilon: float,
    eigen_epsilon: float,
    rank_tolerance: float,
    linalg_chunk_size: int = DEFAULT_LINALG_CHUNK_SIZE,
) -> NuisanceStatistics:
    """Build only the KxK Gram representation of the nuisance subspace."""

    points = torch.as_tensor(features).float().movedim(2, -2)
    center = torch.as_tensor(center, device=points.device).float()
    deviations = points - center.unsqueeze(-2)
    count = int(deviations.shape[-2])
    gram = torch.matmul(deviations, deviations.transpose(-1, -2)) / float(count)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("KxK nuisance Gram matrix contains NaN or Inf")
    eigenvalues, split_retries = _chunked_eigvalsh(
        gram,
        chunk_size=int(linalg_chunk_size),
    )
    eigenvalues = eigenvalues.clamp_min(0.0)
    scale = _median_nonzero(eigenvalues, threshold=float(eigen_epsilon), fallback=1.0)
    largest = eigenvalues.amax(dim=-1, keepdim=True)
    effective_rank = (
        eigenvalues > torch.maximum(
            torch.full_like(largest, float(eigen_epsilon)),
            largest * float(rank_tolerance),
        )
    ).sum(dim=-1)
    base = scale.unsqueeze(-1) + float(epsilon)
    gains = base / (base + eigenvalues)
    return NuisanceStatistics(
        deviations=deviations,
        gram=gram,
        eigenvalues=eigenvalues,
        scale=scale,
        effective_rank=effective_rank,
        gains=gains,
        linalg_chunk_size=min(int(linalg_chunk_size), int(gram.numel() // max(count * count, 1))),
        eigensolver_split_retries=int(split_retries),
    )


def apply_soft_nuisance_metric(
    residual: torch.Tensor,
    deviations: torch.Tensor,
    scale: torch.Tensor,
    *,
    epsilon: float,
    linalg_chunk_size: int = DEFAULT_LINALG_CHUNK_SIZE,
) -> torch.Tensor:
    """Return `r^T S r` through a KxK Woodbury solve, never a DxD matrix."""

    residual = torch.as_tensor(residual).float()
    deviations = torch.as_tensor(deviations, device=residual.device).float()
    if deviations.shape[:-2] != residual.shape[:-1] or deviations.shape[-1] != residual.shape[-1]:
        raise ValueError(
            f"residual/deviation shape mismatch: residual={tuple(residual.shape)}, "
            f"deviations={tuple(deviations.shape)}"
        )
    count = int(deviations.shape[-2])
    gram = torch.matmul(deviations, deviations.transpose(-1, -2)) / float(count)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    cross = torch.matmul(deviations, residual.unsqueeze(-1)).squeeze(-1) / (float(count) ** 0.5)
    eye = torch.eye(count, dtype=gram.dtype, device=gram.device)
    base = torch.as_tensor(scale, device=gram.device, dtype=gram.dtype).unsqueeze(-1).unsqueeze(-1)
    system = gram + (base + float(epsilon)) * eye
    if not bool(torch.isfinite(system).all() and torch.isfinite(cross).all()):
        raise FloatingPointError("KxK nuisance solve contains NaN or Inf")
    solve, _ = _chunked_solve(
        system,
        cross.unsqueeze(-1),
        chunk_size=int(linalg_chunk_size),
    )
    solve = solve.squeeze(-1)
    correction = (cross * solve).sum(dim=-1)
    q = residual.square().sum(dim=-1) - correction
    return q.clamp_min(0.0)


def _median_nonzero(eigenvalues: torch.Tensor, *, threshold: float, fallback: float) -> torch.Tensor:
    values = torch.as_tensor(eigenvalues).float()
    positive = values > float(threshold)
    count = positive.sum(dim=-1)
    sentinel = torch.full_like(values, torch.inf)
    sorted_values = torch.sort(torch.where(positive, values, sentinel), dim=-1).values
    index = ((count - 1).clamp_min(0) // 2).unsqueeze(-1)
    median = sorted_values.gather(-1, index).squeeze(-1)
    return torch.where(count > 0, median, torch.full_like(median, float(fallback)))


def _chunked_eigvalsh(matrix: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, int]:
    """Evaluate small symmetric matrices below CUDA's large batched-solver boundary."""

    value = torch.as_tensor(matrix)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError(f"eigvalsh expects [...,K,K], got {tuple(value.shape)}")
    size = _validate_chunk_size(chunk_size)
    matrix_size = int(value.shape[-1])
    leading_shape = tuple(value.shape[:-2])
    flat = value.reshape(-1, matrix_size, matrix_size)
    if not flat.shape[0]:
        return value.new_empty(*leading_shape, matrix_size), 0
    outputs = []
    retries = 0
    for start in range(0, int(flat.shape[0]), size):
        result, chunk_retries = _eigvalsh_with_split(flat[start : start + size].contiguous())
        outputs.append(result)
        retries += int(chunk_retries)
    return torch.cat(outputs, dim=0).reshape(*leading_shape, matrix_size), retries


def _chunked_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Solve batched KxK systems in chunks while preserving autograd through RHS."""

    matrix = torch.as_tensor(matrix)
    rhs = torch.as_tensor(rhs, device=matrix.device, dtype=matrix.dtype)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"solve expects matrix [...,K,K], got {tuple(matrix.shape)}")
    if rhs.shape[:-2] != matrix.shape[:-2] or rhs.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"solve matrix/RHS mismatch: matrix={tuple(matrix.shape)}, rhs={tuple(rhs.shape)}")
    size = _validate_chunk_size(chunk_size)
    matrix_size = int(matrix.shape[-1])
    rhs_width = int(rhs.shape[-1])
    leading_shape = tuple(matrix.shape[:-2])
    flat_matrix = matrix.reshape(-1, matrix_size, matrix_size)
    flat_rhs = rhs.reshape(-1, matrix_size, rhs_width)
    if not flat_matrix.shape[0]:
        return rhs.new_empty(*leading_shape, matrix_size, rhs_width), 0
    outputs = []
    retries = 0
    for start in range(0, int(flat_matrix.shape[0]), size):
        result, chunk_retries = _solve_with_split(
            flat_matrix[start : start + size].contiguous(),
            flat_rhs[start : start + size].contiguous(),
        )
        outputs.append(result)
        retries += int(chunk_retries)
    return torch.cat(outputs, dim=0).reshape(*leading_shape, matrix_size, rhs_width), retries


def _eigvalsh_with_split(matrix: torch.Tensor) -> tuple[torch.Tensor, int]:
    try:
        return torch.linalg.eigvalsh(matrix), 0
    except RuntimeError as exc:
        if not _is_retryable_cuda_linalg_error(exc) or matrix.shape[0] <= 1:
            raise
        midpoint = int(matrix.shape[0]) // 2
        left, left_retries = _eigvalsh_with_split(matrix[:midpoint])
        right, right_retries = _eigvalsh_with_split(matrix[midpoint:])
        return torch.cat((left, right), dim=0), 1 + left_retries + right_retries


def _solve_with_split(matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, int]:
    try:
        return torch.linalg.solve(matrix, rhs), 0
    except RuntimeError as exc:
        if not _is_retryable_cuda_linalg_error(exc) or matrix.shape[0] <= 1:
            raise
        midpoint = int(matrix.shape[0]) // 2
        left, left_retries = _solve_with_split(matrix[:midpoint], rhs[:midpoint])
        right, right_retries = _solve_with_split(matrix[midpoint:], rhs[midpoint:])
        return torch.cat((left, right), dim=0), 1 + left_retries + right_retries


def _is_retryable_cuda_linalg_error(exc: RuntimeError) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in ("CUSOLVER", "CUBLAS_STATUS", "MAGMA"))


def _validate_chunk_size(chunk_size: int) -> int:
    size = int(chunk_size)
    if size < 1:
        raise ValueError(f"linalg_chunk_size must be positive, got {chunk_size}")
    return size
