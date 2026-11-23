from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


MASK_KEY_ALIASES = {
    "task": ("task_foreground", "task_mask", "union_mask", "foreground", "mask"),
    "ball": ("ball_mask", "ball", "object_ball"),
    "cup": ("cup_mask", "cup", "object_cup"),
    "rope": ("rope_mask", "rope", "string_mask", "object_rope"),
    "agent": ("agent_mask", "agent", "robot_mask"),
    "target": ("target_mask", "target", "goal_mask"),
}


@dataclass(frozen=True)
class LoadedMasks:
    task: np.ndarray | None
    ball: np.ndarray | None
    cup: np.ndarray | None
    rope: np.ndarray | None
    agent: np.ndarray | None
    target: np.ndarray | None
    explicit_object_masks: bool


def load_mask_bundle(path: Path | None) -> LoadedMasks:
    if path is None or not Path(path).exists():
        return LoadedMasks(None, None, None, None, None, None, False)
    with np.load(str(path), allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    task = _first_present(arrays, MASK_KEY_ALIASES["task"])
    ball = _first_present(arrays, MASK_KEY_ALIASES["ball"])
    cup = _first_present(arrays, MASK_KEY_ALIASES["cup"])
    rope = _first_present(arrays, MASK_KEY_ALIASES["rope"])
    agent = _first_present(arrays, MASK_KEY_ALIASES["agent"])
    target = _first_present(arrays, MASK_KEY_ALIASES["target"])
    return LoadedMasks(
        task,
        ball,
        cup,
        rope,
        agent,
        target,
        any(x is not None for x in (ball, cup, rope, agent, target)),
    )


def heuristic_object_masks(task_mask: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    if task_mask is None:
        return None, None, None, torch.tensor(0.0)
    mask = task_mask.float().clamp(0.0, 1.0)
    frames = []
    ball_frames = []
    cup_frames = []
    rope_frames = []
    confidence = []
    for frame in mask:
        fg = frame[0] > 0.25
        if int(fg.sum()) < 8:
            ball_frames.append(torch.zeros_like(frame))
            cup_frames.append(torch.zeros_like(frame))
            rope_frames.append(torch.zeros_like(frame))
            confidence.append(0.0)
            continue
        ys, xs = torch.where(fg)
        y_mid = torch.quantile(ys.float(), 0.45)
        x_med = torch.median(xs.float())
        height = frame.shape[-2]
        width = frame.shape[-1]
        yy = torch.arange(height, device=frame.device).view(height, 1)
        xx = torch.arange(width, device=frame.device).view(1, width)
        ball = fg & (yy <= y_mid)
        cup = fg & (yy > y_mid)
        rope_band = max(1.0, float(width) * 0.025)
        rope = fg & ((xx.float() - x_med).abs() <= rope_band)
        ball_frames.append(ball.float().unsqueeze(0))
        cup_frames.append(cup.float().unsqueeze(0))
        rope_frames.append(rope.float().unsqueeze(0))
        confidence.append(float(min(1.0, fg.float().mean().item() * 25.0)))
    frames.clear()
    conf = torch.tensor(float(np.mean(confidence)) if confidence else 0.0, device=mask.device)
    return torch.stack(ball_frames), torch.stack(cup_frames), torch.stack(rope_frames), conf


def build_patch_targets(
    batch: Mapping[str, torch.Tensor],
    *,
    time_steps: int,
    grid_size: tuple[int, int] = (8, 8),
    native_grid_size: tuple[int, int] | None = None,
    dilation: int = 1,
) -> dict[str, torch.Tensor]:
    task = _prepare_mask(batch.get("task_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    ball = _prepare_mask(batch.get("ball_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    cup = _prepare_mask(batch.get("cup_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    rope = _prepare_mask(batch.get("rope_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    agent = _prepare_mask(batch.get("agent_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    target_component = _prepare_mask(batch.get("target_mask"), time_steps=time_steps, grid_size=grid_size, dilation=dilation)
    native = None
    if native_grid_size is not None:
        native = _prepare_mask(batch.get("task_mask"), time_steps=time_steps, grid_size=native_grid_size, dilation=dilation)
    valid = batch.get("object_valid")
    if valid is None:
        device = task.device if task is not None else torch.device("cpu")
        batch_size = 1 if task is None else task.shape[0]
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return {
        "task": torch.zeros(0) if task is None else task,
        "ball": torch.zeros(0) if ball is None else ball,
        "cup": torch.zeros(0) if cup is None else cup,
        "rope": torch.zeros(0) if rope is None else rope,
        "agent": torch.zeros(0) if agent is None else agent,
        "target_component": torch.zeros(0) if target_component is None else target_component,
        "native_task": torch.zeros(0) if native is None else native,
        "object_valid": valid.bool(),
    }


def resize_time(mask: torch.Tensor, time_steps: int) -> torch.Tensor:
    if mask.shape[1] == int(time_steps):
        return mask
    idx = torch.linspace(0, mask.shape[1] - 1, int(time_steps), device=mask.device).round().long()
    return mask.index_select(1, idx)


def downsample_mask(mask: torch.Tensor, grid_size: tuple[int, int], *, dilation: int = 0) -> torch.Tensor:
    bsz, time, channels, height, width = mask.shape
    flat = mask.reshape(bsz * time, channels, height, width).float()
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        flat = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=int(dilation))
    pooled = F.adaptive_avg_pool2d(flat, tuple(int(x) for x in grid_size))
    return pooled.reshape(bsz, time, *tuple(int(x) for x in grid_size)).clamp(0.0, 1.0)


def _prepare_mask(mask: torch.Tensor | None, *, time_steps: int, grid_size: tuple[int, int], dilation: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.ndim != 5:
        raise ValueError(f"Expected mask [B,T,1,H,W], got {tuple(mask.shape)}.")
    mask = resize_time(mask.float().clamp(0.0, 1.0), time_steps)
    return downsample_mask(mask, grid_size, dilation=dilation)


def _first_present(arrays: Mapping[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return arrays[name]
    return None
