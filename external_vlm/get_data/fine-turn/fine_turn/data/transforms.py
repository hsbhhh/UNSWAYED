from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PairedTransformSpec:
    frame_indices: tuple[int, ...]
    image_size: int = 224
    normalize: bool = True


def sample_frame_indices(length: int, num_frames: int) -> tuple[int, ...]:
    length = int(length)
    num_frames = int(num_frames)
    if length <= 0:
        raise ValueError("Cannot sample frames from an empty video.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if length == num_frames:
        return tuple(range(length))
    return tuple(int(x) for x in np.linspace(0, length - 1, num_frames).round())


def make_transform_spec(length: int, *, num_frames: int = 8, image_size: int = 224, normalize: bool = True) -> PairedTransformSpec:
    return PairedTransformSpec(sample_frame_indices(length, num_frames), int(image_size), bool(normalize))


def video_to_tensor(frames: np.ndarray | torch.Tensor, spec: PairedTransformSpec) -> torch.Tensor:
    tensor = _as_video_tensor(frames)
    tensor = tensor[list(spec.frame_indices)]
    tensor = F.interpolate(tensor, size=(int(spec.image_size), int(spec.image_size)), mode="bicubic", align_corners=False)
    tensor = tensor.clamp(0.0, 1.0)
    if spec.normalize:
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def mask_to_tensor(mask: np.ndarray | torch.Tensor | None, spec: PairedTransformSpec, *, threshold: float | None = None) -> torch.Tensor | None:
    if mask is None:
        return None
    tensor = _as_mask_tensor(mask)
    tensor = tensor[list(spec.frame_indices)]
    tensor = F.interpolate(tensor, size=(int(spec.image_size), int(spec.image_size)), mode="nearest")
    tensor = tensor.clamp(0.0, 1.0)
    if threshold is not None:
        tensor = (tensor >= float(threshold)).float()
    return tensor


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
    return (video * std + mean).clamp(0.0, 1.0)


def frame_indices_to_json(indices: Sequence[int]) -> list[int]:
    return [int(x) for x in indices]


def _as_video_tensor(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim != 4:
            raise ValueError(f"Expected video with 4 dims, got {arr.shape}.")
        if arr.shape[-1] == 3:
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
        elif arr.shape[1] == 3:
            tensor = torch.from_numpy(np.ascontiguousarray(arr))
        else:
            raise ValueError(f"Expected RGB video [T,H,W,3] or [T,3,H,W], got {arr.shape}.")
    else:
        tensor = frames.detach().clone() if frames.requires_grad else frames
        if tensor.ndim != 4:
            raise ValueError(f"Expected video with 4 dims, got {tuple(tensor.shape)}.")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 3:
            raise ValueError(f"Expected RGB video [T,H,W,3] or [T,3,H,W], got {tuple(tensor.shape)}.")
    tensor = tensor.float()
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    return tensor.contiguous()


def _as_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., None]
        if arr.ndim != 4:
            raise ValueError(f"Expected mask [T,H,W] or [T,H,W,1], got {arr.shape}.")
        if arr.shape[-1] == 1:
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
        elif arr.shape[1] == 1:
            tensor = torch.from_numpy(np.ascontiguousarray(arr))
        else:
            raise ValueError(f"Expected mask with one channel, got {arr.shape}.")
    else:
        tensor = mask.detach().clone() if mask.requires_grad else mask
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(f"Expected mask [T,H,W] or [T,1,H,W], got {tuple(tensor.shape)}.")
        if tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 1:
            raise ValueError(f"Expected mask with one channel, got {tuple(tensor.shape)}.")
    return tensor.float().contiguous()

