from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformSpec:
    frame_indices: tuple[int, ...]
    crop_y: int = 0
    crop_x: int = 0
    crop_h: int | None = None
    crop_w: int | None = None
    flip: bool = False
    image_size: int = 224


class TransformSampler:
    def __init__(self, *, num_frames: int, image_size: int, seed: int = 0, random_flip: bool = False):
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.rng = np.random.default_rng(seed)
        self.random_flip = bool(random_flip)

    def sample(self, length: int, height: int, width: int) -> TransformSpec:
        if length <= 0:
            raise ValueError("Cannot sample transform from an empty video")
        if length == self.num_frames:
            indices = tuple(range(length))
        else:
            indices = tuple(int(x) for x in np.linspace(0, length - 1, self.num_frames).round())
        side = min(height, width)
        crop_y = max(0, (height - side) // 2)
        crop_x = max(0, (width - side) // 2)
        flip = bool(self.random_flip and self.rng.random() < 0.5)
        return TransformSpec(indices, crop_y, crop_x, side, side, flip, self.image_size)


def load_npz_frames(path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as data:
        if "frames" not in data.files:
            raise KeyError(f"{path} does not contain a 'frames' array")
        return assert_video(data["frames"])


def load_task_mask(path) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(str(path), allow_pickle=False) as data:
        key = "task_foreground" if "task_foreground" in data.files else data.files[0]
        mask = np.asarray(data[key], dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., None]
    return mask


def assert_video(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def apply_video_transform(frames: np.ndarray, spec: TransformSpec):
    import torch
    import torch.nn.functional as F

    frames = assert_video(frames)
    sampled = frames[list(spec.frame_indices)]
    crop_h = spec.crop_h or sampled.shape[1]
    crop_w = spec.crop_w or sampled.shape[2]
    sampled = sampled[:, spec.crop_y : spec.crop_y + crop_h, spec.crop_x : spec.crop_x + crop_w]
    if spec.flip:
        sampled = sampled[:, :, ::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(sampled)).float().permute(0, 3, 1, 2) / 255.0
    tensor = F.interpolate(tensor, size=(int(spec.image_size), int(spec.image_size)), mode="bicubic", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def apply_mask_transform(mask: np.ndarray | None, spec: TransformSpec):
    if mask is None:
        return None
    import torch
    import torch.nn.functional as F

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., None]
    sampled = mask[list(spec.frame_indices)]
    crop_h = spec.crop_h or sampled.shape[1]
    crop_w = spec.crop_w or sampled.shape[2]
    sampled = sampled[:, spec.crop_y : spec.crop_y + crop_h, spec.crop_x : spec.crop_x + crop_w]
    if spec.flip:
        sampled = sampled[:, :, ::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(sampled)).float().permute(0, 3, 1, 2)
    return F.interpolate(tensor, size=(int(spec.image_size), int(spec.image_size)), mode="nearest")


def collate_transform_specs(specs: Sequence[TransformSpec]) -> list[dict]:
    return [dict(spec.__dict__) for spec in specs]

