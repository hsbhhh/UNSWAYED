from __future__ import annotations

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_frames(frames: np.ndarray, num_frames: int) -> tuple[np.ndarray, list[int]]:
    frames = assert_uint8_video(frames)
    if frames.shape[0] == int(num_frames):
        return frames, list(range(frames.shape[0]))
    indices = np.linspace(0, frames.shape[0] - 1, int(num_frames)).round().astype(np.int64)
    return frames[indices], [int(x) for x in indices]


def frames_to_internvideo2_tensor(
    frames: np.ndarray,
    *,
    num_frames: int,
    image_size: int,
    device: str = "cpu",
):
    import torch
    import torch.nn.functional as F

    sampled, indices = sample_frames(frames, num_frames)
    tensor = torch.from_numpy(sampled).to(device=device).float()
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    tensor = F.interpolate(tensor, size=(int(image_size), int(image_size)), mode="bicubic", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), indices


def assert_uint8_video(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected raw frames [T,H,W,C=3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames
