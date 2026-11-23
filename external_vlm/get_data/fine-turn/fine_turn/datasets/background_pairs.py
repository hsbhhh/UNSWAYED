from __future__ import annotations

from pathlib import Path

from fine_turn.datasets.metadata_adapter import PairSample, background_sample
from fine_turn.datasets.transforms import (
    TransformSampler,
    apply_mask_transform,
    apply_video_transform,
    load_npz_frames,
    load_task_mask,
)


class BackgroundPairDataset:
    def __init__(self, root: Path, *, num_frames: int = 8, image_size: int = 224, samples: list[PairSample] | None = None):
        self.root = Path(root)
        self.samples = samples if samples is not None else discover_background_pairs(self.root)
        self.transform_sampler = TransformSampler(num_frames=num_frames, image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        anchor = load_npz_frames(sample.anchor_path)
        paired = load_npz_frames(sample.paired_path)
        if anchor.shape[:3] != paired.shape[:3]:
            raise ValueError(f"Pair shape mismatch for {sample.pair_id}: {anchor.shape} vs {paired.shape}")
        spec = self.transform_sampler.sample(anchor.shape[0], anchor.shape[1], anchor.shape[2])
        mask = load_task_mask(sample.task_mask_path)
        return {
            "sample": sample,
            "anchor": apply_video_transform(anchor, spec),
            "paired": apply_video_transform(paired, spec),
            "task_mask": apply_mask_transform(mask, spec),
            "transform_spec": spec,
        }


def discover_background_pairs(root: Path) -> list[PairSample]:
    root = Path(root)
    if not root.exists():
        return []
    samples = []
    for pair_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        sample = background_sample(pair_dir)
        if sample.anchor_path.exists() and sample.paired_path.exists():
            samples.append(sample)
    return samples

