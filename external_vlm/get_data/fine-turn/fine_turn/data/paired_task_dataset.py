from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from fine_turn.data.mask_targets import heuristic_object_masks, load_mask_bundle
from fine_turn.data.transforms import make_transform_spec, mask_to_tensor, video_to_tensor


@dataclass(frozen=True)
class PairedTaskRecord:
    pair_id: str
    clean_video_path: str
    davis_video_path: str
    task_mask_path: str | None = None
    ball_mask_path: str | None = None
    cup_mask_path: str | None = None
    rope_mask_path: str | None = None
    agent_mask_path: str | None = None
    target_mask_path: str | None = None
    metadata_path: str | None = None
    episode_id: str | None = None
    davis_video_id: str | None = None
    split: str | None = None


class PairedTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        pair_root: Path | None = None,
        manifest_path: Path | None = None,
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 224,
        max_items: int | None = None,
        synthetic_if_empty: bool = False,
    ):
        records = load_records(pair_root=pair_root, manifest_path=manifest_path)
        if split != "all":
            records = [record for record in records if (record.split or "train") == split]
        if max_items is not None:
            records = records[: int(max_items)]
        if not records and synthetic_if_empty:
            records = synthetic_records(max_items or 8)
        self.records = records
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        clean = _load_video(record.clean_video_path)
        davis = _load_video(record.davis_video_path)
        if clean.shape[0] != davis.shape[0]:
            length = min(clean.shape[0], davis.shape[0])
            clean = clean[:length]
            davis = davis[:length]
        spec = make_transform_spec(clean.shape[0], num_frames=self.num_frames, image_size=self.image_size, normalize=True)
        mask_bundle = load_mask_bundle(Path(record.task_mask_path)) if record.task_mask_path else load_mask_bundle(None)
        metadata = _read_json(Path(record.metadata_path)) if record.metadata_path else {}
        source_indices = metadata.get("frame_indices")
        task = mask_to_tensor(mask_bundle.task, spec, threshold=None)
        ball = _load_explicit_or_bundle(record.ball_mask_path, mask_bundle.ball, spec)
        cup = _load_explicit_or_bundle(record.cup_mask_path, mask_bundle.cup, spec)
        rope = _load_explicit_or_bundle(record.rope_mask_path, mask_bundle.rope, spec)
        agent_bundle = load_mask_bundle(Path(record.agent_mask_path)) if record.agent_mask_path else mask_bundle
        target_bundle = (
            agent_bundle
            if record.target_mask_path and record.target_mask_path == record.agent_mask_path
            else load_mask_bundle(Path(record.target_mask_path)) if record.target_mask_path else mask_bundle
        )
        agent = _component_to_tensor(agent_bundle.agent, spec, source_indices)
        target_component = _component_to_tensor(target_bundle.target, spec, source_indices)
        explicit = bool(
            mask_bundle.explicit_object_masks
            or record.ball_mask_path
            or record.cup_mask_path
            or record.rope_mask_path
            or record.agent_mask_path
            or record.target_mask_path
        )
        if task is None:
            task = _foreground_from_frame_difference(clean, davis, spec)
        if ball is None or cup is None or rope is None:
            h_ball, h_cup, h_rope, confidence = heuristic_object_masks(task)
            ball = ball if ball is not None else h_ball
            cup = cup if cup is not None else h_cup
            rope = rope if rope is not None else h_rope
        else:
            confidence = torch.tensor(1.0)
        agent = torch.zeros_like(task) if agent is None else agent
        target_component = torch.zeros_like(task) if target_component is None else target_component
        return {
            "clean_video": video_to_tensor(clean, spec),
            "davis_video": video_to_tensor(davis, spec),
            "task_mask": task,
            "ball_mask": ball,
            "cup_mask": cup,
            "rope_mask": rope,
            "agent_mask": agent,
            "target_mask": target_component,
            "object_valid": torch.tensor(bool(explicit), dtype=torch.bool),
            "heuristic_confidence": confidence.float(),
            "episode_id": record.episode_id or record.pair_id,
            "pair_id": record.pair_id,
            "frame_indices": torch.tensor(spec.frame_indices, dtype=torch.long),
            "record": record,
        }


def paired_task_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clean_video": torch.stack([item["clean_video"] for item in batch], dim=0),
        "davis_video": torch.stack([item["davis_video"] for item in batch], dim=0),
        "task_mask": torch.stack([item["task_mask"] for item in batch], dim=0),
        "ball_mask": torch.stack([item["ball_mask"] for item in batch], dim=0),
        "cup_mask": torch.stack([item["cup_mask"] for item in batch], dim=0),
        "rope_mask": torch.stack([item["rope_mask"] for item in batch], dim=0),
        "agent_mask": torch.stack([item["agent_mask"] for item in batch], dim=0),
        "target_mask": torch.stack([item["target_mask"] for item in batch], dim=0),
        "object_valid": torch.stack([item["object_valid"] for item in batch], dim=0),
        "heuristic_confidence": torch.stack([item["heuristic_confidence"] for item in batch], dim=0),
        "episode_id": [item["episode_id"] for item in batch],
        "pair_id": [item["pair_id"] for item in batch],
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "records": [item["record"] for item in batch],
    }


def load_records(*, pair_root: Path | None = None, manifest_path: Path | None = None) -> list[PairedTaskRecord]:
    if manifest_path is not None and Path(manifest_path).exists():
        return records_from_manifest(Path(manifest_path))
    if pair_root is None:
        return []
    return discover_paired_task_records(Path(pair_root))


def records_from_manifest(path: Path) -> list[PairedTaskRecord]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("items", raw) if isinstance(raw, Mapping) else raw
    return [PairedTaskRecord(**_normalize_record(row, base=path.parent)) for row in rows]


def write_manifest(path: Path, records: Sequence[PairedTaskRecord], *, metadata: Mapping[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "ball_in_cup_patch_relevance_manifest",
        "metadata": dict(metadata or {}),
        "items": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_paired_task_records(pair_root: Path) -> list[PairedTaskRecord]:
    root = Path(pair_root)
    candidates = []
    if (root / "background").exists():
        candidates.append(root / "background")
    elif (root / "pairs" / "background").exists():
        candidates.append(root / "pairs" / "background")
    else:
        candidates.append(root)
    records = []
    for bg_root in candidates:
        if not bg_root.exists():
            continue
        for pair_dir in sorted(path for path in bg_root.iterdir() if path.is_dir()):
            clean = pair_dir / "clean_frames.npz"
            davis = pair_dir / "distracted_frames.npz"
            if not clean.exists() or not davis.exists():
                continue
            meta_path = pair_dir / "metadata.json"
            meta = _read_json(meta_path)
            task_mask = pair_dir / "task_masks.npz"
            records.append(
                PairedTaskRecord(
                    pair_id=str(meta.get("pair_id") or pair_dir.name),
                    clean_video_path=str(clean),
                    davis_video_path=str(davis),
                    task_mask_path=str(task_mask) if task_mask.exists() else None,
                    ball_mask_path=_optional_existing(pair_dir / "ball_mask.npz"),
                    cup_mask_path=_optional_existing(pair_dir / "cup_mask.npz"),
                    rope_mask_path=_optional_existing(pair_dir / "rope_mask.npz"),
                    agent_mask_path=_optional_existing(pair_dir / "agent_mask.npz"),
                    target_mask_path=_optional_existing(pair_dir / "target_mask.npz"),
                    metadata_path=str(meta_path) if meta_path.exists() else None,
                    episode_id=_optional_str(meta.get("episode_id") or meta.get("physics_trajectory_id")),
                    davis_video_id=_optional_str(meta.get("davis_video_id")),
                    split=_optional_str(meta.get("split")),
                )
            )
    return records


def assign_episode_splits(
    records: Sequence[PairedTaskRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> list[PairedTaskRecord]:
    del test_ratio
    train_ratio = float(train_ratio)
    val_cut = train_ratio + float(val_ratio)
    out = []
    for record in records:
        key = f"{seed}:{record.episode_id or record.pair_id}"
        value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "train" if value < train_ratio else "val" if value < val_cut else "test"
        out.append(PairedTaskRecord(**{**asdict(record), "split": record.split or split}))
    return out


def synthetic_records(count: int = 8) -> list[PairedTaskRecord]:
    return [
        PairedTaskRecord(
            pair_id=f"synthetic_{idx:04d}",
            clean_video_path=f"synthetic://clean/{idx}",
            davis_video_path=f"synthetic://davis/{idx}",
            episode_id=f"synthetic_episode_{idx // 2}",
            split="train",
        )
        for idx in range(int(count))
    ]


def _load_video(path: str | Path) -> np.ndarray:
    raw = str(path)
    if raw.startswith("synthetic://"):
        return _synthetic_video(seed=int(raw.rsplit("/", 1)[-1]), distracted="davis" in raw)
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as data:
        key = "frames" if "frames" in data.files else data.files[0]
        frames = np.asarray(data[key])
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3] in {path}, got {frames.shape}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _synthetic_video(*, seed: int, distracted: bool) -> np.ndarray:
    rng = np.random.default_rng(seed + (1000 if distracted else 0))
    frames = rng.integers(0, 40, size=(8, 64, 64, 3), dtype=np.uint8)
    for t in range(8):
        x = 20 + t * 2
        frames[t, 10 + t : 18 + t, x : x + 8] = [220, 235, 245]
        frames[t, 34:52, 22:44] = [80, 90, 105]
        frames[t, 18 + t : 34, x + 3 : x + 5] = [180, 190, 200]
        if distracted:
            frames[t, :, :, 1] = np.maximum(frames[t, :, :, 1], rng.integers(80, 180, size=(64, 64), dtype=np.uint8))
    return frames


def _foreground_from_frame_difference(clean: np.ndarray, davis: np.ndarray, spec) -> torch.Tensor:
    clean_t = video_to_tensor(clean, spec)
    davis_t = video_to_tensor(davis, spec)
    diff = (clean_t - davis_t).abs().mean(dim=1, keepdim=True)
    threshold = torch.quantile(diff.flatten(), 0.75)
    return (diff <= threshold).float()


def _load_explicit_or_bundle(path: str | None, bundle_array: np.ndarray | None, spec):
    if path:
        bundle = load_mask_bundle(Path(path))
        array = _first_array(bundle.task, bundle.ball, bundle.cup, bundle.rope, bundle.agent, bundle.target)
        return mask_to_tensor(array, spec, threshold=None)
    return mask_to_tensor(bundle_array, spec, threshold=None)


def _component_to_tensor(array: np.ndarray | None, spec, source_indices):
    if array is None:
        return None
    if source_indices and int(array.shape[0]) != len(source_indices):
        indices = np.asarray(source_indices, dtype=np.int64)
        array = array[indices]
    return mask_to_tensor(array, spec, threshold=None)


def _first_array(*arrays: np.ndarray | None) -> np.ndarray | None:
    for array in arrays:
        if array is not None:
            return array
    return None


def _normalize_record(row: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    aliases = {
        "clean_video": "clean_video_path",
        "clean_path": "clean_video_path",
        "anchor_path": "clean_video_path",
        "davis_video": "davis_video_path",
        "distracted_path": "davis_video_path",
        "paired_path": "davis_video_path",
        "task_mask": "task_mask_path",
    }
    data = {aliases.get(key, key): value for key, value in row.items()}
    for key in (
        "clean_video_path",
        "davis_video_path",
        "task_mask_path",
        "ball_mask_path",
        "cup_mask_path",
        "rope_mask_path",
        "agent_mask_path",
        "target_mask_path",
        "metadata_path",
    ):
        if data.get(key) and "://" not in str(data[key]):
            path = Path(str(data[key]))
            data[key] = str(path if path.is_absolute() else (base / path).resolve())
    return data


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_existing(path: Path) -> str | None:
    return str(path) if path.exists() else None
