from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DavisFrame:
    image: np.ndarray
    video_id: str
    frame_id: int | str
    frame_index: int
    video_length: int


class DavisFrameStore:
    """DAVIS videos indexed as ordered per-video sequences."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int,
        allowed_video_ids: Iterable[str] | None = None,
    ):
        self.root = Path(root).expanduser()
        self.image_size = int(image_size)
        if not self.root.exists():
            raise RuntimeError(f"DAVIS backend is unavailable; root does not exist: {self.root}")
        discovered = self._discover()
        if allowed_video_ids is not None:
            requested = tuple(dict.fromkeys(str(value) for value in allowed_video_ids))
            if not requested:
                raise RuntimeError("DAVIS allowed_video_ids must contain at least one video")
            unknown = sorted(set(requested) - set(discovered))
            if unknown:
                raise RuntimeError(
                    "DAVIS allowed_video_ids contains unknown videos: "
                    f"{unknown}; available={sorted(discovered)}"
                )
            self.videos = {video_id: discovered[video_id] for video_id in requested}
        else:
            self.videos = discovered
        self.video_ids = tuple(sorted(self.videos))
        self.frames = [
            (video_id, frame_id, path)
            for video_id in self.video_ids
            for frame_id, path in self.videos[video_id]
        ]
        if len(self.frames) < 3:
            raise RuntimeError(f"DAVIS backend needs at least 3 frames, found {len(self.frames)} in {self.root}")

    def choose_video(self, rng: np.random.Generator, *, exclude_video_ids: set[str] | None = None) -> str:
        excluded = set(exclude_video_ids or ())
        candidates = [video_id for video_id in self.video_ids if video_id not in excluded]
        if not candidates:
            raise RuntimeError(
                "DAVIS cannot satisfy the requested distinct-video assignment; "
                f"available={len(self.video_ids)}, excluded={sorted(excluded)}"
            )
        return candidates[int(rng.integers(0, len(candidates)))]

    def read(self, video_id: str, frame_index: int) -> DavisFrame:
        rows = self.videos[str(video_id)]
        index = int(frame_index)
        if not 0 <= index < len(rows):
            raise IndexError(f"DAVIS frame index {index} is outside video {video_id!r} with length {len(rows)}")
        frame_id, path = rows[index]
        return DavisFrame(
            image=_read_rgb(path, self.image_size),
            video_id=str(video_id),
            frame_id=frame_id,
            frame_index=index,
            video_length=len(rows),
        )

    def video_length(self, video_id: str) -> int:
        return len(self.videos[str(video_id)])

    def sample(self, rng: np.random.Generator, *, exclude_video_ids: set[str] | None = None) -> DavisFrame:
        """Compatibility helper; production paired views use stateful streams."""

        video_id = self.choose_video(rng, exclude_video_ids=exclude_video_ids)
        return self.read(video_id, int(rng.integers(0, self.video_length(video_id))))

    def _discover(self) -> dict[str, tuple[tuple[int | str, Path], ...]]:
        suffixes = {".jpg", ".jpeg", ".png"}
        grouped: dict[str, list[tuple[int | str, Path]]] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            video_id = path.parent.relative_to(self.root).as_posix()
            frame_id: int | str
            try:
                frame_id = int(path.stem)
            except ValueError:
                frame_id = path.stem
            grouped.setdefault(video_id, []).append((frame_id, path))
        return {
            video_id: tuple(sorted(rows, key=lambda row: _frame_sort_key(row[0], row[1])))
            for video_id, rows in grouped.items()
            if rows
        }


def compose_background(clean: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    out = mask * clean + (1.0 - mask) * background
    return np.clip(out, 0, 255).astype(np.uint8)


def _frame_sort_key(frame_id: int | str, path: Path) -> tuple:
    if isinstance(frame_id, int):
        return (0, frame_id, path.name)
    parts = tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(frame_id)))
    return (1, parts, path.name)


def _read_rgb(path: Path, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("DAVIS image loading requires Pillow on the server.") from exc
    with Image.open(path) as image:
        image = image.convert("RGB").resize((int(image_size), int(image_size)))
        return np.asarray(image, dtype=np.uint8)
