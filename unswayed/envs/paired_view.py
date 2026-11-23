from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .davis import DavisFrame, DavisFrameStore, compose_background


@dataclass
class PairedView:
    clean: np.ndarray
    raw_a: np.ndarray
    raw_b_views: np.ndarray
    foreground_mask: np.ndarray
    background_id_a: str
    background_ids_b: tuple[str, ...]
    view_ids_b: tuple[str, ...]
    paired_view_valid: bool
    metadata: dict[str, Any]

    @property
    def raw_b1(self) -> np.ndarray:
        return self.raw_b_views[0]

    @property
    def raw_b2(self) -> np.ndarray:
        return self.raw_b_views[1]

    @property
    def background_id_b1(self) -> str:
        return self.background_ids_b[0]

    @property
    def background_id_b2(self) -> str:
        return self.background_ids_b[1]

    @property
    def view_id_b1(self) -> str:
        return self.view_ids_b[0]

    @property
    def view_id_b2(self) -> str:
        return self.view_ids_b[1]


@dataclass
class DavisStreamState:
    view: str
    video_id: str
    current_frame_index: int
    start_frame: int
    temporal_stride: int
    episode_id: int
    end_policy: str

    def take(self, store: DavisFrameStore) -> DavisFrame:
        frame = store.read(self.video_id, self.current_frame_index)
        candidate = self.current_frame_index + self.temporal_stride
        if candidate < frame.video_length:
            self.current_frame_index = candidate
        elif self.end_policy == "loop":
            self.current_frame_index = candidate % frame.video_length
        elif self.end_policy == "restart":
            self.current_frame_index = self.start_frame
        else:
            raise ValueError(f"unsupported DAVIS end_policy={self.end_policy!r}")
        return frame


class MockPairedViewGenerator:
    def __init__(self, config, *, env_id: int = 0):
        self.image_size = int(config.env.image_size)
        self.rng = np.random.default_rng(int(config.training.seed) + int(env_id) * 9173)
        self.env_id = int(env_id)
        self.generation = 0
        self.counterfactual_k = int(config.get("counterfactual", {}).get("views", 2))
        configured_ids = config.get("counterfactual", {}).get("view_ids", None)
        self.view_labels = tuple(
            map(str, configured_ids or [f"B{index + 1}" for index in range(self.counterfactual_k)])
        )

    def render(self, clean: np.ndarray, mask: np.ndarray, *, info: dict[str, Any] | None = None) -> PairedView:
        self.generation += 1
        bg_a = self._background(0)
        backgrounds = [self._background(index + 1) for index in range(self.counterfactual_k)]
        raw_a = compose_background(clean, mask, bg_a)
        raw_b_views = np.stack([compose_background(clean, mask, background) for background in backgrounds])
        background_ids = tuple(
            f"mock_{label.lower()}:{self.env_id}:{self.generation}" for label in self.view_labels
        )
        view_ids = tuple(
            f"{label}:mock:{self.env_id}:{self.generation}" for label in self.view_labels
        )
        return PairedView(
            clean=np.asarray(clean, dtype=np.uint8),
            raw_a=raw_a,
            raw_b_views=raw_b_views,
            foreground_mask=np.asarray(mask, dtype=np.float32),
            background_id_a=f"mock_a:{self.env_id}:{self.generation}",
            background_ids_b=background_ids,
            view_ids_b=view_ids,
            paired_view_valid=len({image.tobytes() for image in raw_b_views}) == self.counterfactual_k,
            metadata=dict(info or {}),
        )

    def _background(self, phase: int) -> np.ndarray:
        size = self.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        base = (xx * (phase + 3) + yy * (phase + 5) + int(self.rng.integers(0, 255))) % 255
        return np.stack([(base + 20 * phase) % 255, (base * 2 + 11) % 255, (255 - base)], axis=-1).astype(np.uint8)


class DavisPairedViewGenerator:
    def __init__(
        self,
        config,
        *,
        env_id: int = 0,
        allowed_video_ids: tuple[str, ...] | list[str] | None = None,
    ):
        root = config.env.davis.root or config.paths.davis_root
        if root in (None, "", "null"):
            raise RuntimeError("DAVIS backend is unavailable; set env.davis.root, paths.davis_root, or --davis-root.")
        if allowed_video_ids is None:
            configured_ids = config.env.davis.get("allowed_video_ids")
            allowed_video_ids = None if configured_ids in (None, []) else list(configured_ids)
        self.store = DavisFrameStore(
            root,
            image_size=int(config.env.image_size),
            allowed_video_ids=allowed_video_ids,
        )
        self.env_id = int(env_id)
        self.global_seed = int(config.training.seed)
        self.davis_seed = int(config.env.davis.seed)
        self.temporal_stride = int(config.env.davis.temporal_stride)
        self.end_policy = str(config.env.davis.end_policy)
        self.require_distinct_b_views = bool(config.env.davis.require_distinct_b_views)
        self.counterfactual_k = int(config.get("counterfactual", {}).get("views", 2))
        configured_ids = config.get("counterfactual", {}).get("view_ids", None)
        self.view_labels = tuple(
            map(str, configured_ids or [f"B{index + 1}" for index in range(self.counterfactual_k)])
        )
        self.streams: dict[str, DavisStreamState] = {}
        self.active_episode_id: int | None = None

    def render(self, clean: np.ndarray, mask: np.ndarray, *, info: dict[str, Any] | None = None) -> PairedView:
        source_info = dict(info or {})
        episode_id = int(source_info.get("episode_id", 0))
        if self.active_episode_id != episode_id:
            self._initialize_episode(episode_id)
        frames = {view: self.streams[view].take(self.store) for view in ("A", *self.view_labels)}
        a = frames["A"]
        b_frames = [frames[view] for view in self.view_labels]
        raw_a = compose_background(clean, mask, a.image)
        raw_b_views = np.stack([compose_background(clean, mask, frame.image) for frame in b_frames])
        metadata = dict(source_info)
        metadata.update(self._frame_metadata(frames))
        identities = [(frame.video_id, frame.frame_index) for frame in b_frames]
        return PairedView(
            clean=np.asarray(clean, dtype=np.uint8),
            raw_a=raw_a,
            raw_b_views=raw_b_views,
            foreground_mask=np.asarray(mask, dtype=np.float32),
            background_id_a=f"{a.video_id}:{a.frame_id}",
            background_ids_b=tuple(f"{frame.video_id}:{frame.frame_id}" for frame in b_frames),
            view_ids_b=tuple(
                f"{label}:{frame.video_id}:{frame.frame_id}"
                for label, frame in zip(self.view_labels, b_frames, strict=True)
            ),
            paired_view_valid=len(set(identities)) == self.counterfactual_k,
            metadata=metadata,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "active_episode_id": self.active_episode_id,
            "streams": {view: asdict(state) for view, state in self.streams.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("env_id", self.env_id)) != self.env_id:
            raise ValueError("cannot load DAVIS stream state for a different env_id")
        self.active_episode_id = None if state.get("active_episode_id") is None else int(state["active_episode_id"])
        self.streams = {view: DavisStreamState(**row) for view, row in dict(state.get("streams") or {}).items()}

    def _initialize_episode(self, episode_id: int) -> None:
        available_slots = sum(self.store.video_length(video_id) for video_id in self.store.video_ids)
        if self.require_distinct_b_views and available_slots < self.counterfactual_k:
            raise RuntimeError(
                f"DAVIS formal mode needs at least {self.counterfactual_k} distinct video/frame starts, "
                f"found {available_slots}"
            )
        selected: dict[str, str] = {}
        used_b_starts: set[tuple[str, int]] = set()
        used_b_videos: set[str] = set()
        for view in ("A", *self.view_labels):
            rng = self._rng(episode_id, view)
            excluded: set[str] = set()
            if view != "A" and self.require_distinct_b_views:
                remaining = [video_id for video_id in self.store.video_ids if video_id not in used_b_videos]
                if remaining:
                    excluded = set(used_b_videos)
            video_id = self.store.choose_video(rng, exclude_video_ids=excluded)
            start = int(rng.integers(0, self.store.video_length(video_id)))
            if view != "A" and self.require_distinct_b_views:
                original = start
                while (video_id, start) in used_b_starts:
                    start = (start + 1) % self.store.video_length(video_id)
                    if start == original:
                        candidates = [candidate for candidate in self.store.video_ids if candidate != video_id]
                        if not candidates:
                            raise RuntimeError("DAVIS cannot assign distinct deterministic K-view stream starts")
                        video_id = candidates[int(rng.integers(0, len(candidates)))]
                        start = int(rng.integers(0, self.store.video_length(video_id)))
                        original = start
                used_b_starts.add((video_id, start))
                used_b_videos.add(video_id)
            selected[view] = video_id
            self.streams[view] = DavisStreamState(
                view=view,
                video_id=video_id,
                current_frame_index=start,
                start_frame=start,
                temporal_stride=self.temporal_stride,
                episode_id=int(episode_id),
                end_policy=self.end_policy,
            )
        self.active_episode_id = int(episode_id)

    def _rng(self, episode_id: int, view: str) -> np.random.Generator:
        entropy = [
            self.global_seed & 0xFFFFFFFF,
            self.davis_seed & 0xFFFFFFFF,
            self.env_id & 0xFFFFFFFF,
            int(episode_id) & 0xFFFFFFFF,
            0 if view == "A" else self.view_labels.index(view) + 1,
        ]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def _frame_metadata(self, frames: dict[str, DavisFrame]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "davis_end_policy": self.end_policy,
            "davis_stream_contract": "continuous_per_env_view",
        }
        for view, frame in frames.items():
            key = view.lower()
            stream = self.streams[view]
            out.update(
                {
                    f"background_video_id_{key}": frame.video_id,
                    f"background_frame_index_{key}": int(frame.frame_index),
                    f"background_source_frame_id_{key}": frame.frame_id,
                    f"background_start_frame_{key}": int(stream.start_frame),
                    f"background_temporal_stride_{key}": int(stream.temporal_stride),
                    f"background_episode_id_{key}": int(stream.episode_id),
                    f"background_video_length_{key}": int(frame.video_length),
                }
            )
        return out
