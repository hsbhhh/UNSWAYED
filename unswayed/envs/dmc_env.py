from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ForegroundSegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForegroundMaskResult:
    mask: np.ndarray
    valid: bool
    area: float | None
    failure_reason: str | None = None


@dataclass
class ActionSpace:
    shape: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high).astype(np.float32)


@dataclass
class EnvStep:
    observation: np.ndarray
    reward: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    continue_: float
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_continue(self) -> float:
        return self.continue_


class SyntheticWalkerEnv:
    def __init__(self, config, *, env_id: int = 0):
        self.config = config
        self.env_id = int(env_id)
        self.image_size = int(config.env.image_size)
        self.action_dim = int(config.model.action_dim)
        self.episode_length = int(config.env.synthetic.episode_length)
        self.rng = np.random.default_rng(int(config.training.seed) + self.env_id * 1009)
        self.action_space = ActionSpace(
            shape=(self.action_dim,),
            low=-np.ones(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
        )
        self.episode_id = 0
        self.step_id = 0
        self._state = np.zeros(2, dtype=np.float32)

    def reset(self) -> EnvStep:
        self.episode_id += 1
        self.step_id = 0
        self._state = self.rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        return EnvStep(self.render(), 0.0, True, False, False, 1.0, self._info())

    def step(self, action) -> EnvStep:
        action = np.asarray(action, dtype=np.float32)
        self.step_id += 1
        self._state += np.array([action[0], action[1] if action.size > 1 else 0.0], dtype=np.float32) * 0.03
        self._state = np.clip(self._state, -1.0, 1.0)
        reward = float(1.0 - np.linalg.norm(self._state) * 0.5)
        is_last = self.step_id >= self.episode_length
        return EnvStep(self.render(), reward, False, bool(is_last), False, 1.0, self._info())

    def render(self) -> np.ndarray:
        size = self.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        cx = size * (0.5 + 0.35 * float(self._state[0]))
        cy = size * (0.55 + 0.25 * float(self._state[1]))
        body = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (size * 0.08) ** 2
        leg = (np.abs(xx - cx) < size * 0.025) & (yy > cy) & (yy < cy + size * 0.25)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[..., :] = np.array([80, 80, 80], dtype=np.uint8)
        img[body | leg] = np.array([230, 220, 50], dtype=np.uint8)
        return img

    def foreground_mask(self, clean: np.ndarray) -> np.ndarray:
        mask = (clean[..., 0] > 160).astype(np.float32)
        return mask[..., None]

    def foreground_mask_with_status(self, clean: np.ndarray) -> ForegroundMaskResult:
        mask = self.foreground_mask(clean)
        return ForegroundMaskResult(mask=mask, valid=True, area=float(mask.mean()))

    def close(self) -> None:
        pass

    def _info(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "camera_id": 0,
            "physics_state": self._state.copy(),
            "physics_state_hash": f"synthetic:{self.env_id}:{self.episode_id}:{self.step_id}:{self._state.tolist()}",
        }


class DMCWalkerEnv:
    """Lazy dm_control Walker backend.

    Importing this module does not require dm_control. Constructing this class
    does.
    """

    def __init__(self, config, *, env_id: int = 0):
        self.config = config
        self.env_id = int(env_id)
        try:
            from dm_control import suite
        except Exception as exc:
            raise RuntimeError("DMC backend is unavailable; install dm_control on the server.") from exc
        domain = str(config.env.dmc.domain)
        task_name = str(config.env.dmc.task_name)
        seed = int(config.training.seed) + self.env_id * 100003
        self.env = suite.load(domain, task_name, task_kwargs={"random": seed})
        action_spec = self.env.action_spec()
        self.action_space = ActionSpace(
            shape=tuple(action_spec.shape),
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
        )
        self.image_size = int(config.env.image_size)
        self.camera_id = int(config.env.camera_id)
        self.action_repeat = int(config.env.action_repeat)
        self.segmentation_failure_policy = str(config.env.dmc.segmentation_failure_policy)
        self.episode_id = 0
        self.step_id = 0
        self._time_step = None

    def reset(self) -> EnvStep:
        self.episode_id += 1
        self.step_id = 0
        self._time_step = self.env.reset()
        return EnvStep(self.render(), 0.0, True, False, False, 1.0, self._info())

    def step(self, action) -> EnvStep:
        action = np.asarray(action, dtype=np.float32)
        reward = 0.0
        discount = 1.0
        is_last = False
        is_terminal = False
        for _ in range(max(1, self.action_repeat)):
            self._time_step = self.env.step(action)
            reward += float(self._time_step.reward or 0.0)
            discount = float(self._time_step.discount if self._time_step.discount is not None else 1.0)
            is_last = bool(self._time_step.last())
            is_terminal = bool(is_last and discount == 0.0)
            if is_last:
                break
        self.step_id += 1
        return EnvStep(self.render(), reward, False, is_last, is_terminal, 0.0 if is_terminal else 1.0, self._info(discount=discount))

    def render(self) -> np.ndarray:
        return np.asarray(
            self.env.physics.render(self.image_size, self.image_size, camera_id=self.camera_id),
            dtype=np.uint8,
        )

    def foreground_mask(self, clean: np.ndarray) -> np.ndarray:
        return self.foreground_mask_with_status(clean).mask

    def foreground_mask_with_status(self, clean: np.ndarray) -> ForegroundMaskResult:
        try:
            seg = self.env.physics.render(self.image_size, self.image_size, camera_id=self.camera_id, segmentation=True)
            arr = np.asarray(seg)
            if arr.ndim not in {2, 3}:
                return self._segmentation_failure(clean, f"unexpected segmentation rank {arr.ndim}", area=None)
            channel = arr[..., 0] if arr.ndim == 3 else arr
            if tuple(channel.shape) != tuple(clean.shape[:2]):
                return self._segmentation_failure(
                    clean,
                    f"segmentation shape {tuple(channel.shape)} != RGB shape {tuple(clean.shape[:2])}",
                    area=None,
                )
            mask = channel > 0
            area = float(mask.mean())
            if not 0.0005 <= area <= 0.65:
                return self._segmentation_failure(clean, "mask area outside [0.0005,0.65]", area=area)
            return ForegroundMaskResult(mask=mask.astype(np.float32)[..., None], valid=True, area=area)
        except ForegroundSegmentationError:
            raise
        except Exception as exc:
            return self._segmentation_failure(clean, f"segmentation render failed: {type(exc).__name__}: {exc}", area=None)

    def _segmentation_failure(
        self,
        clean: np.ndarray,
        reason: str,
        *,
        area: float | None,
    ) -> ForegroundMaskResult:
        message = (
            "DMC foreground segmentation failed: "
            f"env_id={self.env_id}, episode_id={self.episode_id}, frame_id={self.step_id}, "
            f"mask_area={area}, reason={reason}"
        )
        if self.segmentation_failure_policy == "error":
            raise ForegroundSegmentationError(message)
        if self.segmentation_failure_policy != "invalidate_pair":
            raise ValueError(f"unsupported segmentation_failure_policy={self.segmentation_failure_policy!r}")
        return ForegroundMaskResult(
            mask=np.zeros((*clean.shape[:2], 1), dtype=np.float32),
            valid=False,
            area=area,
            failure_reason=message,
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _info(self, *, discount: float = 1.0) -> dict[str, Any]:
        try:
            state = np.asarray(self.env.physics.get_state(), dtype=np.float64)
            state_hash = str(hash(state.tobytes()))
        except Exception:
            state_hash = "unavailable"
        return {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "camera_id": self.camera_id,
            "discount": float(discount),
            "physics_state_hash": state_hash,
        }
