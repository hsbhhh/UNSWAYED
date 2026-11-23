from __future__ import annotations

from .dmc_env import DMCWalkerEnv, SyntheticWalkerEnv
from .paired_view import DavisPairedViewGenerator, MockPairedViewGenerator


def create_env(config, backend: str | None = None, *, env_id: int = 0):
    backend = str(backend or config.env.backend)
    if backend == "synthetic":
        return SyntheticWalkerEnv(config, env_id=env_id)
    if backend == "dmc":
        return DMCWalkerEnv(config, env_id=env_id)
    raise ValueError(f"unsupported env backend={backend!r}")


def create_paired_view_generator(
    config,
    backend: str | None = None,
    *,
    env_id: int = 0,
    allowed_video_ids=None,
):
    backend = str(backend or config.env.davis.backend)
    if backend in {"mock", "synthetic"}:
        if allowed_video_ids is not None:
            raise ValueError("allowed_video_ids is supported only by the DAVIS paired-view backend")
        return MockPairedViewGenerator(config, env_id=env_id)
    if backend == "davis":
        return DavisPairedViewGenerator(
            config,
            env_id=env_id,
            allowed_video_ids=allowed_video_ids,
        )
    raise ValueError(f"unsupported paired-view backend={backend!r}")
