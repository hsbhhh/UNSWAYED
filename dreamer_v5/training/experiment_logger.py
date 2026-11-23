from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import torch

from .metrics import flatten_metrics


TB_TAGS = {
    "loss/total": "world/total",
    "dreamer/dyn_kl_clipped": "world/dyn",
    "dreamer/rep_kl_clipped": "world/rep",
    "dreamer/reward_loss": "world/reward",
    "dreamer/continue_loss": "world/continue",
    "vgc/scaled": "world/vgc",
    "msir/loss": "world/msir",
    "capc/weighted": "world/capc",
    "behavior/actor_loss": "actor/loss",
    "behavior/transformed_std_mean": "actor/std_mean",
    "behavior/policy_entropy": "actor/entropy",
    "behavior/action_saturation_gt_099": "actor/stochastic_saturation",
    "behavior/deterministic_saturation_gt_099": "actor/deterministic_saturation",
    "behavior/executed_action_rms": "actor/action_rms",
    "behavior/value_loss": "critic/loss",
    "system/fps": "system/fps",
    "system/gpu_memory": "system/gpu_memory",
}


class ExperimentLogger:
    """Minimal production logger: JSONL, CSV, TensorBoard and warnings."""

    def __init__(self, root: str | Path, config):
        self.root = Path(root)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.episodes_path = self.logs_dir / "episodes.csv"
        self.started = time.perf_counter()
        self.last_time = self.started
        self.last_step = 0
        self.config = config
        self.tb = None
        if bool(config.get("logging", {}).get("tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.logs_dir / "tensorboard"))
            except Exception as exc:  # TensorBoard is optional, flat logs are not.
                warnings.warn(f"TensorBoard disabled: {exc}", RuntimeWarning)

    def write_metrics(self, row: dict[str, Any]) -> dict[str, Any]:
        flat = flatten_metrics(row)
        step = int(flat.get("global_env_steps", flat.get("step", 0)) or 0)
        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1.0e-9)
        fps = max(step - self.last_step, 0) / elapsed
        flat["system/fps"] = float(fps)
        flat["system/gpu_memory"] = (
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        )
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(flat, ensure_ascii=False) + "\n")
        if self.tb is not None:
            for source, tag in TB_TAGS.items():
                value = flat.get(source)
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(tag, value, step)
            self.tb.flush()
        self._actor_health(flat)
        self.last_time = now
        self.last_step = step
        return flat

    def write_episodes(self, rows: list[dict[str, Any]], *, global_env_steps: int) -> None:
        if not rows:
            return
        fields = ["global_env_steps", "episode", "env_id", "episode_return", "episode_length"]
        needs_header = not self.episodes_path.exists() or self.episodes_path.stat().st_size == 0
        with self.episodes_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            for row in rows:
                payload = {key: row.get(key) for key in fields}
                payload["global_env_steps"] = int(global_env_steps)
                writer.writerow(payload)
                if self.tb is not None:
                    episode_step = int(row.get("episode", 0))
                    self.tb.add_scalar("train/return", float(row["episode_return"]), episode_step)
                    self.tb.add_scalar("train/episode_length", int(row["episode_length"]), episode_step)
        if self.tb is not None:
            self.tb.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()

    def _actor_health(self, row: dict[str, Any]) -> None:
        health = self.config.get("logging", {}).get("actor_health", {})
        sat_limit = float(health.get("saturation_warn", 0.95))
        std_limit = float(health.get("std_warn", 0.95))
        saturation = row.get("behavior/deterministic_saturation_gt_099")
        std_mean = row.get("behavior/transformed_std_mean")
        reasons = []
        if isinstance(saturation, (int, float)) and saturation >= sat_limit:
            reasons.append(f"deterministic_saturation={saturation:.4f}")
        if isinstance(std_mean, (int, float)) and std_mean >= std_limit:
            reasons.append(f"std_mean={std_mean:.4f}")
        if not reasons:
            return
        message = "Actor health warning: " + ", ".join(reasons)
        if bool(self.config.get("logging", {}).get("behavior_safety_stop", False)):
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)
