from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..config import to_plain
from ..dreamer.behavior import DreamerBehavior
from ..dreamer.optim import assert_parameter_isolation, build_behavior_optimizer, build_warmup_scheduler, build_world_optimizer
from ..envs.factory import create_env
from ..model import WorldModel
from ..pipeline.collector import OnlineCollector
from ..pipeline.replay import ReplayBuffer
from ..pipeline.schema import ReplaySchema
from ..pipeline.shard import ShardStore
from .behavior_step import behavior_train_step
from .checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint, write_checkpoint_report
from .losses import compute_world_model_losses
from .experiment_logger import ExperimentLogger
from .metrics import flatten_metrics
from .world_model_step import world_model_train_step


@dataclass
class TrainState:
    global_env_steps: int = 0
    gradient_updates: int = 0
    shards_collected: int = 0
    replay_records: int = 0
    policy_version: int = 0


class ManagedVLMServer:
    def __init__(self, config, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "vlm_server.log"
        self.ready_path = self.output_dir / "vlm_server_ready.json"
        # The resident server removes ready_path during shutdown because it is
        # an ephemeral liveness marker. Preserve the structured startup proof
        # separately so post-run diagnostics do not depend on process order.
        self.ready_snapshot_path = self.output_dir / "vlm_server_ready_snapshot.json"
        self.stop_path = self.output_dir / "vlm_server.stop"
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def start(self) -> None:
        if str(self.config.external_vlm.backend) != "resident":
            return
        if not bool(self.config.pipeline.vlm.start_managed_server):
            return
        checkpoint = self.config.paths.vlm_checkpoint or os.environ.get("VLM_FINETUNE_V2_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("Resident VLM managed server requires paths.vlm_checkpoint or VLM_FINETUNE_V2_CHECKPOINT.")
        configured_python = self.config.external_vlm.get("python_executable", None)
        python_executable = str(configured_python or os.environ.get("VLM_PYTHON_EXECUTABLE") or sys.executable)
        python_path = Path(python_executable).expanduser().resolve()
        if not python_path.is_file():
            raise FileNotFoundError(f"VLM Python executable does not exist: {python_path}")
        args = [
            str(python_path),
            "-m",
            "unswayed.vlm.server",
            "--shared-memory-name",
            str(self.config.external_vlm.shared_memory_name),
            "--env-num",
            str(int(self.config.env.env_num)),
            "--clip-len",
            str(int(self.config.external_vlm.get("clip_len", 8))),
            "--input-height",
            str(int(self.config.external_vlm.input_height)),
            "--input-width",
            str(int(self.config.external_vlm.input_width)),
            "--grid-size",
            str(int(self.config.external_vlm.grid_size)),
            "--feature-dim",
            str(int(self.config.external_vlm.feature_dim)),
            "--view-num",
            str(int(self.config.external_vlm.get("view_num", 2))),
            "--micro-batch-size",
            str(int(self.config.external_vlm.micro_batch_size)),
            "--max-batch-wait-ms",
            str(float(self.config.external_vlm.max_batch_wait_ms)),
            "--device",
            str(self.config.training.device),
            "--dtype",
            str(self.config.external_vlm.dtype),
            "--internvideo2-backend",
            str(self.config.external_vlm.get("internvideo2_backend", "naive")),
            "--task-name",
            str(self.config.external_vlm.task_name),
            "--checkpoint",
            str(checkpoint),
            "--heartbeat-path",
            str(self.ready_path),
            "--logdir",
            str(self.output_dir),
            "--q-sem-only",
            "--disable-reference-features",
        ]
        if bool(self.config.external_vlm.strict_per_frame):
            args.append("--strict-per-frame")
        base = self.config.paths.internvideo2_base_checkpoint or os.environ.get("INTERNVIDEO2_BASE_CHECKPOINT")
        if base:
            args.extend(["--base-checkpoint", str(base)])
        fine_turn = self.config.paths.fine_turn_root or os.environ.get("FINE_TURN_ROOT")
        if fine_turn:
            args.extend(["--fine-turn-root", str(fine_turn)])
        self.ready_path.unlink(missing_ok=True)
        self.ready_snapshot_path.unlink(missing_ok=True)
        self.stop_path.unlink(missing_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        env = _vlm_subprocess_env(os.environ, python_path)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self.process = subprocess.Popen(args, cwd=str(Path(__file__).resolve().parents[2]), stdout=self._log_handle, stderr=subprocess.STDOUT, env=env)
        self._wait_ready()

    def _wait_ready(self) -> None:
        timeout = float(
            self.config.external_vlm.get(
                "startup_timeout_sec",
                self.config.external_vlm.strict_timeout_sec,
            )
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"managed VLM server exited early with code {self.process.returncode}; see {self.log_path}")
            if self.ready_path.exists():
                ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
                expected_backend = str(self.config.external_vlm.get("internvideo2_backend", "naive"))
                actual_backend = str(ready.get("internvideo2_backend_actual", "naive"))
                if actual_backend != expected_backend:
                    raise RuntimeError(
                        "Managed VLM server reported an unexpected InternVideo2 backend: "
                        f"expected={expected_backend}, actual={actual_backend}"
                    )
                temporary = self.ready_snapshot_path.with_name(
                    f".{self.ready_snapshot_path.name}.{os.getpid()}.tmp"
                )
                temporary.unlink(missing_ok=True)
                temporary.write_text(
                    json.dumps(ready, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(self.ready_snapshot_path)
                return
            time.sleep(0.25)
        raise TimeoutError(f"managed VLM server did not become ready within {timeout:.1f}s; see {self.log_path}")

    def close(self) -> None:
        if self.process is None:
            return
        self.stop_path.write_text("stop\n", encoding="utf-8")
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _vlm_subprocess_env(parent_env: dict[str, str], python_executable: Path) -> dict[str, str]:
    env = dict(parent_env)
    prefix = python_executable.parent.parent
    runtime_paths: list[Path] = [prefix / "lib"]
    for site_packages in sorted((prefix / "lib").glob("python*/site-packages")):
        runtime_paths.extend(
            [
                site_packages / "torch" / "lib",
                *sorted((site_packages / "nvidia").glob("cu*/lib")),
                site_packages / "nvidia" / "cuda_runtime" / "lib",
            ]
        )
    existing = [str(path.resolve()) for path in runtime_paths if path.is_dir()]
    inherited = str(env.get("LD_LIBRARY_PATH", "")).strip()
    if inherited:
        existing.append(inherited)
    if existing:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(existing)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class ProductionTrainer:
    def __init__(
        self,
        config,
        output_dir: str | Path,
        *,
        env_backend: str | None = None,
        paired_backend: str | None = None,
        vlm_backend: str | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.eval_dir = self.output_dir / "eval"
        self.debug_dir = self.output_dir / "debug"
        self.videos_dir = self.output_dir / "videos"
        for path in (
            self.checkpoints_dir,
            self.logs_dir,
            self.eval_dir,
            self.debug_dir,
            self.videos_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or config.training.device)
        self.config.training.device = str(self.device)
        self.config.r2_rssm.device = str(self.device)
        if not self.config.external_vlm.shared_memory_name:
            self.config.external_vlm.shared_memory_name = f"v5_{self.output_dir.name}_{os.getpid()}"
        self.env_backend = env_backend or str(self.config.env.backend)
        self.paired_backend = paired_backend or ("mock" if self.env_backend == "synthetic" else str(self.config.env.davis.backend))
        self.vlm_backend = vlm_backend or str(self.config.external_vlm.backend)
        self.config.env.backend = self.env_backend
        self.config.external_vlm.backend = self.vlm_backend
        if self.paired_backend == "mock":
            self.config.env.davis.backend = "mock"
        self.model = WorldModel(config).to(self.device)
        self.behavior = DreamerBehavior(config, belief_dim=int(self.model.belief_dim), action_dim=int(self.model.action_dim), device=self.device).to(self.device)
        assert_parameter_isolation(self.model.world_parameters(), self.behavior.trainable_parameters())
        self.world_optimizer = build_world_optimizer(self.model, config)
        self.behavior_optimizer = build_behavior_optimizer(self.behavior, config)
        self.world_scheduler = build_warmup_scheduler(self.world_optimizer, config)
        self.behavior_scheduler = build_warmup_scheduler(self.behavior_optimizer, config)
        self.replay_schema = ReplaySchema.from_config(config)
        self.replay = ReplayBuffer(
            capacity=int(config.replay.capacity),
            sequence_length=int(config.training.sequence_length),
            seed=int(config.training.seed),
            expected_schema=self.replay_schema,
        )
        shard_cfg = config.pipeline.shard
        self.shards = ShardStore(
            self.checkpoints_dir if bool(config.pipeline.checkpoint.persist_shards) else None,
            max_collected_shards=int(shard_cfg.max_collected_shards),
            max_ready_shards=int(shard_cfg.max_ready_shards),
            retain_ready_shards=int(shard_cfg.retain_ready_shards),
            schema=self.replay_schema,
        )
        self.logger = ExperimentLogger(self.output_dir, config)
        self.state = TrainState()
        self.stop_requested = False
        self.vlm_server = ManagedVLMServer(config, self.logs_dir)
        self.collector: OnlineCollector | None = None

    def request_stop(self) -> None:
        """Request a checkpointed stop at the next complete training-slice boundary."""

        self.stop_requested = True

    def train(
        self,
        *,
        total_env_steps: int | None = None,
        resume: str | Path | None = None,
        resume_world_lr: float | None = None,
    ) -> TrainState:
        if resume:
            report = load_checkpoint(
                resume,
                config=self.config,
                model=self.model,
                behavior=self.behavior,
                world_optimizer=self.world_optimizer,
                behavior_optimizer=self.behavior_optimizer,
                world_scheduler=self.world_scheduler,
                behavior_scheduler=self.behavior_scheduler,
                trainer_state=self.state,
                replay=self.replay,
                shard_store=self.shards,
                restore_training_state=True,
            )
            write_checkpoint_report(self.logs_dir / "checkpoint_load_report.json", report)
            if resume_world_lr is not None:
                self._apply_resume_world_learning_rate(resume_world_lr, checkpoint=resume)
        self._write_resolved_config()
        target = int(total_env_steps or self.config.training.steps)
        next_log = self.state.global_env_steps
        next_ckpt = self.state.global_env_steps
        try:
            self.vlm_server.start()
            self.collector = OnlineCollector(
                self.config,
                self.model,
                self.behavior,
                env_backend=self.env_backend,
                paired_backend=self.paired_backend,
                vlm_backend=self.vlm_backend,
                device=self.device,
            )
            while self.state.global_env_steps < target and not self.stop_requested:
                self.collector.policy_version = int(self.state.policy_version)
                shard, collect_report = self.collector.collect_shard(self.shards)
                ready = self.shards.mark_ready(shard)
                self.shards.save_ready(ready)
                self.replay.append_shard(ready)
                self.state.shards_collected += 1
                self.state.global_env_steps += int(collect_report.transitions)
                self.state.replay_records = len(self.replay)
                self.logger.write_episodes(
                    collect_report.completed_episodes,
                    global_env_steps=self.state.global_env_steps,
                )
                updates = self._updates_for_records(int(collect_report.transitions))
                latest_metrics: dict[str, Any] = dict(collect_report.metrics)
                latest_metrics.update(
                    {f"pipeline/shard_{key}": float(value) for key, value in self.shards.retention_snapshot().items()}
                )
                for _ in range(updates):
                    latest_metrics.update(self.train_one_update())
                if self.state.global_env_steps >= next_log or self.state.global_env_steps >= target:
                    latest_metrics.update(self._evaluate_current_policy_score())
                    self._write_metrics("train_slice", latest_metrics, extra={"updates_this_slice": updates})
                    next_log = self.state.global_env_steps + int(self.config.training.log_every)
                if self.state.global_env_steps >= next_ckpt or self.state.global_env_steps >= target:
                    self.save_latest()
                    next_ckpt = self.state.global_env_steps + int(self.config.training.checkpoint_every)
            if self.stop_requested:
                self.save_latest()
                stopped_metrics: dict[str, Any] = {"trainer/stopped_by_request": 1.0}
                stopped_metrics.update(self._evaluate_current_policy_score())
                self._write_metrics("train_stopped", stopped_metrics)
                return self.state
            complete_metrics: dict[str, Any] = {"trainer/complete": 1.0}
            complete_metrics.update(self._evaluate_current_policy_score())
            self._write_metrics("train_complete", complete_metrics)
            return self.state
        finally:
            if self.collector is not None:
                self.collector.close()
            self.vlm_server.close()
            self.logger.close()

    def _apply_resume_world_learning_rate(self, learning_rate: float, *, checkpoint: str | Path) -> None:
        requested_lr = float(learning_rate)
        if not np.isfinite(requested_lr) or requested_lr <= 0.0:
            raise ValueError(f"resume world learning rate must be finite and positive, got {learning_rate!r}")

        optimizer_lrs_before = [float(group["lr"]) for group in self.world_optimizer.param_groups]
        scheduler_base_lrs_before = [float(value) for value in self.world_scheduler.base_lrs]
        scheduler_last_lrs_before = [float(value) for value in self.world_scheduler.get_last_lr()]
        behavior_lrs_before = [float(group["lr"]) for group in self.behavior_optimizer.param_groups]

        self.config.training.lr = requested_lr
        for group in self.world_optimizer.param_groups:
            group["initial_lr"] = requested_lr

        scheduler = self.world_scheduler
        scheduler.base_lrs = [requested_lr for _ in scheduler.base_lrs]
        factors = [float(fn(scheduler.last_epoch)) for fn in scheduler.lr_lambdas]
        effective_lrs = [requested_lr * factor for factor in factors]
        for group, effective_lr in zip(self.world_optimizer.param_groups, effective_lrs, strict=True):
            group["lr"] = effective_lr
        scheduler._last_lr = list(effective_lrs)

        report = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "world_lr_override_requested": requested_lr,
            "world_optimizer_lrs_before": optimizer_lrs_before,
            "world_optimizer_lrs_after": [float(group["lr"]) for group in self.world_optimizer.param_groups],
            "world_scheduler_base_lrs_before": scheduler_base_lrs_before,
            "world_scheduler_base_lrs_after": [float(value) for value in scheduler.base_lrs],
            "world_scheduler_last_lrs_before": scheduler_last_lrs_before,
            "world_scheduler_last_lrs_after": [float(value) for value in scheduler.get_last_lr()],
            "world_scheduler_last_epoch": int(scheduler.last_epoch),
            "world_scheduler_factors": factors,
            "warmup_updates": int(self.config.optimizer.warmup_updates),
            "behavior_optimizer_lrs_before": behavior_lrs_before,
            "behavior_optimizer_lrs_after": [float(group["lr"]) for group in self.behavior_optimizer.param_groups],
        }
        (self.logs_dir / "resume_learning_rate_override.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"[UNSWAYED_RESUME_LR_OVERRIDE] world_lr={requested_lr:g}", flush=True)

    def debug_once(
        self,
        *,
        checkpoint: str | Path | None = None,
        collect_steps_per_env: int = 2,
    ) -> dict[str, Any]:
        if checkpoint:
            report = load_checkpoint(
                checkpoint,
                config=self.config,
                model=self.model,
                behavior=self.behavior,
                restore_training_state=False,
            )
            write_checkpoint_report(self.debug_dir / "checkpoint_load_report.json", report)
        self._write_resolved_config()
        previous_model_training = bool(self.model.training)
        previous_behavior_training = bool(self.behavior.training)
        previous_rng_state = capture_rng_state()
        debug_collector: OnlineCollector | None = None
        self.model.eval()
        self.behavior.eval()
        try:
            self.vlm_server.start()
            debug_collector = OnlineCollector(
                self.config,
                self.model,
                self.behavior,
                env_backend=self.env_backend,
                paired_backend=self.paired_backend,
                vlm_backend=self.vlm_backend,
                device=self.device,
            )
            debug_shards = ShardStore(
                max_collected_shards=1,
                max_ready_shards=1,
                retain_ready_shards=1,
                schema=self.replay_schema,
            )
            debug_replay = ReplayBuffer(
                capacity=max(int(self.config.replay.capacity), int(self.config.env.env_num) * int(collect_steps_per_env)),
                sequence_length=int(self.config.training.sequence_length),
                seed=int(self.config.training.seed),
                expected_schema=self.replay_schema,
            )
            shard, report = debug_collector.collect_shard(debug_shards, collect_steps_per_env=int(collect_steps_per_env))
            ready = debug_shards.mark_ready(shard)
            debug_replay.append_shard(ready)
            debug_sequence_length = max(1, min(int(self.config.training.sequence_length), int(collect_steps_per_env)))
            batch = debug_replay.sample(batch_size=1, sequence_length=debug_sequence_length)
            batch = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.no_grad():
                loss, metrics, output = compute_world_model_losses(self.model, batch)
                behavior_metrics = self.behavior.compute_losses(
                    self.model,
                    output.stoch.detach(),
                    output.deter.detach(),
                    batch,
                ).metrics
            summary = {
                "source": "online_rollout",
                "env_backend": self.env_backend,
                "paired_backend": self.paired_backend,
                "vlm_backend": self.vlm_backend,
                "shared_memory_name": str(self.config.external_vlm.shared_memory_name),
                "records": int(len(ready.records)),
                "world_loss": float(loss.detach().cpu()),
                "collector": report.metrics,
                "metrics": flatten_metrics(metrics | behavior_metrics),
                "checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
            }
            (self.debug_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            torch.save(
                {"records": ready.records, "batch": {k: v.cpu() for k, v in batch.items()}},
                self.debug_dir / "debug_records.pt",
            )
            return summary
        finally:
            if debug_collector is not None:
                debug_collector.close()
            self.vlm_server.close()
            self.model.train(previous_model_training)
            self.behavior.train(previous_behavior_training)
            restore_rng_state(previous_rng_state)

    def save_latest(self) -> None:
        save_checkpoint(
            self.checkpoints_dir / "latest.pt",
            config=self.config,
            model=self.model,
            behavior=self.behavior,
            world_optimizer=self.world_optimizer,
            behavior_optimizer=self.behavior_optimizer,
            world_scheduler=self.world_scheduler,
            behavior_scheduler=self.behavior_scheduler,
            trainer_state=self.state,
            replay=self.replay,
            shard_store=self.shards,
        )
        if bool(self.config.pipeline.checkpoint.save_numbered):
            save_checkpoint(
                self.checkpoints_dir / f"step_{self.state.global_env_steps:09d}.pt",
                config=self.config,
                model=self.model,
                behavior=self.behavior,
                world_optimizer=self.world_optimizer,
                behavior_optimizer=self.behavior_optimizer,
                world_scheduler=self.world_scheduler,
                behavior_scheduler=self.behavior_scheduler,
                trainer_state=self.state,
                replay=self.replay,
                shard_store=self.shards,
            )
            self._prune_periodic_checkpoints()

    def train_one_update(self) -> dict[str, Any]:
        batch = self._sample_batch()
        wm = world_model_train_step(
            self.model,
            batch,
            self.world_optimizer,
            self.config,
            update_index=self.state.gradient_updates,
        )
        beh = behavior_train_step(
            self.behavior,
            self.model,
            wm.output,
            batch,
            self.behavior_optimizer,
            self.config,
            update_index=self.state.gradient_updates,
        )
        self.world_scheduler.step()
        self.behavior_scheduler.step()
        self.state.gradient_updates += 1
        self.state.policy_version = int(self.state.gradient_updates)
        metrics: dict[str, Any] = {}
        metrics.update(wm.metrics)
        metrics.update(beh.metrics)
        metrics["optimizer/world_lr"] = float(self.world_optimizer.param_groups[0]["lr"])
        metrics["optimizer/behavior_lr"] = float(self.behavior_optimizer.param_groups[0]["lr"])
        metrics["replay/sample_identities"] = str(self.replay.last_sample_identities)
        return metrics

    def _evaluate_current_policy_score(self) -> dict[str, Any]:
        previous_model_training = bool(self.model.training)
        previous_behavior_training = bool(self.behavior.training)
        previous_rng_state = capture_rng_state()
        env_num = int(self.config.env.env_num)
        action_dim = int(self.config.model.action_dim)
        steps_per_env = max(1, min(int(self.config.pipeline.shard.collect_steps_per_env), 128))
        envs = []
        try:
            self.model.eval()
            self.behavior.eval()
            envs = [create_env(self.config, self.env_backend, env_id=env_id) for env_id in range(env_num)]
            current_steps = [env.reset() for env in envs]
            incoming_actions = np.zeros((env_num, action_dim), dtype=np.float32)
            policy_state = None
            horizon_scores = [0.0 for _ in range(env_num)]
            episode_scores = [0.0 for _ in range(env_num)]
            completed_episode_scores: list[float] = []
            with torch.no_grad():
                for _ in range(steps_per_env):
                    raw_a = torch.as_tensor(
                        np.stack([step.observation for step in current_steps], axis=0),
                        device=self.device,
                    )
                    incoming_action_t = torch.as_tensor(incoming_actions, device=self.device)
                    is_first = torch.as_tensor([step.is_first for step in current_steps], dtype=torch.bool, device=self.device)
                    action_t, policy_state, _ = self.behavior.act(
                        self.model,
                        raw_a,
                        incoming_action_t,
                        is_first,
                        policy_state,
                        deterministic=True,
                    )
                    actions = action_t.detach().cpu().numpy().astype(np.float32)
                    next_steps = []
                    next_incoming_actions = np.zeros_like(incoming_actions)
                    reset_policy_state = False
                    for env_id, (env, step, action) in enumerate(zip(envs, current_steps, actions, strict=True)):
                        if step.is_last:
                            next_steps.append(env.reset())
                            episode_scores[env_id] = 0.0
                            reset_policy_state = True
                            continue
                        next_step = env.step(action)
                        reward = float(next_step.reward)
                        horizon_scores[env_id] += reward
                        episode_scores[env_id] += reward
                        if next_step.is_last:
                            completed_episode_scores.append(float(episode_scores[env_id]))
                            episode_scores[env_id] = 0.0
                            next_steps.append(env.reset())
                            reset_policy_state = True
                        else:
                            next_steps.append(next_step)
                            next_incoming_actions[env_id] = action
                    current_steps = next_steps
                    incoming_actions = next_incoming_actions
                    if reset_policy_state:
                        policy_state = None
            score_t = torch.as_tensor(horizon_scores, dtype=torch.float32)
            mean_score = float(score_t.mean().item())
            metrics: dict[str, Any] = {
                "score": mean_score,
                "eval/score_mean": mean_score,
                "eval/score_min": float(score_t.min().item()),
                "eval/score_max": float(score_t.max().item()),
                "eval/score_std": float(score_t.std(unbiased=False).item()),
                "eval/score_steps_per_env": int(steps_per_env),
                "eval/score_env_num": int(env_num),
                "eval/score_completed_episodes": int(len(completed_episode_scores)),
                "eval/score_deterministic": True,
                "eval/score_weight_source": "current_model_behavior_weights",
                "eval/score_policy_version": int(self.state.policy_version),
            }
            if completed_episode_scores:
                completed_t = torch.as_tensor(completed_episode_scores, dtype=torch.float32)
                metrics["eval/score_completed_episode_mean"] = float(completed_t.mean().item())
            else:
                metrics["eval/score_completed_episode_mean"] = None
            return metrics
        finally:
            for env in envs:
                env.close()
            self.model.train(previous_model_training)
            self.behavior.train(previous_behavior_training)
            restore_rng_state(previous_rng_state)

    def _sample_batch(self, *, batch_size: int | None = None, sequence_length: int | None = None) -> dict[str, torch.Tensor]:
        batch = self.replay.sample(
            batch_size=int(batch_size or self.config.training.batch_size),
            sequence_length=int(sequence_length or self.config.training.sequence_length),
        )
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _updates_for_records(self, records: int) -> int:
        samples_per_update = int(self.config.training.batch_size) * int(self.config.training.sequence_length)
        return max(1, int(float(records) * float(self.config.training.train_ratio) / max(1, samples_per_update)))

    def _write_metrics(self, event: str, metrics: dict[str, Any], *, extra: dict[str, Any] | None = None) -> None:
        row = {
            "event": event,
            "step": int(self.state.global_env_steps),
            "global_env_steps": int(self.state.global_env_steps),
            "gradient_updates": int(self.state.gradient_updates),
            "replay_records": int(self.state.replay_records),
            "shards_collected": int(self.state.shards_collected),
            "shared_memory_name": str(self.config.external_vlm.shared_memory_name),
        }
        row.update(flatten_metrics(metrics))
        row.update(extra or {})
        row = self.logger.write_metrics(row)
        compact_keys = {
            "update": row.get("gradient_updates"),
            "env_step": row.get("global_env_steps"),
            "return": row.get("eval/score_mean"),
            "world": row.get("loss/total"),
            "actor": row.get("behavior/actor_loss"),
            "critic": row.get("behavior/value_loss"),
            "vgc": row.get("vgc/scaled"),
            "msir": row.get("msir/loss"),
            "capc": row.get("capc/weighted"),
            "std": row.get("behavior/transformed_std_mean"),
            "entropy": row.get("behavior/policy_entropy"),
            "sat": row.get("behavior/deterministic_saturation_gt_099"),
            "fps": row.get("system/fps"),
        }
        print("[UNSWAYED] " + " ".join(f"{key}={_compact(value)}" for key, value in compact_keys.items()), flush=True)

    def _write_resolved_config(self) -> None:
        (self.output_dir / "config.yaml").write_text(
            yaml.safe_dump(to_plain(self.config), sort_keys=False), encoding="utf-8"
        )

    def _prune_periodic_checkpoints(self) -> None:
        keep = int(self.config.get("checkpoint", {}).get("keep_last_n", 3))
        if keep < 1:
            return
        paths = sorted(self.checkpoints_dir.glob("step_*.pt"))
        for path in paths[:-keep]:
            path.unlink()


def _compact(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


# Kept as a source-level alias for small downstream scripts; it does not refer
# to or import any UNSWAYED source tree.
V4Trainer = ProductionTrainer
