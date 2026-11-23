from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

from dreamer_v5.config import load_config
from dreamer_v5.dreamer.behavior import DreamerBehavior
from dreamer_v5.envs.factory import create_env, create_paired_view_generator
from dreamer_v5.model import V4WorldModel
from dreamer_v5.training.checkpoint import load_checkpoint, write_checkpoint_report


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    paired_backend = "none" if args.domain == "clean" else "davis"
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_eval_root(checkpoint) / args.domain
    )
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, _build_overrides(args))
    allowed_video_ids = _domain_video_ids(config, args.domain)
    device = torch.device(args.device)
    model = V4WorldModel(config).to(device)
    behavior = DreamerBehavior(
        config,
        belief_dim=int(model.belief_dim),
        action_dim=int(model.action_dim),
        device=device,
    ).to(device)
    report = load_checkpoint(
        args.checkpoint,
        config=config,
        model=model,
        behavior=behavior,
        restore_training_state=False,
    )
    write_checkpoint_report(output / "checkpoint_load_report.json", report)
    summary = evaluate_checkpoint(
        config=config,
        model=model,
        behavior=behavior,
        checkpoint_report=report.to_dict(),
        output_dir=output,
        env_backend=str(args.env_backend),
        episodes=int(args.episodes),
        episode_steps=int(args.episode_steps),
        score_window_steps=int(args.score_window_steps),
        deterministic=bool(args.deterministic),
        paired_backend=paired_backend,
        video_fps=int(args.video_fps),
        save_frame_every=int(args.save_frame_every),
        no_video=bool(args.no_video),
        allowed_video_ids=allowed_video_ids,
    )
    summary["domain"] = args.domain
    (output / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    _append_eval_csv(_default_eval_root(checkpoint) / "eval.csv", summary, checkpoint)
    print(json.dumps(_to_jsonable(summary), ensure_ascii=False), flush=True)
    return 0


def evaluate_checkpoint(
    *,
    config,
    model: V4WorldModel,
    behavior: DreamerBehavior,
    checkpoint_report: dict[str, Any],
    output_dir: Path,
    env_backend: str,
    episodes: int,
    episode_steps: int,
    score_window_steps: int,
    deterministic: bool,
    paired_backend: str,
    video_fps: int,
    save_frame_every: int,
    no_video: bool,
    allowed_video_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if episode_steps < 1:
        raise ValueError("episode_steps must be at least 1")
    if score_window_steps < 1:
        raise ValueError("score_window_steps must be at least 1")
    model.eval()
    behavior.eval()
    step_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    video_paths: list[str] = []
    frame_dirs: list[str] = []
    with torch.no_grad():
        for episode in range(episodes):
            episode_result = _run_episode(
                config=config,
                model=model,
                behavior=behavior,
                output_dir=output_dir,
                env_backend=env_backend,
                episode=episode,
                episode_steps=episode_steps,
                score_window_steps=score_window_steps,
                deterministic=deterministic,
                paired_backend=paired_backend,
                video_fps=video_fps,
                save_frame_every=save_frame_every,
                no_video=no_video,
                allowed_video_ids=allowed_video_ids,
            )
            step_rows.extend(episode_result["steps"])
            window_rows.extend(episode_result["windows"])
            episode_rows.append(episode_result["episode"])
            if episode_result.get("video"):
                video_paths.append(str(episode_result["video"]))
            if episode_result.get("frames"):
                frame_dirs.append(str(episode_result["frames"]))
    _write_jsonl(output_dir / "score_timeseries.jsonl", step_rows)
    _write_csv(output_dir / "score_timeseries.csv", step_rows)
    _write_jsonl(output_dir / "score_windows.jsonl", window_rows)
    _write_csv(output_dir / "score_windows.csv", window_rows)
    _write_jsonl(output_dir / "episode_scores.jsonl", episode_rows)
    _write_csv(output_dir / "episode_scores.csv", episode_rows)
    curve_path = output_dir / "score_curve.png"
    _write_score_curve(curve_path, step_rows)
    returns = [float(row["score"]) for row in episode_rows]
    window_means = _window_score_means(window_rows)
    summary = {
        "source": "checkpoint_real_environment_evaluation",
        "score_name": _score_name(env_backend, paired_backend),
        "env_backend": env_backend,
        "paired_backend": paired_backend,
        "policy_observation_source": _policy_observation_source(paired_backend),
        "deterministic": bool(deterministic),
        "allowed_video_ids": None
        if allowed_video_ids is None
        else list(allowed_video_ids),
        "checkpoint_report": checkpoint_report,
        "episodes": int(episodes),
        "episode_steps_limit": int(episode_steps),
        "score_window_steps": int(score_window_steps),
        "score_mean": _mean(returns),
        "score_std": _std(returns),
        "score_min": min(returns),
        "score_max": max(returns),
        "episode_scores": returns,
        "episode_lengths": [int(row["steps"]) for row in episode_rows],
        "action_rms": _mean([float(row["action_rms"]) for row in step_rows]),
        "deterministic_saturation": _mean(
            [float(row["deterministic_saturation"]) for row in step_rows]
        ),
        "window_score_mean_by_index": window_means,
        "outputs": {
            "summary_json": str((output_dir / "summary.json").resolve()),
            "score_timeseries_jsonl": str((output_dir / "score_timeseries.jsonl").resolve()),
            "score_timeseries_csv": str((output_dir / "score_timeseries.csv").resolve()),
            "score_windows_jsonl": str((output_dir / "score_windows.jsonl").resolve()),
            "score_windows_csv": str((output_dir / "score_windows.csv").resolve()),
            "episode_scores_jsonl": str((output_dir / "episode_scores.jsonl").resolve()),
            "episode_scores_csv": str((output_dir / "episode_scores.csv").resolve()),
            "score_curve_png": str(curve_path.resolve()),
            "episode_videos": video_paths,
            "episode_frame_dirs": frame_dirs,
        },
    }
    return summary


def _run_episode(
    *,
    config,
    model: V4WorldModel,
    behavior: DreamerBehavior,
    output_dir: Path,
    env_backend: str,
    episode: int,
    episode_steps: int,
    score_window_steps: int,
    deterministic: bool,
    paired_backend: str,
    video_fps: int,
    save_frame_every: int,
    no_video: bool,
    allowed_video_ids: tuple[str, ...] | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    action_dim = int(config.model.action_dim)
    episode_dir = output_dir / "episodes" / f"episode_{episode:03d}"
    frames_dir = episode_dir / "frames"
    episode_dir.mkdir(parents=True, exist_ok=True)
    if save_frame_every > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)
    env = create_env(config, env_backend, env_id=episode)
    paired = (
        None
        if paired_backend == "none"
        else create_paired_view_generator(
            config,
            paired_backend,
            env_id=episode,
            allowed_video_ids=allowed_video_ids,
        )
    )
    frames: list[np.ndarray] = []
    steps: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    try:
        current = env.reset()
        current_view = _render_policy_view(env, paired, current, paired_backend=paired_backend)
        policy_state = None
        incoming_action = np.zeros((1, action_dim), dtype=np.float32)
        cumulative_score = 0.0
        window_score = 0.0
        window_start = 1
        _remember_frame(frames, current_view["raw_a"])
        _maybe_write_frame(frames_dir, 0, current_view["raw_a"], save_frame_every=save_frame_every)
        for step_index in range(1, episode_steps + 1):
            raw_a = torch.as_tensor(current_view["raw_a"][None], device=device)
            incoming_action_t = torch.as_tensor(incoming_action, device=device)
            is_first = torch.as_tensor([current.is_first], dtype=torch.bool, device=device)
            action_t, policy_state, policy_metrics = behavior.act(
                model,
                raw_a,
                incoming_action_t,
                is_first,
                policy_state,
                deterministic=deterministic,
            )
            action = action_t.detach().cpu().numpy().reshape(action_dim).astype(np.float32)
            next_step = env.step(action)
            reward = float(next_step.reward)
            cumulative_score += reward
            window_score += reward
            row = {
                "episode": int(episode),
                "step": int(step_index),
                "reward": reward,
                "cumulative_score": cumulative_score,
                "window_index": int((step_index - 1) // score_window_steps),
                "is_last": bool(next_step.is_last),
                "is_terminal": bool(next_step.is_terminal),
                "action_norm": float(np.linalg.norm(action)),
                "action_rms": float(np.sqrt(np.mean(np.square(action)))),
                "deterministic_saturation": float(np.mean(np.abs(action) > 0.99)),
                "behavior_action_entropy": float(policy_metrics["behavior/action_entropy"].detach().cpu()),
                "physics_state_hash": str(next_step.info.get("physics_state_hash", "")),
                "policy_observation_source": _policy_observation_source(paired_backend),
                "background_id_a": str(current_view.get("background_id_a", "")),
                "background_id_b1": str(current_view.get("background_id_b1", "")),
                "background_id_b2": str(current_view.get("background_id_b2", "")),
                "paired_view_valid": bool(current_view.get("paired_view_valid", True)),
                "foreground_mask_valid": bool(current_view.get("foreground_mask_valid", True)),
                "foreground_mask_area": current_view.get("foreground_mask_area"),
            }
            steps.append(row)
            next_view = _render_policy_view(env, paired, next_step, paired_backend=paired_backend)
            _remember_frame(frames, next_view["raw_a"])
            _maybe_write_frame(frames_dir, step_index, next_view["raw_a"], save_frame_every=save_frame_every)
            is_window_end = step_index % score_window_steps == 0 or bool(next_step.is_last) or step_index == episode_steps
            if is_window_end:
                steps_in_window = step_index - window_start + 1
                windows.append(
                    {
                        "episode": int(episode),
                        "window_index": int((step_index - 1) // score_window_steps),
                        "start_step": int(window_start),
                        "end_step": int(step_index),
                        "steps": int(steps_in_window),
                        "score": float(window_score),
                        "mean_reward": float(window_score / max(1, steps_in_window)),
                        "cumulative_score_end": float(cumulative_score),
                        "ended_episode": bool(next_step.is_last),
                    }
                )
                window_start = step_index + 1
                window_score = 0.0
            current = next_step
            current_view = next_view
            incoming_action[0] = action
            if next_step.is_last:
                break
        if steps:
            Image.fromarray(np.asarray(frames[-1], dtype=np.uint8)).save(episode_dir / "last_frame.png")
        video_path = None
        if not no_video:
            candidate = episode_dir / f"episode_{episode:03d}.mp4"
            if _write_video(candidate, frames, fps=video_fps):
                video_path = candidate.resolve()
        episode_row = {
            "episode": int(episode),
            "score": float(cumulative_score),
            "steps": int(len(steps)),
            "completed": bool(steps[-1]["is_last"]) if steps else False,
            "mean_reward": float(cumulative_score / max(1, len(steps))),
            "video": None if video_path is None else str(video_path),
            "frames": None if save_frame_every <= 0 else str(frames_dir.resolve()),
        }
        return {"steps": steps, "windows": windows, "episode": episode_row, "video": video_path, "frames": episode_row["frames"]}
    finally:
        env.close()


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    collect_steps = max(1, int(args.score_window_steps))
    overrides: dict[str, object] = {
        "training.device": args.device,
        "r2_rssm.device": args.device,
        "env.backend": args.env_backend,
        "env.env_num": 1,
        "env.davis.backend": "mock" if args.domain == "clean" else "davis",
        "external_vlm.backend": "mock",
        "pipeline.vlm.start_managed_server": False,
        "pipeline.shard.collect_steps_per_env": collect_steps,
        "pipeline.shard.transitions_per_shard": collect_steps,
    }
    if args.davis_root:
        overrides["env.davis.root"] = args.davis_root
        overrides["paths.davis_root"] = args.davis_root
    if args.camera_id is not None:
        overrides["env.camera_id"] = int(args.camera_id)
    if args.action_repeat is not None:
        overrides["env.action_repeat"] = int(args.action_repeat)
    return overrides


def _render_policy_view(env, paired, step, *, paired_backend: str) -> dict[str, Any]:
    clean = np.asarray(step.observation, dtype=np.uint8)
    if paired is None:
        return {
            "clean": clean,
            "raw_a": clean,
            "raw_b1": None,
            "raw_b2": None,
            "foreground_mask": None,
            "background_id_a": "clean",
            "background_id_b1": "",
            "background_id_b2": "",
            "paired_view_valid": True,
            "foreground_mask_valid": True,
            "foreground_mask_area": None,
        }
    mask_result = env.foreground_mask_with_status(clean)
    pair = paired.render(clean, mask_result.mask, info=step.info)
    return {
        "clean": pair.clean,
        "raw_a": pair.raw_a,
        "raw_b1": pair.raw_b1,
        "raw_b2": pair.raw_b2,
        "foreground_mask": pair.foreground_mask,
        "background_id_a": pair.background_id_a,
        "background_id_b1": pair.background_id_b1,
        "background_id_b2": pair.background_id_b2,
        "paired_view_valid": bool(pair.paired_view_valid and mask_result.valid),
        "foreground_mask_valid": bool(mask_result.valid),
        "foreground_mask_area": mask_result.area,
        "foreground_mask_failure_reason": mask_result.failure_reason,
        "paired_backend": paired_backend,
    }


def _score_name(env_backend: str, paired_backend: str) -> str:
    if env_backend == "dmc" and paired_backend == "davis":
        return "true_dmc_environment_return_on_davis_composed_observations"
    if env_backend == "dmc":
        return "true_dmc_environment_return"
    return "true_environment_return"


def _policy_observation_source(paired_backend: str) -> str:
    if paired_backend == "davis":
        return "dmc_foreground_on_davis_background_raw_a"
    if paired_backend == "mock":
        return "dmc_foreground_on_mock_background_raw_a"
    return "clean_environment_rgb"


def _remember_frame(frames: list[np.ndarray], frame: np.ndarray) -> None:
    frames.append(np.asarray(frame, dtype=np.uint8).copy())


def _maybe_write_frame(frames_dir: Path, step: int, frame: np.ndarray, *, save_frame_every: int) -> None:
    if save_frame_every <= 0:
        return
    if step % int(save_frame_every) != 0:
        return
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(frames_dir / f"frame_{step:06d}.png")


def _write_video(path: Path, frames: list[np.ndarray], *, fps: int) -> bool:
    if not frames:
        return False
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=int(fps), codec="libx264", quality=8)
        return True
    except Exception:
        pass
    try:
        import cv2

        first = np.asarray(frames[0], dtype=np.uint8)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except Exception:
        return False


def _write_score_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 960, 420
    pad_left, pad_top, pad_right, pad_bottom = 70, 28, 28, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    draw.rectangle([pad_left, pad_top, pad_left + plot_w, pad_top + plot_h], outline=(40, 40, 40))
    if not rows:
        canvas.save(path)
        return
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    max_step = max(int(row["step"]) for row in rows)
    scores = [float(row["cumulative_score"]) for row in rows]
    min_score = min(0.0, min(scores))
    max_score = max(scores)
    if abs(max_score - min_score) < 1.0e-9:
        max_score = min_score + 1.0
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]
    for idx, (episode, episode_rows) in enumerate(sorted(by_episode.items())):
        points = []
        for row in episode_rows:
            x = pad_left + int((float(row["step"]) / max(1, max_step)) * plot_w)
            y = pad_top + plot_h - int(((float(row["cumulative_score"]) - min_score) / (max_score - min_score)) * plot_h)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=palette[idx % len(palette)], width=2)
        elif points:
            x, y = points[0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=palette[idx % len(palette)])
    draw.text((pad_left, height - 35), "step", fill=(0, 0, 0))
    draw.text((10, pad_top), "cumulative score", fill=(0, 0, 0))
    draw.text((pad_left, 6), f"score range [{min_score:.2f}, {max_score:.2f}]", fill=(0, 0, 0))
    canvas.save(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_to_jsonable(row) for row in rows)


def _window_score_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["window_index"]), []).append(float(row["score"]))
    return {str(index): _mean(values) for index, values in sorted(grouped.items())}


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _to_jsonable(value.detach().cpu().reshape(()).item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _domain_video_ids(config, domain: str) -> tuple[str, ...] | None:
    if domain == "clean":
        return None
    split_path = Path(str(config.evaluation.split_file))
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    key = "seen_eval_video_ids" if domain == "seen_davis" else "unseen_eval_video_ids"
    values = tuple(str(value) for value in payload[key])
    if not values:
        raise ValueError(f"DAVIS split {key!r} is empty in {split_path}")
    return values


def _default_eval_root(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / "eval"
    return checkpoint.parent / "eval"


def _append_eval_csv(path: Path, summary: dict[str, Any], checkpoint: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "domain": summary["domain"],
        "deterministic": summary["deterministic"],
        "episodes": summary["episodes"],
        "mean_return": summary["score_mean"],
        "std_return": summary["score_std"],
        "mean_episode_length": _mean([float(v) for v in summary["episode_lengths"]]),
        "action_rms": summary["action_rms"],
        "deterministic_saturation": summary["deterministic_saturation"],
    }
    fields = list(row)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    try:
        from torch.utils.tensorboard import SummaryWriter

        experiment_root = path.parent.parent
        writer = SummaryWriter(str(experiment_root / "logs" / "tensorboard"))
        tag = {
            "clean": "eval/clean_return",
            "seen_davis": "eval/seen_return",
            "unseen_davis": "eval/unseen_return",
        }[str(summary["domain"])]
        writer.add_scalar(tag, float(summary["score_mean"]), int(summary["checkpoint_report"]["step"]))
        writer.close()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an independent V5 checkpoint.")
    parser.add_argument("--config", default="configs/walker_walk.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--domain", choices=["clean", "seen_davis", "unseen_davis"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-backend", choices=["dmc", "synthetic"], default="dmc")
    parser.add_argument("--davis-root")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--score-window-steps", type=int, default=50)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--save-frame-every", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--deterministic", dest="deterministic", action="store_true")
    policy.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--action-repeat", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
