from __future__ import annotations

import argparse
import random
import signal
from pathlib import Path

import numpy as np
import torch

from unswayed.config import load_config
from unswayed.dreamer.behavior import ACTOR_VARIANT
from unswayed.training import ProductionTrainer


def main() -> int:
    args = parse_args()
    overrides = _overrides(args)
    config = load_config(args.config, overrides)
    _set_seed(int(config.training.seed))
    output = Path(args.output).expanduser().resolve()
    _print_startup(config, output, args.resume)
    trainer = ProductionTrainer(
        config,
        output,
        env_backend=args.env_backend,
        paired_backend=args.paired_backend,
        vlm_backend=args.vlm_backend,
        device=args.device,
    )

    def request_stop(signum, _frame):
        print(f"[UNSWAYED] received signal {signum}; saving at the next slice boundary", flush=True)
        trainer.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if bool(config.get("debug", {}).get("enabled", False)):
        trainer.debug_once(checkpoint=args.resume, collect_steps_per_env=int(args.debug_steps))
    state = trainer.train(total_env_steps=args.steps, resume=args.resume)
    print(
        f"[UNSWAYED] finished env_step={state.global_env_steps} update={state.gradient_updates} "
        f"checkpoint={output / 'checkpoints' / 'latest.pt'}",
        flush=True,
    )
    return 0


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    if args.seed is not None:
        values["training.seed"] = int(args.seed)
        values["env.davis.seed"] = int(args.seed)
    if args.device is not None:
        values["training.device"] = args.device
        values["r2_rssm.device"] = args.device
    if args.steps is not None:
        values["training.steps"] = int(args.steps)
    if args.no_vgc:
        values["ablation.use_vgc"] = False
    if args.no_msir:
        values["ablation.use_msir"] = False
    if args.no_capc:
        values["ablation.use_capc"] = False
    for name in ("lambda_vgc", "lambda_msir", "lambda_capc"):
        value = getattr(args, name)
        if value is not None:
            values[f"ablation.{name}"] = float(value)
    if args.debug:
        values["debug.enabled"] = True
    return values


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_startup(config, output: Path, resume: str | None) -> None:
    rows = {
        "PROJECT": config.project.name,
        "ENV": config.env.task,
        "SEED": int(config.training.seed),
        "DEVICE": config.training.device,
        "K": int(config.model.num_views),
        "VGC": "ON" if config.model.vlm_gaussian_consensus.enabled else "OFF",
        "MSIR": "ON" if config.model.msir.enabled else "OFF",
        "CAPC": "ON" if config.capc.enabled else "OFF",
        "ACTOR_VARIANT": ACTOR_VARIANT,
        "OPTIMIZER": "LaProp",
        "RESUME": resume or "NONE",
        "OUTPUT_DIR": output,
    }
    print("\n".join(f"{key} = {value}" for key, value in rows.items()), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNSWAYED.")
    parser.add_argument("--config", default="configs/walker_walk.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--env-backend", choices=["dmc", "synthetic"])
    parser.add_argument("--paired-backend", choices=["davis", "mock"])
    parser.add_argument("--vlm-backend", choices=["resident", "mock"])
    parser.add_argument("--no-vgc", action="store_true")
    parser.add_argument("--no-msir", action="store_true")
    parser.add_argument("--no-capc", action="store_true")
    parser.add_argument("--lambda-vgc", type=float)
    parser.add_argument("--lambda-msir", type=float)
    parser.add_argument("--lambda-capc", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-steps", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
