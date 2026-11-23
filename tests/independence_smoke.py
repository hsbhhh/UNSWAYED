from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dreamer_v5.config import load_config
from dreamer_v5.model import V4WorldModel
from dreamer_v5.training.losses import compute_world_model_losses
from dreamer_v5.training.synthetic_data import make_synthetic_sequence
from dreamer_v5.training.trainer import ProductionTrainer


def main() -> int:
    args = parse_args()
    device = args.device
    overrides = {
        "training.device": device,
        "r2_rssm.device": device,
        "training.batch_size": 1,
        "training.sequence_length": 2,
        "training.steps": 2,
        "training.train_ratio": 1,
        "training.log_every": 1,
        "training.checkpoint_every": 1,
        "env.backend": "synthetic",
        "env.env_num": 1,
        "env.synthetic.episode_length": 2,
        "env.davis.backend": "mock",
        "external_vlm.backend": "mock",
        "pipeline.vlm.start_managed_server": False,
        "pipeline.shard.collect_steps_per_env": 2,
        "pipeline.shard.transitions_per_shard": 2,
        "logging.tensorboard": False,
    }
    config = load_config(args.config, overrides)
    _seed(123)
    model = V4WorldModel(config).to(device)
    batch, _ = make_synthetic_sequence(
        batch_size=1,
        sequence_length=2,
        action_dim=int(config.model.action_dim),
        image_size=int(config.model.image_size),
        patch_count=int(config.external_vlm.patch_count),
        feature_dim=int(config.external_vlm.feature_dim),
        seed=123,
        num_views=int(config.model.num_views),
        schema_version=str(config.replay.schema_version),
    )
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        no_update_loss, _, _ = compute_world_model_losses(model, batch)

    output = Path(args.output).expanduser().resolve()
    trainer = ProductionTrainer(
        config,
        output,
        env_backend="synthetic",
        paired_backend="mock",
        vlm_backend="mock",
        device=device,
    )
    trained = trainer.train(total_env_steps=2)
    checkpoint = output / "checkpoints" / "latest.pt"

    resume_config = load_config(args.config, overrides)
    resumed_trainer = ProductionTrainer(
        resume_config,
        output,
        env_backend="synthetic",
        paired_backend="mock",
        vlm_backend="mock",
        device=device,
    )
    resumed = resumed_trainer.train(total_env_steps=2, resume=checkpoint)
    report = {
        "status": "PASS",
        "import": True,
        "model_created": True,
        "no_update_loss_finite": bool(torch.isfinite(no_update_loss).item()),
        "short_train_update": int(trained.gradient_updates),
        "resume_step": int(resumed.global_env_steps),
        "checkpoint": str(checkpoint),
    }
    (output / "independence_smoke.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/walker_walk.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
