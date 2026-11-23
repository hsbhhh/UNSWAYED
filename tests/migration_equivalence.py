from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from dreamer_v5.config import load_config as load_v5_config
from dreamer_v5.dreamer.behavior import DreamerBehavior as V5Behavior
from dreamer_v5.model import V4WorldModel as V5WorldModel
from dreamer_v5.training.checkpoint import load_checkpoint as load_v5_checkpoint
from dreamer_v5.training.losses import compute_world_model_loss_components as v5_components
from dreamer_v5.training.synthetic_data import make_synthetic_sequence


def main() -> int:
    args = parse_args()
    v4_root = Path(args.v4_root).expanduser().resolve()
    sys.path.insert(0, str(v4_root))
    from v4.config import load_config as load_v4_config
    from v4.dreamer.behavior import DreamerBehavior as V4Behavior
    from v4.model import V4WorldModel
    from v4.training.checkpoint import load_checkpoint as load_v4_checkpoint
    from v4.training.losses import compute_world_model_loss_components as v4_components

    overrides = {"training.device": args.device, "r2_rssm.device": args.device}
    c4 = load_v4_config(args.v4_config, overrides)
    c5 = load_v5_config(args.v5_config, overrides)
    device = torch.device(args.device)
    m4, m5 = V4WorldModel(c4).to(device).eval(), V5WorldModel(c5).to(device).eval()
    b4 = V4Behavior(c4, belief_dim=m4.belief_dim, action_dim=m4.action_dim, device=device).to(device).eval()
    b5 = V5Behavior(c5, belief_dim=m5.belief_dim, action_dim=m5.action_dim, device=device).to(device).eval()
    load_v4_checkpoint(args.checkpoint, config=c4, model=m4, behavior=b4, restore_training_state=False)
    load_v5_checkpoint(args.checkpoint, config=c5, model=m5, behavior=b5, restore_training_state=False)

    batch, _ = make_synthetic_sequence(
        batch_size=1,
        sequence_length=3,
        action_dim=int(c5.model.action_dim),
        image_size=int(c5.model.image_size),
        patch_count=int(c5.external_vlm.patch_count),
        feature_dim=int(c5.external_vlm.feature_dim),
        seed=20260906,
        num_views=int(c5.model.num_views),
        schema_version=str(c5.replay.schema_version),
    )
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    checks: dict[str, dict[str, float | bool]] = {}

    with torch.no_grad():
        enc4 = m4.encode_observation(batch["raw_a"])
        enc5 = m5.encode_observation(batch["raw_a"])
    _check(checks, "encoder_output", enc4, enc5)

    _seed(991)
    comp4 = v4_components(m4, batch, update_index=17)
    _seed(991)
    comp5 = v5_components(m5, batch, update_index=17)
    for name, left, right in (
        ("rssm_posterior_logits", comp4.output.rssm_post_logit, comp5.output.rssm_post_logit),
        ("rssm_prior_logits", comp4.output.rssm_prior_logit, comp5.output.rssm_prior_logit),
        ("belief", comp4.output.online_belief, comp5.output.online_belief),
        ("reward_prediction", comp4.output.reward_pred, comp5.output.reward_pred),
        ("continue_prediction", comp4.output.continue_logit, comp5.output.continue_logit),
        ("vgc_prediction", comp4.output.vgc_patch_prediction, comp5.output.vgc_patch_prediction),
        ("msir_reconstruction", comp4.output.rgb_reconstruction, comp5.output.rgb_reconstruction),
        ("vgc_raw_loss", comp4.metrics["vgc/raw"], comp5.metrics["vgc/raw"]),
        ("vgc_weighted_loss", comp4.cvgc, comp5.cvgc),
        ("msir_weighted_loss", comp4.msir, comp5.msir),
        ("capc_raw_loss", comp4.capc_raw, comp5.capc_raw),
        ("capc_weighted_loss", comp4.capc, comp5.capc),
        ("world_total_loss", comp4.total, comp5.total),
    ):
        _check(checks, name, left, right)

    feat4 = comp4.output.online_belief.detach()
    feat5 = comp5.output.online_belief.detach()
    with torch.no_grad():
        p4, p5 = b4.actor(feat4), b5.actor(feat5)
        _check(checks, "actor_mean", p4.mean, p5.mean)
        _check(checks, "actor_std", p4.base_dist.scale, p5.base_dist.scale)
        _check(checks, "critic_value", b4.value(feat4).mode(), b5.value(feat5).mode())
        _check(checks, "deterministic_action", p4.mean.clamp(-1, 1), p5.mean.clamp(-1, 1))

    start4 = (
        comp4.output.stoch.reshape(-1, *comp4.output.stoch.shape[2:]).detach(),
        comp4.output.deter.reshape(-1, comp4.output.deter.shape[-1]).detach(),
    )
    start5 = (
        comp5.output.stoch.reshape(-1, *comp5.output.stoch.shape[2:]).detach(),
        comp5.output.deter.reshape(-1, comp5.output.deter.shape[-1]).detach(),
    )
    _seed(711)
    imag4, pre4, executed4 = b4.imagine(start4, m4, horizon=4)
    _seed(711)
    imag5, pre5, executed5 = b5.imagine(start5, m5, horizon=4)
    _check(checks, "pre_action", pre4, pre5)
    _check(checks, "executed_action", executed4, executed5)
    logpi4 = b4.actor(imag4.detach()).log_prob(pre4.detach())
    logpi5 = b5.actor(imag5.detach()).log_prob(pre5.detach())
    _check(checks, "actor_logpi", logpi4, logpi5)

    _seed(812)
    loss4 = b4.compute_losses(m4, comp4.output.stoch, comp4.output.deter, batch)
    _seed(812)
    loss5 = b5.compute_losses(m5, comp5.output.stoch, comp5.output.deter, batch)
    _check(checks, "actor_loss", loss4.actor_loss, loss5.actor_loss)
    _check(checks, "critic_loss", loss4.value_loss, loss5.value_loss)

    passed = all(bool(row["allclose"]) for row in checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checks": checks,
        "note": "One-time build validation only; V5 runtime does not consume this report or V4.",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(report), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


def _check(checks, name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    a, b = left.detach().float(), right.detach().float()
    diff = (a - b).abs()
    checks[name] = {
        "allclose": bool(torch.allclose(a, b, atol=ARGS.atol, rtol=ARGS.rtol)),
        "max_abs_diff": float(diff.max().cpu()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().cpu()) if diff.numel() else 0.0,
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _markdown(report: dict) -> str:
    rows = [
        "# Migration Equivalence Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Check | Allclose | Max abs diff | Mean abs diff |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["checks"].items():
        rows.append(
            f"| {name} | {result['allclose']} | {result['max_abs_diff']:.8g} | {result['mean_abs_diff']:.8g} |"
        )
    rows.extend(["", report["note"], ""])
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", required=True)
    parser.add_argument("--v4-config", required=True)
    parser.add_argument("--v5-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    return parser.parse_args()


ARGS = parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
