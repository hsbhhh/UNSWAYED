from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


from dreamer_v5.config import load_config
from dreamer_v5.model.world_model import V4WorldModel
from dreamer_v5.semantic.semantic_lowrank_gaussian import (
    k_view_laplacian_pyramid,
    laplacian_pyramid,
    prepare_k_view_reconstruction_images,
    prepare_reconstruction_image,
)
from dreamer_v5.semantic.semantic_weighting import semantic_weighting_from_batch
from dreamer_v5.training.checkpoint import load_checkpoint
from dreamer_v5.training.losses import compute_world_model_losses
from dreamer_v5.training.trainer import ProductionTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Export lightweight V5 K-view/model visualizations.")
    parser.add_argument("--config", default="configs/walker_walk.yaml")
    parser.add_argument("--records")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-backend", choices=["dmc", "synthetic"])
    parser.add_argument("--paired-backend", choices=["davis", "mock"])
    parser.add_argument("--vlm-backend", choices=["resident", "mock"])
    parser.add_argument("--collect-steps", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(
        args.config,
        {"training.device": args.device, "r2_rssm.device": args.device},
    )
    records_path = None if args.records is None else Path(args.records).expanduser().resolve()
    if records_path is None:
        capture_root = output_dir / "_capture"
        trainer = ProductionTrainer(
            config,
            capture_root,
            env_backend=args.env_backend,
            paired_backend=args.paired_backend,
            vlm_backend=args.vlm_backend,
            device=args.device,
        )
        trainer.debug_once(
            checkpoint=args.checkpoint,
            collect_steps_per_env=int(args.collect_steps),
        )
        trainer.logger.close()
        records_path = capture_root / "debug" / "debug_records.pt"
    payload = torch.load(records_path, map_location="cpu", weights_only=False)
    batch = payload.get("batch")
    if not isinstance(batch, dict):
        raise RuntimeError("debug_records.pt does not contain a batch")
    weights = semantic_weighting_from_batch(batch, config, device="cpu")
    frame = _select_frame(batch, weights.fused_weights)
    raw_views = _k_view_rgb(torch.as_tensor(batch["raw_b_views"]))
    count = int(raw_views.shape[2])
    if count != int(config.model.num_views):
        raise ValueError(f"records contain K={count}, config requires K={int(config.model.num_views)}")

    device = torch.device(args.device)
    model = V4WorldModel(config).to(device).eval()
    load_report = load_checkpoint(
        args.checkpoint,
        config=config,
        model=model,
        restore_training_state=False,
    )
    batch_device = {
        key: value.to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)
        if isinstance(value, np.ndarray)
        else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        forward = model.forward_training_batch(batch_device)
        decoder = prepare_reconstruction_image(forward.rgb_reconstruction).cpu()
        world_loss, world_metrics, _ = compute_world_model_losses(model, batch_device)

    for view in range(count):
        _save_rgb(output_dir / f"view_{view}.png", raw_views[0, frame, view])
    mean_target = raw_views.mean(dim=2)
    _save_rgb(output_dir / "kview_mean_target.png", mean_target[0, frame])
    _save_rgb(output_dir / "decoder_reconstruction.png", decoder[0, frame])
    _save_heatmap(output_dir / "semantic_weight_map.png", weights.fused_weights[0, frame])

    target_levels = k_view_laplacian_pyramid(raw_views, levels=int(config.model.msir.pyramid_levels))
    decoder_levels = laplacian_pyramid(decoder, levels=int(config.model.msir.pyramid_levels))
    pyramid_shapes = []
    for level, (target_views, decoder_level) in enumerate(zip(target_levels, decoder_levels, strict=True)):
        target = target_views.mean(dim=2)
        _save_coefficient(output_dir / f"msir_target_L{level}.png", target[0, frame])
        _save_coefficient(output_dir / f"decoder_L{level}.png", decoder_level[0, frame])
        pyramid_shapes.append(
            {
                "level": level,
                "view_target_shape": list(target_views.shape),
                "mean_target_shape": list(target.shape),
                "decoder_shape": list(decoder_level.shape),
            }
        )

    blur = _foreground_blur_audit(raw_views[0, frame], mean_target[0, frame], batch, frame)
    report = {
        "status": "PASS",
        "configured_num_views": int(config.model.num_views),
        "actual_msir_views": count,
        "selected_batch": 0,
        "selected_frame": frame,
        "raw_view_shape": list(raw_views.shape),
        "mean_target_shape": list(mean_target.shape),
        "decoder_shape": list(decoder.shape),
        "pyramid_shapes": pyramid_shapes,
        "foreground_blur_audit": blur,
        "checkpoint_load": load_report.to_dict(),
        "world_summary": {
            "total": float(world_loss.detach().cpu()),
            "vgc_raw": float(world_metrics.get("vgc/raw", torch.zeros(())).detach().cpu()),
            "vgc_weighted": float(world_metrics.get("vgc/scaled", torch.zeros(())).detach().cpu()),
            "msir_weighted": float(world_metrics.get("msir/loss", torch.zeros(())).detach().cpu()),
            "capc_raw": float(world_metrics.get("capc/raw", torch.zeros(())).detach().cpu()),
            "capc_weighted": float(world_metrics.get("capc/weighted", torch.zeros(())).detach().cpu()),
        },
        "files": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "k_view_debug_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


def _k_view_rgb(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.ndim != 6:
        raise ValueError(f"raw_b_views must be rank 6, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 1, 2, 5, 3, 4).contiguous()
    return prepare_k_view_reconstruction_images(tensor)


def _select_frame(batch: dict, weights: torch.Tensor) -> int:
    mask = _mask(batch.get("task_foreground_mask"))
    if mask is None:
        return 0
    grid = int(round(weights.shape[-1] ** 0.5))
    pooled = F.adaptive_avg_pool2d(mask.reshape(-1, 1, *mask.shape[-2:]), (grid, grid)).reshape_as(weights)
    advantage = (weights * pooled).sum(-1) / pooled.sum(-1).clamp_min(1e-6)
    advantage -= (weights * (1 - pooled)).sum(-1) / (1 - pooled).sum(-1).clamp_min(1e-6)
    return int(advantage[0].argmax())


def _foreground_blur_audit(views: torch.Tensor, mean: torch.Tensor, batch: dict, frame: int) -> dict:
    mask = _mask(batch.get("task_foreground_mask"))
    region = torch.ones_like(mean[0], dtype=torch.bool) if mask is None else mask[0, frame] >= 0.5
    per_view_edges = torch.stack([_edge_energy(view, region) for view in views])
    mean_edge = _edge_energy(mean, region)
    reference = per_view_edges.mean().clamp_min(1e-8)
    ratio = mean_edge / reference
    return {
        "foreground_edge_energy_per_view_mean": float(reference),
        "foreground_edge_energy_kview_mean": float(mean_edge),
        "kview_mean_to_per_view_edge_ratio": float(ratio),
        "obvious_foreground_blur_detected": bool(float(ratio) < 0.70),
        "threshold": 0.70,
    }


def _edge_energy(image: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0)
    dx = (gray[:, 1:] - gray[:, :-1]).abs()
    dy = (gray[1:] - gray[:-1]).abs()
    mx = region[:, 1:] & region[:, :-1]
    my = region[1:] & region[:-1]
    values = torch.cat([dx[mx], dy[my]])
    return values.mean() if values.numel() else torch.zeros(())


def _mask(value) -> torch.Tensor | None:
    if value is None:
        return None
    mask = torch.as_tensor(value).float()
    if mask.ndim == 5 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim == 5 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    return mask if mask.ndim == 4 else None


def _save_rgb(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255).astype(np.uint8), mode="RGB").save(path)


def _save_coefficient(path: Path, chw: torch.Tensor) -> None:
    value = chw.detach().float().cpu()
    low = torch.quantile(value, 0.01)
    high = torch.quantile(value, 0.99)
    normalized = ((value - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
    _save_rgb(path, normalized)


def _save_heatmap(path: Path, flat: torch.Tensor) -> None:
    side = int(round(flat.numel() ** 0.5))
    value = flat.detach().float().reshape(1, 1, side, side)
    value = F.interpolate(value, size=(256, 256), mode="nearest")[0, 0]
    normalized = ((value - value.min()) / (value.max() - value.min()).clamp_min(1e-8)).cpu().numpy()
    rgb = np.stack([normalized, 1.0 - np.abs(2.0 * normalized - 1.0), 1.0 - normalized], axis=-1)
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
