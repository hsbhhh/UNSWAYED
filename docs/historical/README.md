# DreamerV3-VLM V5

V5 is the independent production package distilled from the final validated V4/u252544 implementation. It preserves the trained model math and E1 actor behavior while removing historical authorization, calibration, validator, evidence-bundle, and hash-gate runtime paths.

## Environment

Run from this repository root. The production server currently uses:

- training Python: the active Dreamer environment with PyTorch, NumPy, PyYAML, dm-control, Pillow, TensorBoard and imageio;
- resident VLM Python: `/root/autodl-tmp/hsb/conda_envs/dreamerv3_vlm_flash/bin/python3.11`;
- DAVIS frames: `/root/autodl-tmp/hsb/DAVIS/JPEGImages/480p`;
- frozen VLM checkpoint: `/autodl-fs/data/hsb/walker_vlm_finetune/dmc_patch_relevance_v1/stage2/best.pt`;
- InternVideo2 checkpoint: `/autodl-fs/data/models/internvideo2/InternVideo2-CLIP-1B-224p-f8/1B_clip.pth`;
- fine-turn runtime: `/root/autodl-tmp/hsb/get_data/fine-turn`.

These are explicit data/model/runtime resources. V5 never imports or reads the V4 source directory.

## Train

```bash
cd /root/autodl-tmp/hsb/dreamerv3_vlm/dreamerv3_vlm_v5
python train.py --config configs/walker_walk.yaml --output outputs/walker_seed0
```

Use another seed without editing source:

```bash
python train.py --config configs/walker_walk.yaml --output outputs/walker_seed1 --seed 1
```

## Resume

```bash
python train.py \
  --config configs/walker_walk.yaml \
  --output outputs/walker_seed0 \
  --resume outputs/walker_seed0/checkpoints/latest.pt
```

Resume is strict for model keys and restores behavior, both optimizers, both schedulers, counters, RNG, replay references and shard-store state from a V5 full checkpoint.

## Evaluation

```bash
python eval.py --config configs/walker_walk.yaml \
  --checkpoint outputs/walker_seed0/checkpoints/latest.pt \
  --domain clean --episodes 10 --deterministic
```

`--domain` supports `clean`, `seen_davis`, and `unseen_davis`. Results include return mean/std, episode returns/lengths, action RMS, deterministic saturation, curves, optional videos, and a cumulative `eval/eval.csv`.

## Lightweight debug visualizations

```bash
python debug.py --config configs/walker_walk.yaml \
  --checkpoint outputs/walker_seed0/checkpoints/latest.pt \
  --output-dir outputs/walker_seed0/debug/latest
```

The command captures a small online batch and writes A/K-view inputs, the semantic heatmap, decoder reconstruction, three MSIR pyramid levels, and VGC/MSIR/CAPC numeric summaries. It does not build a forensic evidence package.

## Ablations

The default model enables VGC, MSIR and CAPC. CLI examples:

```bash
python train.py --config configs/walker_walk.yaml --output outputs/no_capc --no-capc
python train.py --config configs/walker_walk.yaml --output outputs/no_msir --no-msir
python train.py --config configs/walker_walk.yaml --output outputs/vgc_weight --lambda-vgc 0.015625
```

The public config keys are `ablation.use_vgc`, `use_msir`, `use_capc`, `lambda_vgc`, `lambda_msir`, and `lambda_capc`. Ablations only change loss aggregation; model shapes remain fixed.

## Outputs

Each experiment has a compact layout:

```text
outputs/<experiment>/
├── config.yaml
├── checkpoints/
│   ├── latest.pt
│   ├── step_*.pt
│   └── shards/
├── logs/
│   ├── metrics.jsonl
│   ├── episodes.csv
│   └── tensorboard/
├── eval/
│   └── eval.csv
├── debug/
└── videos/
```

`checkpoint.keep_last_n` controls periodic retention. The actor health monitor is warning-only by default; set `logging.behavior_safety_stop: true` only when an experiment intentionally requires automatic stopping.

## Model identity

The architecture, losses, E1 actor, LaProp optimizer and frozen VLM route are documented in [MODEL.md](MODEL.md). The checkpoint marker/revision remain compatible with the final u252544 checkpoint; this compatibility value is not a V4 source dependency.

The final reference checkpoint remains in durable storage and is not duplicated:

`/autodl-fs/data/hsb/dreamerv3_vlm_outputs/actor_e1_forced_completion_u226610_049_20260905_123341_clean2/09_checkpoints/formal_terminal_u252544.pt`
