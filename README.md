# UNSWAYED

Walker-walk visual world model with State VGC, CPC, and image reconstruction.

## Setup

Python 3.11+, PyTorch with `torch.nn.RMSNorm`. The frozen InternVideo2 teacher
requires its CUDA dependencies. DAVIS videos and pretrained weights are supplied separately.

```bash
pip install -r requirements.txt
python tools/configure_paths.py \
  --davis-root /path/to/DAVIS/JPEGImages/480p \
  --vlm-checkpoint /path/to/walker/stage2/best.pt \
  --base-checkpoint /path/to/InternVideo2/1B_clip.pth \
  --vlm-python /path/to/vlm/environment/bin/python
```

Apply the environment variables printed by this command. The teacher loader also
requires the upstream visual, tokenizer, and text resources in the weights directory.

## Train and evaluate

```bash
python train.py --config configs/walker_walk.local.yaml --output outputs/walker
python eval.py --config configs/walker_walk.local.yaml \
  --checkpoint outputs/walker/checkpoints/latest.pt --domain seen_davis
```

Resume with `train.py --resume /path/to/checkpoint.pt` and the same config/output arguments.
Evaluation domains: `clean`, `seen_davis`, `unseen_davis`.
Verify existing weights with `python tools/verify_checkpoint.py --checkpoint /path/to/checkpoint.pt`.

## Layout

- `unswayed/`: model, objectives, training, replay, and environments.
- `configs/`: Walker settings and DAVIS splits.
- `third_party/`: required VLM runtime and upstream license.
- `tools/`: resource setup, checkpoint verification, and visualization.

InternVideo2 runtime is adapted from [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo);
its license is retained in `third_party/InternVideo/LICENSE`.
