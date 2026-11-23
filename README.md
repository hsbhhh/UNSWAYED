# UNSWAYED — Walker-walk

The Walker-walk implementation corresponding to the trained UNSWAYED checkpoint
\`formal_terminal_u252544.pt\`. The model reached 1,010,176 recorded environment
interactions and 252,544 model updates.

## Model

The historical implementation combines a convolutional visual encoder, a categorical
RSSM, an E1 actor/critic, State Visual Gaussian Consensus (VGC), multi-scale image
reconstruction (MSIR), and Cross-view Posterior Consistency (CPC, named CAPC in code).

| Component | Walker configuration |
| --- | --- |
| Environment | DMC Walker-walk, camera 0, 64 × 64 RGB |
| Action repeat / episode | 2 / 500 policy decisions |
| Recurrent state | 256 deterministic units, 32 × 32 categorical latent |
| Actor/readout input | \`[flatten(z), h]\`, 1,280 dimensions |
| Auxiliary views | 4 matched physical states with different DAVIS backgrounds |
| State VGC | 64 patches × 64 features, loss coefficient 0.03125 |
| Spatial relevance | mean-one softmax weights, temperature 1.0 |
| Covariance | \`0.2525 I + U Uᵀ\`, centered columns scaled by \`1/sqrt(K-1)\` |
| CPC | \`mean KL(stopgrad(mean_k posterior) || prior_k)\`, coefficient 3.0 |
| Decoder / MSIR | historical \`TaskRGBDecoder\`, three pyramid levels, coefficient 4.0 |
| Optimization | LaProp, AGC and warmup |

The Python package name \`dreamer_v5\` and historical V4 checkpoint markers are retained
for strict checkpoint compatibility. The complete formulas are in [MODEL.md](MODEL.md).

## Source layout

\`\`\`text
configs/                 Walker configuration and DAVIS split
dreamer_v5/
  model/                 encoder, RSSM interface, TaskRGBDecoder, State VGC readout
  dreamer/               recurrent dynamics, E1 actor/critic, distributions, returns
  semantic/              spatial weighting, Woodbury VGC and MSIR
  capc/                  categorical cross-view posterior consensus
  vlm/                   teacher, resident server and client protocol
  pipeline/              independent background streams and replay shards
  training/              native objectives, optimizer, checkpoint and trainer
external_vlm/
  get_data/              archived fine-turn package and visual adapter
  InternVideo/           required InternVideo2 multimodality source and upstream license
tools/                   debug, resource configuration and checkpoint verification
docs/                    provenance, validation, recorded evaluation, historical notes
train.py                 native training entry point
eval.py                  closed-loop evaluation entry point
debug.py                 reconstruction and supervision visualization
\`\`\`

## Environment and resources

Use Python 3.11 and a recent PyTorch with \`torch.nn.RMSNorm\`. Policy inference can
run on CPU. The archived resident InternVideo2 teacher uses a Linux CUDA environment
with FlashAttention and its fused MLP/RMSNorm extensions. Its additional dependencies
are recorded in \`external_vlm/get_data/fine-turn/requirements-finetune.txt\`.

\`\`\`bash
python -m pip install -r requirements.txt
\`\`\`

Resource files are supplied separately:

* DAVIS \`JPEGImages/480p\`, using the video splits in \`configs/davis_splits.json\`.
* The trained Walker policy checkpoint.
* The Walker relevance-head checkpoint, originally
  \`walker_vlm_finetune/dmc_patch_relevance_v1/stage2/best.pt\`.
* InternVideo2 CLIP \`1B_clip.pth\` and its stage-2 visual checkpoint.
* Any tokenizer/text resources required by the archived default teacher loader.

The resource adapter and its discovery rules are retained in
\`external_vlm/get_data/adapters/internvideo2_adapter.py\`. The original default
teacher loader can instantiate the upstream text branch. The historical vision-only
analysis adapter is preserved under \`docs/historical/\` for provenance.

Set paths for your machine by generating a local configuration:

\`\`\`bash
python tools/configure_paths.py \
  --davis-root /path/to/DAVIS/JPEGImages/480p \
  --vlm-checkpoint /path/to/walker/stage2/best.pt \
  --base-checkpoint /path/to/InternVideo2/1B_clip.pth \
  --vlm-python /path/to/flashattention/environment/bin/python
\`\`\`

This writes \`configs/walker_walk.local.yaml\` and prints the environment settings for
the bundled VLM source. Use the weights directory containing the official visual
resources and tokenizer when setting \`INTERNVIDEO2_WEIGHTS_ROOT\`. Generated
configurations and local paths are ignored by Git.

## Verify an existing checkpoint

\`\`\`bash
python tools/verify_checkpoint.py \
  --checkpoint /path/to/formal_terminal_u252544.pt --device cpu
\`\`\`

The command strictly restores the world model, actor, critic and slow critic, then
runs two synthetic observation steps without constructing an optimizer.

## Evaluate

\`\`\`bash
python eval.py --config configs/walker_walk.local.yaml \
  --checkpoint /path/to/formal_terminal_u252544.pt \
  --domain clean --episodes 5 --episode-steps 500 --deterministic --device cpu --no-video
\`\`\`

For background evaluation, use \`--domain seen_davis\` or \`--domain unseen_davis\`.
The frozen VLM is an auxiliary training teacher; policy evaluation consumes the
observation encoder, RSSM and actor. DAVIS evaluation requires the actual video frames.

## Train and resume

\`\`\`bash
python train.py --config configs/walker_walk.local.yaml --output outputs/walker_seed0 --seed 0

python train.py --config configs/walker_walk.local.yaml --output outputs/walker_seed0 \
  --resume outputs/walker_seed0/checkpoints/latest.pt
\`\`\`

Training continuation requires a full checkpoint with optimizer/RNG state and its
referenced replay shards. A compact terminal policy checkpoint supports inference
and model initialization. Its available fields must be checked before attempting
a training-state restore.

## Recorded evaluation

The terminal checkpoint's recorded evaluation used two complete episodes per domain:

| Condition | Mean return | Episode SD |
| --- | ---: | ---: |
| Clean | 947.2970 | 15.6432 |
| Seen DAVIS | 947.4193 | 0.4316 |
| Unseen DAVIS | 906.4125 | 24.1626 |

A subsequent evaluation of the same checkpoint used five environment reset seeds
and three episodes per seed, obtaining Clean 934.614, ID 955.765 and OOD 922.858.
These are repeated evaluations of one trained model. Full evaluation definitions
and provenance are in [docs/evaluation_summary.json](docs/evaluation_summary.json).

## Provenance and validation

The server's preserved V5 production export supplies the Walker core.
\`docs/source_manifest.json\` records every downloaded source file's SHA-256,
including the external runtime. Original Git metadata was unavailable for this
historical export and upstream copy. The manifest records that explicitly.
The original source README and lineage notes are under \`docs/historical/\`.

The publication validation checks syntax, source hashes and strict checkpoint
loading. See [docs/validation.json](docs/validation.json).

## Third-party code

The bundled InternVideo2 source comes from [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo).
Its upstream license is retained at \`external_vlm/InternVideo/LICENSE\`.
That license applies to the upstream code it covers. Training images and pretrained
model weights are separate resources with their own terms.
