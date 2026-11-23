# Model Definition

## Fixed production architecture

- Input: 64×64 RGB, 6-dimensional continuous Walker action.
- Visual path: canonical convolutional stem with 64 channels followed by a 256-dimensional Dreamer encoder tail.
- RSSM: 32 stochastic variables × 32 categorical classes, deterministic state 256, hidden size 512, two imagination layers, one observation layer, one dynamics layer, eight blocks, SiLU, unimix 0.01.
- Belief: concatenated stochastic and deterministic RSSM feature used by the reward head, continuation head, VGC patch predictor, RGB/MSIR decoder, actor and critic.
- Actor: three 256-wide hidden layers, output scale 0.01, Normal standard deviation transformed to [0.1, 1.0].
- Critic: three 256-wide hidden layers and 255-bin symexp two-hot output; slow value target fraction 0.02.
- Frozen VLM: resident InternVideo2 latest causal 8-frame clip, native FlashAttention backend, K=4 independent view streams, 8×8 score/feature grid, 64-dimensional patch features.

## World-model objective

The default loss is unchanged:

```text
L_world = L_dyn + 0.1 L_rep + L_reward + L_continue
        + 0.03125 L_VGC_raw
        + 4.0 L_MSIR_raw
        + 3.0 L_CAPC_raw
```

VGC uses the K=4 empirical low-rank Gaussian covariance and standard-softmax semantic spatial weighting. MSIR uses three Laplacian-pyramid levels weighted `[1.0, 0.5, 0.25]`. CAPC is the KL from the stop-gradient mean K-view posterior categorical probability to the native action-conditioned RSSM prior, using RSSM unimix probabilities.

No temporal-delta, ACTP, NativeRec, E2 censored likelihood or TanhNormal branch is active or shipped as a production alternative.

## Exact E1 actor behavior

```python
pre_action = policy.rsample()
executed_action = pre_action.clamp(-1.0, 1.0)
next_state = rssm.img_step(stoch, deter, executed_action.detach())
logpi = policy.log_prob(pre_action.detach())
```

The environment and imagined world receive `executed_action`. The policy-gradient score is evaluated on the retained `pre_action`. Deterministic evaluation uses the actor mean and clamps it to the action bounds. There is no censored-normal/E2 calculation in the V5 training path.

The behavior objective remains actor policy-gradient plus entropy regularization, imagined critic loss, and replay value loss weighted by 0.3.

## Optimizer

World model and behavior use separate LaProp optimizers:

- learning rate 4e-5;
- beta1 0.9, beta2 0.999;
- epsilon 1e-20;
- no weight decay;
- AGC 0.3 with pmin 1e-3;
- 1000-update warmup.

## Runtime safety and observability

V5 keeps strict checkpoint keys, replay schema checks, K-view shapes, action shapes, missing resource errors, VLM identity/connection checks, and NaN/Inf checks. It logs world loss components, episode return/length, replay size, throughput, VLM/K-view status, actor/critic losses, imagined return, advantage, action RMS, actor standard deviation, entropy and stochastic/deterministic saturation.

Historical source/checkpoint/dataset hash gates, protocol JSON authorization, independent validators and causal forensic diagnostics are not runtime dependencies.
