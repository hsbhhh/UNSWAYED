from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

import torch

from ..model.visual_preprocessing import preprocess_observation


class CAPCBatchInvalid(RuntimeError):
    """Raised when a replay batch cannot support a trustworthy CAPC target."""


@dataclass
class CAPCResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    artifacts: dict[str, torch.Tensor]


def canonical_capc_objective(
    posterior_prob: torch.Tensor,
    prior_prob: torch.Tensor,
    valid_transition_mask: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure canonical CAPC objective used by production training.

    Inputs are categorical probabilities with shape ``[B,T,K,G,C]``.  The
    valid mask may be ``[B,T]`` (one validity bit per physical transition) or
    ``[B,T,K]`` (one bit per transition/view).  The returned consensus is
    always detached.  An all-invalid mask returns a differentiable zero linked
    only to the prior graph so boundary-only unit tests cannot create NaNs.
    """

    if posterior_prob.shape != prior_prob.shape or posterior_prob.ndim != 5:
        raise ValueError(
            "CAPC expects matching posterior/prior [B,T,K,G,C], got "
            f"{tuple(posterior_prob.shape)} and {tuple(prior_prob.shape)}"
        )
    if posterior_prob.shape[-1] < 2 or posterior_prob.shape[2] < 2:
        raise ValueError("CAPC requires at least two views and two categorical classes")
    mask = torch.as_tensor(valid_transition_mask, device=prior_prob.device).bool()
    if tuple(mask.shape) == tuple(prior_prob.shape[:2]):
        expanded_mask = mask[:, :, None].expand(prior_prob.shape[:3])
    elif tuple(mask.shape) == tuple(prior_prob.shape[:3]):
        expanded_mask = mask
    else:
        raise ValueError(
            f"CAPC valid mask must be [B,T] or [B,T,K], got {tuple(mask.shape)}"
        )
    consensus = posterior_prob.float().mean(dim=2).detach()
    kl = (
        consensus[:, :, None]
        * (
            consensus[:, :, None].clamp_min(float(eps)).log()
            - prior_prob.float().clamp_min(float(eps)).log()
        )
    ).sum(dim=-1)  # [B,T,K,G]
    selected = expanded_mask[:, :, :, None].expand_as(kl)
    if bool(selected.any()):
        loss = kl.masked_select(selected).mean()
    else:
        loss = prior_prob.sum() * 0.0
    return loss, consensus


def compute_capc_loss(
    model,
    batch: dict[str, torch.Tensor],
    config,
    *,
    update_index: int | None = None,
    force_diagnostics: bool = False,
) -> CAPCResult:
    """Compute KL(stopgrad(mean_k q_{t+1}^k) || p_{t+1}^k).

    The K counterfactual RGB streams are rolled through the *native* encoder,
    posterior, recurrent transition, and categorical prior.  Replay action
    semantics are ``obs_t_with_incoming_action``; consequently index ``j``
    (j >= 1) is the physical transition ``j-1 -> j`` and already carries the
    action executed at ``j-1``.
    """

    cfg = config.get("capc", {})
    enabled = bool(cfg.get("enabled", False))
    reference = next(model.parameters())
    if not enabled:
        zero = reference.sum() * 0.0
        return CAPCResult(
            raw_loss=zero,
            weighted_loss=zero,
            metrics={"capc/enabled": zero.detach()},
            artifacts={},
        )

    eps = float(cfg.get("eps", 1.0e-8))
    expected_k = int(cfg.get("views", config.model.num_views))
    raw_views = _to_btkchw(_require(batch, "raw_b_views"), expected_k)
    device = reference.device
    raw_views = raw_views.to(device=device)
    batch_size, time_size, view_count = raw_views.shape[:3]
    if view_count != expected_k or expected_k != int(config.model.num_views):
        raise CAPCBatchInvalid(
            f"CAPC K mismatch: runtime={view_count}, capc={expected_k}, model={int(config.model.num_views)}"
        )
    if time_size < 2:
        raise CAPCBatchInvalid(f"CAPC requires a temporal sequence of length >=2, got {time_size}")

    action = torch.as_tensor(_require(batch, "action"), device=device).float()
    if action.ndim == 2:
        action = action[:, None]
    if tuple(action.shape[:2]) != (batch_size, time_size):
        raise CAPCBatchInvalid(
            f"CAPC action leading shape must be {(batch_size, time_size)}, got {tuple(action.shape)}"
        )
    is_first = _bt_bool(_require(batch, "is_first"), (batch_size, time_size), device)
    integrity = _integrity_and_valid_mask(
        batch,
        batch_size=batch_size,
        time_size=time_size,
        expected_k=expected_k,
        action=action,
        is_first=is_first,
        device=device,
    )
    valid = integrity["valid_transition_mask"]
    valid_count = valid.float().sum()
    if int(valid_count.detach().cpu()) <= 0:
        raise CAPCBatchInvalid("CAPC batch has no valid t->t+1 transitions")

    # [B,T,K,C,H,W] -> [B*K,T,C,H,W].  Keeping each view as a complete
    # sequence is essential; flattening B,T,K together would destroy RSSM
    # recurrence and background-stream identity.
    flat_views = raw_views.permute(0, 2, 1, 3, 4, 5).reshape(
        batch_size * view_count, time_size, *raw_views.shape[3:]
    )
    flat_views = preprocess_observation(flat_views, name="capc_raw_b_views")
    flat_action = action[:, None].expand(-1, view_count, -1, -1).reshape(
        batch_size * view_count, time_size, action.shape[-1]
    )
    flat_first = is_first[:, None].expand(-1, view_count, -1).reshape(
        batch_size * view_count, time_size
    )

    stem = model.stem(flat_views)
    embed = model.encoder(stem)
    initial = model.rssm.initial(batch_size * view_count)
    stoch, deter, post_logit = model.rssm.observe(embed, flat_action, initial, flat_first)
    # Use the native prior network without drawing an unused stochastic prior
    # sample.  deter[:,j] already equals f(h_{j-1}, z_{j-1}, a_{j-1}) under
    # the replay incoming-action contract.
    prior_logit = model.rssm._img_net(deter)
    post_prob = _distribution_probabilities(model, post_logit)
    prior_prob = _distribution_probabilities(model, prior_logit)

    def unflatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch_size, view_count, time_size, *value.shape[2:]).permute(
            0, 2, 1, *range(3, value.ndim + 1)
        ).contiguous()

    post_prob = unflatten(post_prob)  # [B,T,K,G,C]
    prior_prob = unflatten(prior_prob)
    deter_k = unflatten(deter)  # [B,T,K,D]
    stoch_k = unflatten(stoch)  # [B,T,K,G,C]
    embed_k = unflatten(embed)  # [B,T,K,E]

    post_next = post_prob[:, 1:]
    prior_next = prior_prob[:, 1:]
    raw_loss, consensus = canonical_capc_objective(
        post_next,
        prior_next,
        valid,
        eps=eps,
    )
    if consensus.requires_grad:
        raise RuntimeError("CAPC consensus target must be stop-gradient")
    weighted_loss = raw_loss * float(cfg.get("weight", 0.0))

    metrics = {
            "capc/enabled": torch.ones((), device=device),
            "capc/raw": raw_loss.detach(),
            "capc/weighted": weighted_loss.detach(),
            "capc/weight": torch.tensor(float(cfg.get("weight", 0.0)), device=device),
            "capc/target_stopgrad": torch.tensor(float(not consensus.requires_grad), device=device),
            "capc/valid_transition_rate": valid.float().mean(),
            "capc/valid_transition_count": valid_count,
            "capc/K": torch.tensor(float(view_count), device=device),
            "capc/groups": torch.tensor(float(prior_next.shape[-2]), device=device),
            "capc/categories": torch.tensor(float(prior_next.shape[-1]), device=device),
            "capc/batch_integrity_pass": torch.ones((), device=device),
            "capc/q_bar_requires_grad": torch.zeros((), device=device),
        }
    metrics.update({f"capc/integrity_{key}": value for key, value in integrity.items() if key != "valid_transition_mask"})
    artifacts = {
        "q_bar": consensus,
        "posterior_prob": post_next,
        "prior_prob": prior_next,
        "valid_transition_mask": valid,
        "posterior_logit": unflatten(post_logit)[:, 1:],
        "prior_logit": unflatten(prior_logit)[:, 1:],
    }
    return CAPCResult(raw_loss, weighted_loss, metrics, artifacts)


def _distribution_probabilities(model, logits: torch.Tensor) -> torch.Tensor:
    distribution = model.rssm.get_dist(logits)
    base = getattr(distribution, "base_dist", distribution)
    probabilities = getattr(base, "probs", None)
    if probabilities is None:
        probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.float()


def _integrity_and_valid_mask(
    batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
    time_size: int,
    expected_k: int,
    action: torch.Tensor,
    is_first: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    shape = (batch_size, time_size)
    sequence_valid = _bt_bool(_require(batch, "sequence_valid"), shape, device)
    paired_valid = _bt_bool(_require(batch, "paired_view_valid"), shape, device)
    multi_view_valid = _bt_bool(_require(batch, "multi_view_valid"), shape, device)
    vlm_views_valid = _bt_bool(_require(batch, "vlm_views_valid"), shape, device)
    view_valid = paired_valid & multi_view_valid & vlm_views_valid
    valid = sequence_valid[:, :-1] & sequence_valid[:, 1:] & view_valid[:, :-1] & view_valid[:, 1:]
    valid &= ~is_first[:, 1:]
    checks: dict[str, torch.Tensor] = {
        "sequence_valid_fraction": sequence_valid.float().mean(),
        "paired_view_valid_fraction": paired_valid.float().mean(),
        "multi_view_valid_fraction": multi_view_valid.float().mean(),
        "vlm_views_valid_fraction": vlm_views_valid.float().mean(),
        "kview_valid_fraction": view_valid.float().mean(),
    }

    def bt(name: str, *, required: bool = True) -> torch.Tensor | None:
        value = batch.get(name)
        if value is None:
            if required:
                raise CAPCBatchInvalid(f"CAPC integrity requires batch[{name!r}]")
            return None
        tensor = torch.as_tensor(value, device=device)
        if tuple(tensor.shape) != shape:
            raise CAPCBatchInvalid(
                f"CAPC metadata {name!r} must have exact shape {shape}, got {tuple(tensor.shape)}"
            )
        return tensor

    env_id = bt("env_id")
    episode_id = bt("episode_id")
    obs_index = bt("observation_index")
    step_id = bt("step_id")
    same_env = env_id[:, 1:] == env_id[:, :-1]
    same_episode = episode_id[:, 1:] == episode_id[:, :-1]
    consecutive = obs_index[:, 1:] == obs_index[:, :-1] + 1
    consecutive_step = step_id[:, 1:] == step_id[:, :-1] + 1
    checks["same_env_fraction"] = same_env.float().mean()
    checks["same_episode_fraction"] = same_episode.float().mean()
    checks["consecutive_step_fraction"] = consecutive.float().mean()
    checks["consecutive_physical_step_fraction"] = consecutive_step.float().mean()
    structural_rows = sequence_valid[:, :-1] & sequence_valid[:, 1:]
    structural_alignment = same_env & same_episode & consecutive & consecutive_step
    if bool((structural_rows & ~structural_alignment).any()):
        raise CAPCBatchInvalid(
            "CAPC batch crosses env/episode or contains non-consecutive observation/step indices"
        )
    valid &= structural_alignment
    incoming_index, outgoing_index = bt("incoming_action_index"), bt("outgoing_action_index")
    aligned_index = incoming_index[:, 1:] == outgoing_index[:, :-1]
    checks["action_index_alignment_fraction"] = aligned_index.float().mean()
    if bool((structural_rows & ~aligned_index).any()):
        raise CAPCBatchInvalid("CAPC incoming_action_index does not match outgoing_action_index")
    valid &= aligned_index

    incoming_action = torch.as_tensor(_require(batch, "incoming_action"), device=device).float()
    outgoing_action = torch.as_tensor(_require(batch, "outgoing_action"), device=device).float()
    if tuple(incoming_action.shape) != tuple(action.shape) or tuple(outgoing_action.shape) != tuple(action.shape):
        raise CAPCBatchInvalid(
            "CAPC incoming/outgoing action tensors must exactly match action shape "
            f"{tuple(action.shape)}, got {tuple(incoming_action.shape)}/{tuple(outgoing_action.shape)}"
        )
    stored_action_aligned = torch.isclose(action, incoming_action, rtol=0.0, atol=1.0e-6).all(dim=-1)
    transition_action_aligned = torch.isclose(
        action[:, 1:], outgoing_action[:, :-1], rtol=0.0, atol=1.0e-6
    ).all(dim=-1)
    checks["stored_incoming_action_fraction"] = stored_action_aligned.float().mean()
    checks["action_value_alignment_fraction"] = transition_action_aligned.float().mean()
    if not bool(stored_action_aligned.all()):
        raise CAPCBatchInvalid("CAPC batch action differs from replay incoming_action")
    if bool((structural_rows & ~transition_action_aligned).any()):
        raise CAPCBatchInvalid("CAPC action_t does not match replay outgoing_action_{t-1}")
    if bool((is_first & (action.abs().amax(dim=-1) > 1.0e-6)).any()):
        raise CAPCBatchInvalid("CAPC is_first row contains a nonzero incoming action")
    valid &= transition_action_aligned

    counterfactual_k = bt("counterfactual_K")
    if not bool((counterfactual_k == int(expected_k)).all()):
        raise CAPCBatchInvalid(
            f"CAPC counterfactual_K must be exactly {expected_k} on every row"
        )
    checks["constant_k_fraction"] = torch.ones((), device=device)

    def btk(name: str) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            raise CAPCBatchInvalid(f"CAPC integrity requires batch[{name!r}]")
        tensor = torch.as_tensor(value, device=device)
        expected_shape = (batch_size, time_size, expected_k)
        if tuple(tensor.shape) != expected_shape:
            raise CAPCBatchInvalid(
                f"CAPC metadata {name!r} must have exact shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.long()

    def require_view_match(
        name: str,
        shared: torch.Tensor,
        *,
        metric: str,
        message: str,
    ) -> torch.Tensor:
        per_view = btk(name)
        expected = shared.long()[..., None].expand_as(per_view)
        matched = per_view == expected
        checks[metric] = matched.all(dim=-1).float().mean()
        if not bool(matched.all()):
            raise CAPCBatchInvalid(message)
        return per_view

    # These identities are deterministically derived from the immutable replay
    # row metadata by ReplayBuffer._stack_batch.  Requiring them here makes a
    # shuffled view, duplicate DAVIS stream, or non-shared physical rendering a
    # hard error instead of silently training on a corrupted CAPC target.
    view_slot = btk("capc_view_slot_id")
    expected_slots = torch.arange(expected_k, device=device).view(1, 1, expected_k)
    slot_order_ok = (view_slot == expected_slots).all(dim=-1)
    checks["view_slot_order_fraction"] = slot_order_ok.float().mean()
    if not bool(slot_order_ok.all()):
        raise CAPCBatchInvalid("CAPC K-view slot order is permuted or contains an unknown view")

    background_id = btk("capc_background_id")
    sorted_background = background_id.sort(dim=-1).values
    background_unique = (sorted_background[..., 1:] != sorted_background[..., :-1]).all(dim=-1)
    checks["unique_background_fraction"] = background_unique.float().mean()
    if not bool(background_unique.all()):
        raise CAPCBatchInvalid("CAPC K views must use K distinct background identities")

    physics_state_id = btk("capc_physics_state_id")
    physics_shared = (physics_state_id == physics_state_id[..., :1]).all(dim=-1)
    checks["shared_physics_fraction"] = physics_shared.float().mean()
    if not bool(physics_shared.all()):
        raise CAPCBatchInvalid("CAPC K views do not share one physical-state identity")
    row_physics_state_id = bt("capc_row_physics_state_id")
    physics_matches_row = physics_state_id == row_physics_state_id.long()[..., None]
    checks["physics_matches_row_fraction"] = physics_matches_row.all(dim=-1).float().mean()
    if not bool(physics_matches_row.all()):
        raise CAPCBatchInvalid("CAPC K-view physical-state identity differs from replay row")

    camera_id = require_view_match(
        "capc_camera_id",
        bt("camera_id"),
        metric="shared_camera_fraction",
        message="CAPC K views do not share the replay camera identity",
    )
    del camera_id
    for name, shared, metric, message in (
        (
            "capc_env_id",
            env_id,
            "shared_view_env_fraction",
            "CAPC K-view env identity differs from replay row",
        ),
        (
            "capc_episode_id",
            episode_id,
            "shared_view_episode_fraction",
            "CAPC K-view episode identity differs from replay row",
        ),
        (
            "capc_observation_index",
            obs_index,
            "shared_view_observation_fraction",
            "CAPC K-view observation identity differs from replay row",
        ),
        (
            "capc_step_id",
            step_id,
            "shared_view_step_fraction",
            "CAPC K-view step identity differs from replay row",
        ),
        (
            "capc_incoming_action_index",
            incoming_index,
            "shared_view_incoming_action_index_fraction",
            "CAPC K-view incoming-action index differs from replay row",
        ),
        (
            "capc_outgoing_action_index",
            outgoing_index,
            "shared_view_outgoing_action_index_fraction",
            "CAPC K-view outgoing-action index differs from replay row",
        ),
    ):
        require_view_match(name, shared, metric=metric, message=message)

    is_last = bt("is_last").bool()
    is_terminal = bt("is_terminal").bool()
    for name, shared, metric, message in (
        (
            "capc_is_first",
            is_first,
            "shared_view_is_first_fraction",
            "CAPC K-view is_first boundary differs from replay row",
        ),
        (
            "capc_is_last",
            is_last,
            "shared_view_is_last_fraction",
            "CAPC K-view is_last boundary differs from replay row",
        ),
        (
            "capc_is_terminal",
            is_terminal,
            "shared_view_is_terminal_fraction",
            "CAPC K-view terminal boundary differs from replay row",
        ),
        (
            "capc_paired_view_valid",
            paired_valid,
            "shared_view_paired_valid_fraction",
            "CAPC K-view paired-valid flag differs from replay row",
        ),
        (
            "capc_multi_view_valid",
            multi_view_valid,
            "shared_view_multi_valid_fraction",
            "CAPC K-view multi-valid flag differs from replay row",
        ),
        (
            "capc_vlm_views_valid",
            vlm_views_valid,
            "shared_view_vlm_valid_fraction",
            "CAPC K-view VLM-valid flag differs from replay row",
        ),
    ):
        require_view_match(name, shared, metric=metric, message=message)

    def btk_action(name: str) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            raise CAPCBatchInvalid(f"CAPC integrity requires batch[{name!r}]")
        tensor = torch.as_tensor(value, device=device).float()
        expected_shape = (batch_size, time_size, expected_k, action.shape[-1])
        if tuple(tensor.shape) != expected_shape:
            raise CAPCBatchInvalid(
                f"CAPC metadata {name!r} must have exact shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    for name, shared, metric, message in (
        (
            "capc_incoming_action",
            incoming_action,
            "shared_view_incoming_action_fraction",
            "CAPC K-view incoming action differs from replay row",
        ),
        (
            "capc_outgoing_action",
            outgoing_action,
            "shared_view_outgoing_action_fraction",
            "CAPC K-view outgoing action differs from replay row",
        ),
    ):
        per_view = btk_action(name)
        expected = shared[:, :, None, :].expand_as(per_view)
        matched = torch.isclose(per_view, expected, rtol=0.0, atol=1.0e-6).all(dim=-1)
        checks[metric] = matched.all(dim=-1).float().mean()
        if not bool(matched.all()):
            raise CAPCBatchInvalid(message)

    checks["structural_valid_fraction"] = valid.float().mean()
    return {"valid_transition_mask": valid, **checks}


def _transition_identity(
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return the physical identity for each incoming transition ``j-1 -> j``."""

    expected = (valid.shape[0], valid.shape[1] + 1)
    fields = []
    for name in ("env_id", "episode_id", "observation_index"):
        value = torch.as_tensor(_require(batch, name), device=device).reshape(expected)
        fields.append(value[:, 1:].long())
    return torch.stack(fields, dim=-1)


def _different_transition_index(
    transition_identity: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Choose a valid, explicitly different physical transition for each row.

    Returned values are global flattened ``[B,T]`` indices.  Invalid rows and
    batches with no distinct physical identity use ``-1``.  Every nonnegative
    target is guaranteed to have a different ``(env, episode,
    observation_index)`` identity from its query.
    """

    flat_identity = transition_identity.reshape(-1, transition_identity.shape[-1])
    flat_valid = valid.reshape(-1)
    positions = flat_valid.nonzero(as_tuple=False).flatten()
    result = torch.full(
        (flat_valid.numel(),), -1, device=valid.device, dtype=torch.long
    )
    if positions.numel() < 2:
        return result.reshape_as(valid)
    identities = flat_identity[positions]
    if torch.unique(identities, dim=0).shape[0] < 2:
        return result.reshape_as(valid)
    different = ~(identities[:, None, :] == identities[None, :, :]).all(dim=-1)
    if not bool(different.any(dim=1).all()):
        return result.reshape_as(valid)
    target = different.to(torch.int64).argmax(dim=1)
    result[positions] = positions[target]
    return result.reshape_as(valid)


def _geometry_metrics(
    post: torch.Tensor,
    prior: torch.Tensor,
    consensus: torch.Tensor,
    valid: torch.Tensor,
    deter: torch.Tensor,
    stoch: torch.Tensor,
    embed: torch.Tensor,
    *,
    transition_identity: torch.Tensor,
    different_transition_index: torch.Tensor,
    eps: float,
    retrieval: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = prior.device
    same_prior = _pairwise_view_jsd(prior, valid, eps)
    same_post = _pairwise_view_jsd(post, valid, eps)
    prior_consensus = prior.mean(dim=2)
    post_consensus = post.mean(dim=2)
    different_prior = _different_transition_jsd(
        prior, valid, different_transition_index, eps
    )
    different_post = _different_transition_jsd(
        post, valid, different_transition_index, eps
    )
    different_pair_available = valid & (different_transition_index >= 0)
    different_pair_fraction = (
        different_pair_available.float().sum() / valid.float().sum().clamp_min(1.0)
    )
    rho_prior = same_prior / different_prior.clamp_min(eps)
    rho_post = same_post / different_post.clamp_min(eps)
    posterior_entropy = _masked_mean(_entropy(post, eps).mean(dim=(2, 3)), valid)
    prior_entropy = _masked_mean(_entropy(prior, eps).mean(dim=(2, 3)), valid)
    consensus_entropy = _masked_mean(_entropy(consensus, eps).mean(dim=2), valid)
    h_l2, h_cosine = _continuous_cross_view(deter, valid, eps)
    encoder_l2, encoder_cosine = _continuous_cross_view(embed, valid, eps)
    belief = torch.cat((deter, stoch.flatten(start_dim=-2)), dim=-1)
    belief_l2, belief_cosine = _continuous_cross_view(belief, valid, eps)
    categories = stoch.argmax(dim=-1)
    disagreements = []
    for left, right in combinations(range(categories.shape[2]), 2):
        disagreements.append((categories[:, :, left] != categories[:, :, right]).float().mean(dim=-1))
    z_disagreement = _masked_mean(torch.stack(disagreements, dim=0).mean(dim=0), valid)
    posterior_utilization, posterior_occupancy = _categorical_utilization(post, valid, eps)
    prior_utilization, prior_occupancy = _categorical_utilization(prior, valid, eps)
    post_prior_kl = (
        post.clamp_min(eps)
        * (post.clamp_min(eps).log() - prior.clamp_min(eps).log())
    ).sum(dim=-1).mean(dim=(2, 3))
    posterior_prior_kl = _masked_mean(post_prior_kl, valid)
    valid_deter = deter[valid]
    h_mean = valid_deter.mean() if valid_deter.numel() else torch.zeros((), device=device)
    h_std = valid_deter.std(unbiased=False) if valid_deter.numel() else torch.zeros((), device=device)
    h_rms = valid_deter.square().mean().sqrt() if valid_deter.numel() else torch.zeros((), device=device)
    categories_count = int(post.shape[-1])
    complete_collapse = (
        (different_pair_fraction > 0.0)
        & (different_prior <= 1.0e-8)
        & (different_post <= 1.0e-8)
        & (posterior_utilization["occupied_fraction"] <= (1.0 / categories_count + 0.01))
    ).float()
    metrics = {
        "capc/same_transition_prior_jsd": same_prior,
        "capc/different_transition_prior_jsd": different_prior,
        "capc/different_transition_pair_fraction": different_pair_fraction,
        "capc/rho_transition_prior": rho_prior,
        "capc/same_transition_posterior_jsd": same_post,
        "capc/different_transition_posterior_jsd": different_post,
        "capc/rho_transition_posterior": rho_post,
        "capc/prior_entropy": prior_entropy,
        "capc/posterior_entropy": posterior_entropy,
        "capc/consensus_entropy": consensus_entropy,
        "capc/consensus_entropy_gain": consensus_entropy - posterior_entropy,
        "capc/transition_separation_margin": different_prior - same_prior,
        "capc/transition_separation_ratio": different_prior / same_prior.clamp_min(eps),
        "capc/h_cross_view_normalized_l2": h_l2,
        "capc/h_cross_view_cosine": h_cosine,
        "capc/encoder_cross_view_normalized_l2": encoder_l2,
        "capc/encoder_cross_view_cosine": encoder_cosine,
        "capc/z_group_disagreement": z_disagreement,
        "capc/belief_hz_cross_view_normalized_l2": belief_l2,
        "capc/belief_hz_cross_view_cosine": belief_cosine,
        "capc/posterior_prior_kl": posterior_prior_kl,
        "capc/h_mean": h_mean,
        "capc/h_std": h_std,
        "capc/h_rms": h_rms,
        "capc/posterior_max_probability_mean": posterior_utilization["max_probability_mean"],
        "capc/prior_max_probability_mean": prior_utilization["max_probability_mean"],
        "capc/posterior_group_entropy_mean": posterior_utilization["group_entropy_mean"],
        "capc/prior_group_entropy_mean": prior_utilization["group_entropy_mean"],
        "capc/posterior_effective_categories_mean": posterior_utilization["effective_categories_mean"],
        "capc/prior_effective_categories_mean": prior_utilization["effective_categories_mean"],
        "capc/posterior_occupied_category_fraction": posterior_utilization["occupied_fraction"],
        "capc/prior_occupied_category_fraction": prior_utilization["occupied_fraction"],
        "capc/posterior_dead_category_fraction": 1.0 - posterior_utilization["occupied_fraction"],
        "capc/prior_dead_category_fraction": 1.0 - prior_utilization["occupied_fraction"],
        "capc/posterior_collapsed_group_fraction": posterior_utilization["collapsed_group_fraction"],
        "capc/prior_collapsed_group_fraction": prior_utilization["collapsed_group_fraction"],
        "capc/posterior_max_entropy_group_fraction": posterior_utilization["max_entropy_group_fraction"],
        "capc/prior_max_entropy_group_fraction": prior_utilization["max_entropy_group_fraction"],
        "capc/categorical_complete_collapse": complete_collapse,
    }
    artifacts: dict[str, torch.Tensor] = {
        "prior_consensus": prior_consensus,
        "posterior_consensus": post_consensus,
        "posterior_group_occupancy": posterior_occupancy,
        "prior_group_occupancy": prior_occupancy,
    }
    if retrieval:
        recall1, recall5 = _transition_retrieval(
            prior, valid, transition_identity, eps
        )
        metrics["capc/transition_recall_at_1"] = recall1
        metrics["capc/transition_recall_at_5"] = recall5
    return metrics, artifacts


def _pairwise_view_jsd(prob: torch.Tensor, valid: torch.Tensor, eps: float) -> torch.Tensor:
    values = []
    for left, right in combinations(range(prob.shape[2]), 2):
        values.append(_jsd(prob[:, :, left], prob[:, :, right], eps).mean(dim=-1))
    return _masked_mean(torch.stack(values, dim=0).mean(dim=0), valid)


def _different_transition_jsd(
    probability: torch.Tensor,
    valid: torch.Tensor,
    different_transition_index: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    flat = probability.reshape(-1, *probability.shape[2:])
    positions = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    if positions.numel() < 2:
        return torch.zeros((), device=probability.device)
    targets = different_transition_index.reshape(-1)[positions]
    available = targets >= 0
    if not bool(available.any()):
        return torch.zeros((), device=probability.device)
    positions = positions[available]
    targets = targets[available]
    if bool((targets == positions).any()):
        raise RuntimeError("different-transition geometry received a self-pair")
    return _jsd(flat[positions], flat[targets], eps).mean()


def _transition_retrieval(
    prob: torch.Tensor,
    valid: torch.Tensor,
    transition_identity: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # View 0 queries and view 1 candidates avoid a quadratic K-expanded pool.
    # Replay sampling is with replacement, so one physical transition may
    # legitimately occur more than once.  All candidates with the same
    # physical identity are positives; de-duplicating them would make Recall@K
    # depend on which duplicate happened to survive.
    positions = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    flat_identity = transition_identity.reshape(-1, transition_identity.shape[-1])
    count = min(64, int(positions.numel()))
    if count < 2:
        zero = torch.zeros((), device=prob.device)
        return zero, zero
    sample = torch.linspace(
        0, positions.numel() - 1, steps=count, device=prob.device
    ).long()
    selected = positions[sample]
    flat = prob.reshape(-1, *prob.shape[2:])
    query = flat[selected, 0]
    candidate = flat[selected, 1]
    identity = flat_identity[selected]
    distance = _jsd(query[:, None], candidate[None], eps).mean(dim=-1)
    order = distance.argsort(dim=1)
    positives = (identity[:, None, :] == identity[None, :, :]).all(dim=-1)
    ranked_positive = positives.gather(1, order)
    recall1 = ranked_positive[:, :1].any(dim=1).float().mean()
    recall5 = ranked_positive[:, : min(5, count)].any(dim=1).float().mean()
    return recall1, recall5


def _continuous_cross_view(value: torch.Tensor, valid: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    l2, cosine = [], []
    for left, right in combinations(range(value.shape[2]), 2):
        a, b = value[:, :, left].float(), value[:, :, right].float()
        scale = 0.5 * (a.square().mean(dim=-1).sqrt() + b.square().mean(dim=-1).sqrt())
        l2.append((a - b).square().mean(dim=-1).sqrt() / scale.clamp_min(eps))
        cosine.append(torch.nn.functional.cosine_similarity(a, b, dim=-1))
    return (
        _masked_mean(torch.stack(l2, dim=0).mean(dim=0), valid),
        _masked_mean(torch.stack(cosine, dim=0).mean(dim=0), valid),
    )


def _jsd(left: torch.Tensor, right: torch.Tensor, eps: float) -> torch.Tensor:
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def _entropy(prob: torch.Tensor, eps: float) -> torch.Tensor:
    return -(prob * prob.clamp_min(eps).log()).sum(dim=-1)


def _categorical_utilization(
    probability: torch.Tensor,
    valid: torch.Tensor,
    eps: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = probability[valid]  # [N,K,G,C]
    groups = int(probability.shape[-2])
    categories = int(probability.shape[-1])
    device = probability.device
    if not rows.numel():
        zero = torch.zeros((), device=device)
        occupancy = torch.zeros(groups, categories, device=device)
        return {
            "max_probability_mean": zero,
            "group_entropy_mean": zero,
            "effective_categories_mean": zero,
            "occupied_fraction": zero,
            "collapsed_group_fraction": zero,
            "max_entropy_group_fraction": zero,
        }, occupancy
    entropy = _entropy(rows, eps)  # [N,K,G]
    predictive_group_entropy = entropy.mean(dim=(0, 1))
    assignments = rows.argmax(dim=-1)  # [N,K,G]
    occupancy_rows = []
    for group in range(groups):
        counts = torch.bincount(
            assignments[:, :, group].reshape(-1), minlength=categories
        ).float()
        occupancy_rows.append(counts / counts.sum().clamp_min(1.0))
    occupancy = torch.stack(occupancy_rows, dim=0)
    occupancy_entropy = _entropy(occupancy, eps)
    occupied_fraction = (occupancy > 0).float().mean()
    maximum_entropy = math.log(float(categories))
    return {
        "max_probability_mean": rows.max(dim=-1).values.mean(),
        "group_entropy_mean": predictive_group_entropy.mean(),
        "effective_categories_mean": occupancy_entropy.exp().mean(),
        "occupied_fraction": occupied_fraction,
        "collapsed_group_fraction": (occupancy_entropy < 0.10).float().mean(),
        "max_entropy_group_fraction": (
            predictive_group_entropy > 0.98 * maximum_entropy
        ).float().mean(),
    }, occupancy


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def _to_btkchw(value: torch.Tensor, expected_k: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 6:
        raise CAPCBatchInvalid(f"raw_b_views must be rank 6 [B,T,K,...], got {tuple(tensor.shape)}")
    if tensor.shape[2] != expected_k:
        raise CAPCBatchInvalid(f"raw_b_views K must be {expected_k}, got {tensor.shape[2]}")
    if tensor.shape[3] == 3:
        return tensor
    if tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 2, 5, 3, 4).contiguous()
    raise CAPCBatchInvalid(f"raw_b_views must contain RGB channels, got {tuple(tensor.shape)}")


def _bt_bool(
    value: torch.Tensor | None,
    shape: tuple[int, int],
    device: torch.device,
    *,
    default: bool = False,
) -> torch.Tensor:
    if value is None:
        return torch.full(shape, bool(default), dtype=torch.bool, device=device)
    tensor = torch.as_tensor(value, device=device).bool()
    if tensor.ndim == 1:
        tensor = tensor[:, None].expand(shape)
    return tensor.reshape(shape)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise CAPCBatchInvalid(f"CAPC requires batch[{key!r}]")
    return mapping[key]
