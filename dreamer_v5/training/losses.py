from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..capc import compute_capc_loss
from ..counterfactual.losses import counterfactual_vgc_loss, task_normalized_medoid_msir
from ..semantic.semantic_lowrank_gaussian import msir_loss, semantic_feature_consensus_loss
from ..semantic.semantic_weighting import semantic_weighting_from_batch


@dataclass
class WorldLossComponents:
    primary: torch.Tensor
    cvgc: torch.Tensor
    msir: torch.Tensor
    native: torch.Tensor
    capc_raw: torch.Tensor
    capc: torch.Tensor
    total: torch.Tensor
    metrics: dict[str, torch.Tensor]
    output: object
    formal_counterfactual: bool
    capc_artifacts: dict[str, torch.Tensor]


def world_model_losses(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, metrics, _ = compute_world_model_losses(model, batch)
    return total, metrics


def compute_world_model_losses(
    model,
    batch: dict[str, torch.Tensor],
    *,
    output=None,
    update_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], object]:
    components = compute_world_model_loss_components(
        model,
        batch,
        output=output,
        update_index=update_index,
    )
    return components.total, components.metrics, components.output


def compute_world_model_loss_components(
    model,
    batch: dict[str, torch.Tensor],
    *,
    output=None,
    update_index: int | None = None,
) -> WorldLossComponents:
    output = model.forward_training_batch(batch) if output is None else output
    cfg = model.config.model.world_model_loss
    dyn_loss, rep_loss, dyn_raw, rep_raw = model.rssm.kl_loss_details(
        output.rssm_post_logit,
        output.rssm_prior_logit,
        free=float(cfg.kl_free),
    )
    loss_terms: dict[str, torch.Tensor] = {
        "dreamer/dyn_kl": dyn_loss.mean() * float(cfg.dyn_weight),
        "dreamer/rep_kl": rep_loss.mean() * float(cfg.rep_weight),
    }
    metrics = dict(output.metrics)
    metrics["dreamer/dyn_kl_raw"] = dyn_raw.detach().mean()
    metrics["dreamer/rep_kl_raw"] = rep_raw.detach().mean()
    metrics["dreamer/dyn_kl_clipped"] = dyn_loss.detach().mean()
    metrics["dreamer/rep_kl_clipped"] = rep_loss.detach().mean()
    if "reward" in batch:
        reward = torch.as_tensor(batch["reward"], device=output.reward_pred.device).float().reshape_as(output.reward_pred)
        reward_loss = F.mse_loss(output.reward_pred, reward)
        loss_terms["dreamer/reward"] = reward_loss * float(cfg.reward_weight)
        metrics["dreamer/reward_loss"] = reward_loss.detach()
    if "is_terminal" in batch:
        terminal = torch.as_tensor(batch["is_terminal"], device=output.continue_logit.device).float()
        continue_target = 1.0 - terminal.reshape_as(output.continue_logit)
        continue_loss = F.binary_cross_entropy_with_logits(output.continue_logit, continue_target)
        loss_terms["dreamer/continue"] = continue_loss * float(cfg.continue_weight)
        metrics["dreamer/continue_loss"] = continue_loss.detach()
    primary = sum(loss_terms.values())
    metrics["dreamer/world_core_loss"] = primary.detach()
    post_prob = torch.softmax(output.rssm_post_logit.detach().float(), dim=-1)
    prior_prob = torch.softmax(output.rssm_prior_logit.detach().float(), dim=-1)
    metrics["dreamer/posterior_entropy"] = (-(post_prob * post_prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    metrics["dreamer/prior_entropy"] = (-(prior_prob * prior_prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()

    formal_counterfactual = bool(model.config.get("counterfactual", {}).get("enabled", False))
    if formal_counterfactual:
        relevance_cfg = model.config.counterfactual.task_relevance
        task_sensitivity = model.task_relevance(
            primary,
            output.online_spatial_embedding,
            mode=str(relevance_cfg.mode),
            update_period=int(relevance_cfg.update_period),
            ema_momentum=float(relevance_cfg.ema_momentum),
        )
        cvgc, cvgc_metrics, artifacts = counterfactual_vgc_loss(
            output,
            model.config,
            batch,
            task_sensitivity,
        )
        msir, msir_metrics, msir_artifacts = task_normalized_medoid_msir(
            output,
            model.config,
            batch,
            artifacts,
        )
        output.counterfactual_artifacts = artifacts | msir_artifacts
        metrics.update(cvgc_metrics)
        metrics.update(msir_metrics)
        metrics["counterfactual/task_relevance_detached"] = torch.tensor(
            float(not task_sensitivity.requires_grad), device=primary.device
        )
        metrics["counterfactual/K"] = torch.tensor(float(model.config.counterfactual.views), device=primary.device)
        metrics["counterfactual/semantic_precision_enabled"] = torch.zeros((), device=primary.device)
        metrics["counterfactual/tau_used_in_loss"] = torch.zeros((), device=primary.device)
        metrics["counterfactual/decoder_role_diagnostic_only"] = torch.tensor(
            float(not bool(model.config.model.msir.training_enabled)), device=primary.device
        )
        total = primary + cvgc + msir
        loss_terms["counterfactual/cvgc"] = cvgc
        loss_terms["counterfactual/msir"] = msir
    else:
        semantic_weights = None
        weighting_cfg = model.config.model.get("semantic_weighting", {"enabled": False})
        if bool(weighting_cfg.get("enabled", False)):
            semantic_weights = semantic_weighting_from_batch(
                batch,
                model.config,
                device=output.vgc_patch_prediction.device,
            )
        vgc_terms, vgc_metrics = semantic_feature_consensus_loss(
            output,
            model.config,
            batch,
            semantic_weights=semantic_weights,
        )
        msir_terms, msir_metrics = msir_loss(
            output,
            model.config,
            batch,
            semantic_weights=semantic_weights,
        )
        loss_terms.update(vgc_terms)
        loss_terms.update(msir_terms)
        metrics.update(vgc_metrics)
        metrics.update(msir_metrics)
        if semantic_weights is not None:
            metrics["semantic_weight/vgc_msir_shared_tensor"] = torch.tensor(
                1.0, device=primary.device
            )
            metrics["k_view/configured_num_views"] = torch.tensor(
                float(model.config.model.num_views), device=primary.device
            )
            metrics["k_view/actual_score_views"] = torch.tensor(
                float(semantic_weights.raw_scores.shape[2]), device=primary.device
            )
            if "vgc/view_count" not in metrics or "msir/view_count" not in metrics:
                raise RuntimeError(
                    "active K-view VGC/MSIR did not report their runtime view counts"
                )
            metrics["k_view/actual_feature_views"] = metrics["vgc/view_count"]
            metrics["k_view/actual_msir_views"] = metrics["msir/view_count"]
        cvgc = sum(vgc_terms.values()) if vgc_terms else primary * 0.0
        msir = sum(msir_terms.values()) if msir_terms else primary * 0.0
        total = sum(loss_terms.values())
    native = total
    capc_result = compute_capc_loss(
        model,
        batch,
        model.config,
        update_index=update_index,
    )
    if bool(model.config.get("capc", {}).get("enabled", False)):
        loss_terms["capc/predictive_consistency"] = capc_result.weighted_loss
        total = native + capc_result.weighted_loss
    metrics.update(capc_result.metrics)
    metrics["loss/native_world_total"] = native.detach()
    metrics["loss/capc_fraction_of_total"] = (
        capc_result.weighted_loss.detach() / total.detach().abs().clamp_min(1.0e-12)
    )
    metrics["loss/total"] = total.detach()
    for key, value in loss_terms.items():
        metrics[f"loss_term/{key}"] = value.detach()
    return WorldLossComponents(
        primary=primary,
        cvgc=cvgc,
        msir=msir,
        native=native,
        capc_raw=capc_result.raw_loss,
        capc=capc_result.weighted_loss,
        total=total,
        metrics=metrics,
        output=output,
        formal_counterfactual=formal_counterfactual,
        capc_artifacts=capc_result.artifacts,
    )
