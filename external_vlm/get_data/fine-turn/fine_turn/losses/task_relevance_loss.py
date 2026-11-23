from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TaskRelevanceLossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 0.5
    object_weight: float = 0.5
    consistency_weight: float = 0.5
    hard_background_weight: float = 0.5
    temporal_weight: float = 0.05
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    hard_negative_quantile: float = 0.85
    agent_coverage_weight: float = 0.0
    target_coverage_weight: float = 0.0
    balance_weight: float = 0.0
    agent_margin: float = 0.75
    target_margin: float = 0.75


class TaskRelevanceLoss(torch.nn.Module):
    def __init__(self, config: TaskRelevanceLossConfig | Mapping | None = None):
        super().__init__()
        if config is None:
            config = TaskRelevanceLossConfig()
        if isinstance(config, Mapping):
            config = TaskRelevanceLossConfig(**{key: value for key, value in config.items() if key in TaskRelevanceLossConfig.__annotations__})
        self.config = config

    def forward(self, clean, davis, targets: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target = targets["task"].to(clean.q_sem_grid.device).float()
        clean_logits = clean.q_sem_grid.float()
        davis_logits = davis.q_sem_grid.float()
        losses: dict[str, torch.Tensor] = {}
        losses["clean_focal_bce"] = balanced_focal_bce_with_logits(clean_logits, target, self.config.focal_alpha, self.config.focal_gamma)
        losses["davis_focal_bce"] = balanced_focal_bce_with_logits(davis_logits, target, self.config.focal_alpha, self.config.focal_gamma)
        losses["clean_dice"] = soft_dice_loss(torch.sigmoid(clean_logits), target)
        losses["davis_dice"] = soft_dice_loss(torch.sigmoid(davis_logits), target)
        losses["hard_background"] = hard_background_suppression(clean_logits, davis_logits, target, self.config.hard_negative_quantile)
        losses["clean_davis_consistency"] = F.mse_loss(torch.sigmoid(clean_logits), torch.sigmoid(davis_logits))
        losses["temporal"] = temporal_smoothness(clean_logits) + temporal_smoothness(davis_logits)
        object_loss = object_coverage_loss(clean_logits, davis_logits, targets)
        if object_loss is not None:
            losses["object_coverage"] = object_loss
        total = (
            self.config.bce_weight * (losses["clean_focal_bce"] + losses["davis_focal_bce"]) * 0.5
            + self.config.dice_weight * (losses["clean_dice"] + losses["davis_dice"]) * 0.5
            + self.config.hard_background_weight * losses["hard_background"]
            + self.config.consistency_weight * losses["clean_davis_consistency"]
            + self.config.temporal_weight * losses["temporal"]
        )
        if "object_coverage" in losses:
            total = total + self.config.object_weight * losses["object_coverage"]
        component = agent_target_losses(
            clean_logits,
            davis_logits,
            targets,
            agent_margin=self.config.agent_margin,
            target_margin=self.config.target_margin,
        )
        if component is not None:
            losses.update(component)
            total = (
                total
                + self.config.agent_coverage_weight * losses["agent_coverage"]
                + self.config.target_coverage_weight * losses["target_coverage"]
                + self.config.balance_weight * losses["agent_target_balance"]
            )
        metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
        probability = torch.sigmoid(clean_logits)
        background = target < 0.05
        metrics["background_score"] = float(probability[background].mean().detach().cpu()) if int(background.sum()) else 0.0
        if component is not None:
            metrics["agent_score"] = float(component["agent_score"].detach().cpu())
            metrics["target_score"] = float(component["target_score"].detach().cpu())
            metrics["target_agent_ratio"] = float(component["target_agent_ratio"].detach().cpu())
        metrics["loss"] = float(total.detach().cpu())
        return total, metrics


def agent_target_losses(
    clean_logits: torch.Tensor,
    davis_logits: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    *,
    agent_margin: float,
    target_margin: float,
) -> dict[str, torch.Tensor] | None:
    agent = targets.get("agent")
    target = targets.get("target_component")
    if agent is None or target is None or agent.numel() == 0 or target.numel() == 0:
        return None
    agent = agent.to(clean_logits.device).float()
    target = target.to(clean_logits.device).float()
    agent_clean, valid_agent = _masked_mean_per_sample(torch.sigmoid(clean_logits), agent)
    agent_davis, _ = _masked_mean_per_sample(torch.sigmoid(davis_logits), agent)
    target_clean, valid_target = _masked_mean_per_sample(torch.sigmoid(clean_logits), target)
    target_davis, _ = _masked_mean_per_sample(torch.sigmoid(davis_logits), target)
    valid = valid_agent & valid_target
    if int(valid.sum()) == 0:
        return None
    agent_views = torch.cat((agent_clean[valid], agent_davis[valid]))
    target_views = torch.cat((target_clean[valid], target_davis[valid]))
    return {
        "agent_coverage": F.relu(float(agent_margin) - agent_views).pow(2).mean(),
        "target_coverage": F.relu(float(target_margin) - target_views).pow(2).mean(),
        "agent_target_balance": (agent_views - target_views).pow(2).mean(),
        "agent_score": agent_clean[valid].mean(),
        "target_score": target_clean[valid].mean(),
        "target_agent_ratio": target_clean[valid].mean() / agent_clean[valid].mean().clamp_min(1e-6),
    }


def _masked_mean_per_sample(probability: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, probability.ndim))
    denominator = mask.sum(dim=dims)
    score = (probability * mask).sum(dim=dims) / denominator.clamp_min(1e-6)
    return score, denominator > 0


def balanced_focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    target = target.to(logits.device).float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
    focal = alpha_t * (1.0 - pt).clamp_min(1e-4).pow(float(gamma)) * bce
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    weights = target * (target.numel() / (2.0 * pos)) + (1.0 - target) * (target.numel() / (2.0 * neg))
    return (focal * weights).mean()


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.to(prob.device).float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def hard_background_suppression(clean_logits: torch.Tensor, davis_logits: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    neg = target.to(clean_logits.device) < 0.05
    if int(neg.sum()) == 0:
        return clean_logits.new_zeros(())
    scores = torch.maximum(clean_logits, davis_logits)
    negative_scores = scores[neg]
    cutoff = torch.quantile(negative_scores.detach().float(), float(quantile)).to(scores.device)
    hard = neg & (scores >= cutoff)
    if int(hard.sum()) == 0:
        hard = neg
    return F.softplus(scores[hard]).mean()


def temporal_smoothness(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] <= 1:
        return logits.new_zeros(())
    return (logits[:, 1:] - logits[:, :-1]).abs().mean()


def object_coverage_loss(clean_logits: torch.Tensor, davis_logits: torch.Tensor, targets: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    valid = targets.get("object_valid")
    if valid is None or int(valid.sum()) == 0:
        return None
    valid = valid.to(clean_logits.device).bool()
    losses = []
    for key in ("ball", "cup", "rope"):
        mask = targets.get(key)
        if mask is None or mask.numel() == 0:
            continue
        mask = mask.to(clean_logits.device).float()[valid]
        if float(mask.sum().detach().cpu()) <= 0.0:
            continue
        for logits in (clean_logits[valid], davis_logits[valid]):
            prob = torch.sigmoid(logits)
            numerator = (prob * mask).sum(dim=(1, 2, 3))
            denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            losses.append(1.0 - numerator / denom)
    if not losses:
        return None
    return torch.cat([item.reshape(-1) for item in losses]).mean()
