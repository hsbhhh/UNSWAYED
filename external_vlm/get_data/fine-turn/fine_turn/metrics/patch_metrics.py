from __future__ import annotations

import torch


def patch_binary_metrics(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> dict[str, float]:
    prob = torch.sigmoid(logits.detach().float()).reshape(-1)
    target = target.detach().float().to(prob.device).reshape(-1)
    pred = (prob >= float(threshold)).float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    tn = ((1.0 - pred) * (1.0 - target)).sum()
    dice = (2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-6)).item()
    iou = (tp / (tp + fp + fn).clamp_min(1e-6)).item()
    return {
        "dice": float(dice),
        "iou": float(iou),
        "foreground_score": _masked_mean(prob, target > 0.5),
        "background_score": _masked_mean(prob, target <= 0.5),
        "auroc": float(_binary_auroc(prob, target)),
        "auprc": float(_binary_auprc(prob, target)),
        "tp": float(tp.cpu()),
        "fp": float(fp.cpu()),
        "fn": float(fn.cpu()),
        "tn": float(tn.cpu()),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(values[mask].mean().cpu())


def _binary_auroc(prob: torch.Tensor, target: torch.Tensor) -> float:
    pos = int((target > 0.5).sum())
    neg = int((target <= 0.5).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(prob)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, prob.numel() + 1, device=prob.device, dtype=torch.float32)
    pos_ranks = ranks[target > 0.5].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / max(1.0, float(pos * neg))
    return float(auc.cpu())


def _binary_auprc(prob: torch.Tensor, target: torch.Tensor) -> float:
    pos = int((target > 0.5).sum())
    if pos == 0:
        return float("nan")
    order = torch.argsort(prob, descending=True)
    y = target[order]
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    recall_step = y / max(1, pos)
    return float((precision * recall_step).sum().cpu())

