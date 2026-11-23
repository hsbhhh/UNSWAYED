from __future__ import annotations

import torch


def topk_patch_metrics(logits: torch.Tensor, target: torch.Tensor, *, ks: tuple[int, ...] = (4, 8, 16, 32)) -> dict[str, float]:
    scores = torch.sigmoid(logits.detach().float()).reshape(logits.shape[0], -1)
    target = target.detach().float().to(scores.device).reshape(logits.shape[0], -1)
    out: dict[str, float] = {}
    ranks = []
    for item_scores, item_target in zip(scores, target):
        fg = item_target > 0.5
        if int(fg.sum()) == 0:
            continue
        order = torch.argsort(item_scores, descending=True)
        fg_positions = torch.where(fg[order])[0]
        if fg_positions.numel() > 0:
            ranks.append(float(fg_positions.min().cpu()) + 1.0)
    out["best_foreground_rank"] = float(sum(ranks) / len(ranks)) if ranks else float("nan")
    for k in ks:
        k = min(int(k), scores.shape[1])
        top = torch.topk(scores, k=k, dim=1).indices
        hits = torch.gather(target, 1, top)
        fg_total = target.sum(dim=1).clamp_min(1.0)
        out[f"recall@{k}"] = float((hits.sum(dim=1) / fg_total).mean().cpu())
        out[f"precision@{k}"] = float(hits.mean().cpu())
        out[f"background@{k}"] = float((1.0 - hits).mean().cpu())
    return out

