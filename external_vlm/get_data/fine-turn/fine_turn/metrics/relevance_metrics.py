from __future__ import annotations


def dice_score(prob, target, *, threshold: float = 0.5):
    pred = (prob > float(threshold)).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)


def iou_score(prob, target, *, threshold: float = 0.5):
    pred = (prob > float(threshold)).float()
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    return (intersection + 1.0) / (union + 1.0)

