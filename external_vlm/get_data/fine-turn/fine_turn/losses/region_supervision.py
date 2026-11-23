from __future__ import annotations


def region_loss(logits, mask, *, dice_weight: float = 0.5):
    import torch
    import torch.nn.functional as F

    if mask is None:
        return torch.zeros((), device=logits.device), {}
    target = downsample_mask_to_logits(mask, logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = torch.sum(prob * target)
    denom = torch.sum(prob) + torch.sum(target)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    return bce + float(dice_weight) * dice, {"region_bce": bce.detach(), "region_dice": dice.detach()}


def downsample_mask_to_logits(mask, logits):
    import torch.nn.functional as F

    target = mask
    if target.ndim == 5:
        target = target.squeeze(2) if target.shape[2] == 1 else target.mean(dim=2)
    if logits.ndim == 4:
        size = logits.shape[-3:]
        target = F.interpolate(target, size=size[-2:], mode="nearest")
        target = target.mean(dim=1, keepdim=False)
        if target.ndim == 3:
            target = target.unsqueeze(1).expand(-1, size[0], -1, -1)
        return target
    target = F.interpolate(target, size=(logits.shape[-1],), mode="nearest")
    return target.squeeze(1)

