from __future__ import annotations


def background_consistency_loss(anchor_z, paired_z, anchor_map=None, paired_map=None, *, map_weight: float = 0.0):
    import torch
    import torch.nn.functional as F

    global_loss = 1.0 - F.cosine_similarity(anchor_z, paired_z, dim=-1).mean()
    map_loss = torch.zeros((), device=anchor_z.device)
    if anchor_map is not None and paired_map is not None:
        map_loss = torch.mean(torch.abs(anchor_map - paired_map))
    return global_loss + float(map_weight) * map_loss, {"background_global": global_loss.detach(), "background_map": map_loss.detach()}

