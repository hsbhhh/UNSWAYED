from __future__ import annotations


def background_distance(anchor, paired):
    import torch
    import torch.nn.functional as F

    return 1.0 - F.cosine_similarity(anchor, paired, dim=-1)


def background_invariance_score(distances):
    return 1.0 - distances.mean()

