from __future__ import annotations


def interaction_distance(anchor, paired):
    import torch.nn.functional as F

    return 1.0 - F.cosine_similarity(anchor, paired, dim=-1)


def interaction_sensitivity_score(distances):
    return distances.mean()

