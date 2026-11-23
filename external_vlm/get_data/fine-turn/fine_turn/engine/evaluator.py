from __future__ import annotations


def evaluate_pair_distances(model, loader, *, pair_type: str, device: str = "cpu") -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    distances = []
    with torch.inference_mode():
        for batch in loader:
            for key in ("anchor", "paired"):
                batch[key] = batch[key].to(device)
            anchor, paired = model.forward_pair(batch)
            key = "relation_embedding" if pair_type == "interaction" else "global_embedding"
            distance = 1.0 - F.cosine_similarity(anchor[key], paired[key], dim=-1)
            distances.append(distance.detach().cpu())
    if not distances:
        return {"count": 0}
    values = torch.cat(distances)
    return {"count": int(values.numel()), "mean_distance": float(values.mean()), "max_distance": float(values.max())}


def evaluate_pair_distances_detailed(model, loader, *, pair_type: str, device: str = "cpu") -> list[dict]:
    import torch
    import torch.nn.functional as F

    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            for key in ("anchor", "paired"):
                batch[key] = batch[key].to(device)
            anchor, paired = model.forward_pair(batch)
            embedding_key = "relation_embedding" if pair_type == "interaction" else "global_embedding"
            distances = 1.0 - F.cosine_similarity(anchor[embedding_key], paired[embedding_key], dim=-1)
            for sample, distance in zip(batch["samples"], distances.detach().cpu().tolist()):
                rows.append(
                    {
                        "pair_id": sample.pair_id,
                        "pair_type": sample.pair_type,
                        "match_key": sample.physics_trajectory_id or sample.episode_id,
                        "episode_id": sample.episode_id,
                        "physics_trajectory_id": sample.physics_trajectory_id,
                        "intervention_type": sample.intervention_type,
                        "future_effect": sample.future_effect,
                        "distance": float(distance),
                    }
                )
    return rows


def summarize_background(rows: list[dict]) -> dict[str, float | int | None]:
    distances = [row["distance"] for row in rows]
    summary = _summary(distances)
    if summary["count"]:
        summary["bis"] = 1.0 - float(summary["mean_distance"])
    else:
        summary["bis"] = None
    return summary


def summarize_interaction(rows: list[dict]) -> dict[str, float | int | None]:
    summary = _summary([row["distance"] for row in rows])
    summary["iss"] = summary["mean_distance"]
    return summary


def compute_inda(background_rows: list[dict], interaction_rows: list[dict]) -> dict[str, float | int | bool | None]:
    bg_by_key: dict[str, float] = {}
    for row in background_rows:
        key = row.get("match_key")
        if key and key not in bg_by_key:
            bg_by_key[key] = float(row["distance"])
    comparisons = []
    for row in interaction_rows:
        key = row.get("match_key")
        if key in bg_by_key:
            comparisons.append(float(row["distance"]) > bg_by_key[key])
    if not comparisons:
        return {"count": 0, "inda": None, "strict": False}
    return {
        "count": len(comparisons),
        "inda": float(sum(comparisons) / len(comparisons)),
        "strict": True,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_distance": None, "min_distance": None, "max_distance": None}
    return {
        "count": len(values),
        "mean_distance": float(sum(values) / len(values)),
        "min_distance": float(min(values)),
        "max_distance": float(max(values)),
    }
