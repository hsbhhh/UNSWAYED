from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from fine_turn.losses.total_loss import TotalLoss
from fine_turn.models.projection_head import ProjectionHead
from fine_turn.models.relation_head import RelationHead
from fine_turn.models.relevance_head import RelevanceHead


class FineTuneHeadsModel(torch.nn.Module):
    def __init__(self, backbone, *, embed_dim: int, task_text: str = ""):
        super().__init__()
        self.backbone = backbone
        self.projection_head = ProjectionHead(embed_dim)
        self.relevance_head = RelevanceHead(embed_dim)
        self.relation_head = RelationHead(embed_dim)
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))
        self.task_text = task_text
        self.register_buffer("task_text_embedding", torch.zeros(embed_dim), persistent=False)

    def encode(self, video):
        out = self.backbone(video)
        projected = self.projection_head(out.global_embedding)
        relation = self.relation_head(projected, out.temporal_tokens)
        q_sem = None
        if out.patch_tokens is not None:
            text = self.task_text_embedding.expand(video.shape[0], -1)
            q_sem = self.relevance_head(out.patch_tokens, text, token_grid_shape=out.token_grid_shape)
        return {
            "global_embedding": projected,
            "relation_embedding": relation,
            "q_sem": q_sem,
            "encoder_output": out,
        }

    def forward_pair(self, batch: Mapping):
        anchor = self.encode(batch["anchor"])
        paired = self.encode(batch["paired"])
        return anchor, paired


def verify_semantic_token_path(model, video, *, device: str = "cpu") -> dict:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        encoded = model.encode(video.to(device))
    if was_training:
        model.train()
    encoder_output = encoded["encoder_output"]
    if encoder_output.patch_tokens is None or encoded["q_sem"] is None:
        raise RuntimeError(
            "The real InternVideo2 path did not return patch tokens/q_sem. "
            "Do not train or export semantic teacher labels until this is fixed."
        )
    return {
        "global_embedding": list(encoded["global_embedding"].shape),
        "patch_tokens": list(encoder_output.patch_tokens.shape),
        "temporal_tokens": None if encoder_output.temporal_tokens is None else list(encoder_output.temporal_tokens.shape),
        "token_grid_shape": None if encoder_output.token_grid_shape is None else list(encoder_output.token_grid_shape),
        "q_sem": list(encoded["q_sem"].shape),
    }


@dataclass
class TrainStats:
    loss: float
    steps: int


def train_one_epoch(model, loader, optimizer, loss_fn: TotalLoss, *, pair_type: str, device: str = "cpu") -> TrainStats:
    model.train()
    total = 0.0
    steps = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        anchor, paired = model.forward_pair(batch)
        if pair_type == "background":
            loss, _ = loss_fn.background(
                anchor["global_embedding"],
                paired["global_embedding"],
                anchor["q_sem"],
                paired["q_sem"],
                batch.get("task_mask"),
            )
        else:
            future_effect = _future_effect_tensor(batch.get("samples"), device=device)
            loss, _ = loss_fn.interaction(anchor["relation_embedding"], paired["relation_embedding"], future_effect)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return TrainStats(total / max(1, steps), steps)


def _move_batch(batch: Mapping, device: str):
    out = dict(batch)
    for key in ("anchor", "paired", "task_mask"):
        value = out.get(key)
        if hasattr(value, "to"):
            out[key] = value.to(device)
    return out


def _future_effect_tensor(samples, *, device: str):
    if not samples:
        return None
    values = [sample.future_effect for sample in samples]
    if any(value is None for value in values):
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)
