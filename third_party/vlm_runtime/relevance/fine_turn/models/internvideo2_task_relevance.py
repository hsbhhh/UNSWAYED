from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass
class TaskRelevanceOutput:
    global_embedding: torch.Tensor
    patch_tokens: torch.Tensor
    native_logits: torch.Tensor
    native_scores: torch.Tensor
    q_sem_logits: torch.Tensor
    q_sem: torch.Tensor
    q_sem_grid: torch.Tensor
    token_grid_shape: tuple[int, int, int]
    pooled_grid_shape: tuple[int, int]


class NativePatchRelevanceHead(torch.nn.Module):
    def __init__(self, token_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim or token_dim)
        self.token_proj = torch.nn.Linear(int(token_dim), hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(float(dropout))
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.token_proj(patch_tokens)
        hidden = self.norm(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        return self.out(hidden).squeeze(-1)


class InternVideo2TaskRelevanceModel(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        token_dim: int,
        hidden_dim: int | None = None,
        target_grid: tuple[int, int] = (8, 8),
        pooling: str = "logsumexp",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.relevance_head = NativePatchRelevanceHead(token_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.target_grid = tuple(int(x) for x in target_grid)
        self.pooling = str(pooling)

    def forward_video(self, video: torch.Tensor) -> TaskRelevanceOutput:
        encoded = self.backbone(video)
        if encoded.patch_tokens is None:
            raise RuntimeError("InternVideo2 backbone did not return patch_tokens; q_sem patch training cannot continue.")
        patch_tokens = encoded.patch_tokens
        grid = _resolve_grid(encoded.token_grid_shape, patch_tokens.shape[1])
        token_logits = self.relevance_head(patch_tokens)
        native_logits = token_logits.reshape(token_logits.shape[0], *grid)
        q_sem_grid = pool_logits_to_grid(native_logits, self.target_grid, mode=self.pooling)
        q_sem_logits = q_sem_grid.reshape(q_sem_grid.shape[0], q_sem_grid.shape[1], -1)
        return TaskRelevanceOutput(
            global_embedding=encoded.global_embedding,
            patch_tokens=patch_tokens,
            native_logits=native_logits,
            native_scores=torch.sigmoid(native_logits),
            q_sem_logits=q_sem_logits,
            q_sem=torch.sigmoid(q_sem_logits),
            q_sem_grid=q_sem_grid,
            token_grid_shape=grid,
            pooled_grid_shape=self.target_grid,
        )

    def forward_pair(self, batch: Mapping[str, torch.Tensor]) -> tuple[TaskRelevanceOutput, TaskRelevanceOutput]:
        return self.forward_video(batch["clean_video"]), self.forward_video(batch["davis_video"])

    def forward(self, video: torch.Tensor) -> TaskRelevanceOutput:
        return self.forward_video(video)


def pool_logits_to_grid(logits: torch.Tensor, grid_size: tuple[int, int], *, mode: str = "logsumexp") -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,T,H,W], got {tuple(logits.shape)}.")
    grid_h, grid_w = (int(x) for x in grid_size)
    bsz, time, height, width = logits.shape
    flat = logits.reshape(bsz * time, 1, height, width)
    if mode == "avg":
        pooled = F.adaptive_avg_pool2d(flat, (grid_h, grid_w))
    elif mode == "max":
        pooled = F.adaptive_max_pool2d(flat, (grid_h, grid_w))
    elif mode == "logsumexp" and height % grid_h == 0 and width % grid_w == 0:
        block_h = height // grid_h
        block_w = width // grid_w
        blocks = flat.reshape(bsz * time, 1, grid_h, block_h, grid_w, block_w)
        pooled = torch.logsumexp(blocks, dim=(3, 5)) - torch.log(torch.tensor(float(block_h * block_w), device=logits.device))
    elif mode == "logsumexp":
        # The real InternVideo2 224/f14 path is 16x16 -> 8x8, so this fallback is only for mocks.
        pooled = F.adaptive_avg_pool2d(flat, (grid_h, grid_w))
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")
    return pooled.reshape(bsz, time, grid_h, grid_w)


def freeze_backbone(model: InternVideo2TaskRelevanceModel) -> None:
    for param in model.backbone.parameters():
        param.requires_grad_(False)
    for param in model.relevance_head.parameters():
        param.requires_grad_(True)


def trainable_parameter_groups(model: InternVideo2TaskRelevanceModel, *, head_lr: float, lora_lr: float | None = None, weight_decay: float = 0.05):
    head_params = [param for param in model.relevance_head.parameters() if param.requires_grad]
    groups = [{"params": head_params, "lr": float(head_lr), "weight_decay": float(weight_decay)}]
    other_params = [param for name, param in model.named_parameters() if param.requires_grad and not name.startswith("relevance_head.")]
    if other_params:
        groups.append({"params": other_params, "lr": float(lora_lr or head_lr), "weight_decay": float(weight_decay)})
    return groups


def compact_task_relevance_state_dict(model: InternVideo2TaskRelevanceModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        key: value
        for key, value in state.items()
        if key.startswith("relevance_head.") or ".lora_a." in key or ".lora_b." in key
    }


def _resolve_grid(grid: tuple[int, int, int] | None, token_count: int) -> tuple[int, int, int]:
    if grid is not None:
        grid = tuple(int(x) for x in grid)
        if int(grid[0]) * int(grid[1]) * int(grid[2]) == int(token_count):
            return grid
    return _factor_grid(int(token_count), preferred_t=None if grid is None else int(grid[0]))


def _factor_grid(length: int, preferred_t: int | None = None) -> tuple[int, int, int]:
    candidates = [preferred_t] if preferred_t else []
    candidates.extend([8, 4, 2, 1, 16])
    for t_grid in [int(x) for x in candidates if x and length % int(x) == 0]:
        spatial = length // t_grid
        side = int(round(spatial ** 0.5))
        if side * side == spatial:
            return t_grid, side, side
    return 1, 1, int(length)

