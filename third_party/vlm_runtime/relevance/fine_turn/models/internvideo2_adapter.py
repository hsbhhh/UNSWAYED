from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class VideoEncoderOutput:
    global_embedding: Any
    patch_tokens: Any | None
    temporal_tokens: Any | None
    token_grid_shape: tuple[int, int, int] | None


class MockVideoBackbone(torch.nn.Module):
    def __init__(self, *, embed_dim: int = 32, token_grid_shape: tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.token_grid_shape = tuple(int(x) for x in token_grid_shape)
        self.proj = torch.nn.Linear(3, self.embed_dim)

    def forward(self, video):
        bsz, time, channels, height, width = video.shape
        pooled = video.mean(dim=(1, 3, 4))
        global_embedding = F.normalize(self.proj(pooled), dim=-1)
        t_grid, h_grid, w_grid = self.token_grid_shape
        tokens = F.adaptive_avg_pool3d(video.permute(0, 2, 1, 3, 4), (t_grid, h_grid, w_grid))
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(bsz, t_grid * h_grid * w_grid, channels)
        patch_tokens = self.proj(tokens)
        temporal_tokens = patch_tokens.reshape(bsz, t_grid, h_grid * w_grid, self.embed_dim).mean(dim=2)
        return VideoEncoderOutput(global_embedding, patch_tokens, temporal_tokens, self.token_grid_shape)


class InternVideo2VideoEncoder(torch.nn.Module):
    def __init__(self, *, get_data_root: Path, resources, config: dict[str, Any]):
        super().__init__()
        self.get_data_root = Path(get_data_root)
        self.resources = resources
        self.config = config
        self.adapter = None
        self.feature_dim = None
        self.embed_dim = None
        self.backend_report: dict[str, Any] = {}

    def load(self) -> None:
        if self.resources.internvideo2_root is None or not self.resources.has_internvideo2_checkpoint():
            raise FileNotFoundError("SKIPPED: InternVideo2 checkpoint not available")
        if str(self.get_data_root) not in sys.path:
            sys.path.insert(0, str(self.get_data_root))
        from adapters.internvideo2_adapter import InternVideo2Adapter

        model_cfg = self.config.get("model", {})
        self.adapter = InternVideo2Adapter(
            internvideo2_root=self.resources.internvideo2_root,
            model_weights_root=self.resources.weights_root,
            checkpoint=self.resources.clip_checkpoint,
            config=None,
            device=str(self.config.get("device", {}).get("device", "cpu")),
            num_frames=int(model_cfg.get("num_frames") or 8),
            image_size=int(model_cfg.get("image_size") or 224),
            dtype=str(model_cfg.get("precision", "fp32")),
            attention_backend=str(model_cfg.get("attention_backend", "naive")),
            disable_text_lora=bool(model_cfg.get("disable_text_lora", False)),
        )
        self.adapter.load()
        self.backend_report = dict(self.adapter.backend_report)
        self.feature_dim = self.adapter.feature_dim
        self.embed_dim = self.adapter.feature_dim
        self.model = self.adapter.model
        for param in self.adapter.model.parameters():
            param.requires_grad_(False)

    def encode_numpy(self, frames):
        if self.adapter is None:
            self.load()
        return self.adapter.encode_video(frames)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.adapter is not None and not any(param.requires_grad for param in self.adapter.model.parameters()):
            self.adapter.model.eval()
        return self

    def forward(self, video):
        if self.adapter is None:
            self.load()
        device = torch.device(self.adapter.config.device)
        tensor = video.to(device=device)
        return_tokens = bool(self.config.get("model", {}).get("return_patch_tokens", True))
        train_backbone = self.training and any(param.requires_grad for param in self.adapter.model.parameters())
        grad_context = torch.enable_grad() if train_backbone else torch.no_grad()
        with grad_context:
            feat, patch, temporal, grid = self._encode_tensor(tensor, capture_tokens=return_tokens)
            feat = F.normalize(_as_global_feature(feat).float(), dim=-1)
            if patch is not None:
                patch = patch.float()
            if temporal is not None:
                temporal = temporal.float()
        if not train_backbone:
            feat = feat.detach().clone()
            if patch is not None:
                patch = patch.detach().clone()
            if temporal is not None:
                temporal = temporal.detach().clone()
        return VideoEncoderOutput(feat, patch_tokens=patch, temporal_tokens=temporal, token_grid_shape=grid)

    def _encode_tensor(self, tensor, *, capture_tokens: bool):
        captured: dict[str, torch.Tensor] = {}
        handle = None
        if capture_tokens:
            clip_projector = _module_by_path(self.adapter.model, "vision_encoder.clip_projector")
            if clip_projector is not None:
                def _capture_input(_module, inputs, _output):
                    if inputs and torch.is_tensor(inputs[0]):
                        captured["tokens"] = inputs[0]

                handle = clip_projector.register_forward_hook(_capture_input)
        try:
            encoded = self.adapter.model.encode_vision(tensor, test=True)
        finally:
            if handle is not None:
                handle.remove()

        global_feat, token_seq = self._split_encoded_output(encoded, captured.get("tokens"))
        if token_seq is None:
            return global_feat, None, None, None

        token_seq, grid = self._prepare_patch_tokens(token_seq, use_image=tensor.shape[1] == 1)
        temporal = _temporal_tokens(token_seq, grid)
        return global_feat, token_seq, temporal, grid

    def _split_encoded_output(self, encoded, captured_tokens):
        tensors = [item for item in encoded if torch.is_tensor(item)] if isinstance(encoded, tuple) else [encoded]
        tensors = [item for item in tensors if torch.is_tensor(item)]
        if not tensors:
            raise RuntimeError("InternVideo2 encode_vision did not return any tensors.")

        token_seq = _first_token_sequence(tensors)
        global_feat = _last_global_tensor(tensors)
        if global_feat is None:
            global_source = token_seq if token_seq is not None else captured_tokens
            global_feat = _pool_global_from_tokens(global_source)
        if captured_tokens is not None and bool(self.config.get("model", {}).get("prefer_hook_tokens", True)):
            token_seq = captured_tokens
        if token_seq is encoded and getattr(encoded, "ndim", 0) <= 2:
            token_seq = None
        return global_feat, token_seq

    def _prepare_patch_tokens(self, tokens, *, use_image: bool):
        grid = _infer_token_grid(self.adapter.config, self.adapter.model, use_image=use_image)
        tokens, grid = _strip_cls_token(tokens, grid)
        if self.feature_dim is not None and tokens.shape[-1] != int(self.feature_dim):
            tokens = self._project_clip_tokens(tokens)
        if self.feature_dim is not None and tokens.shape[-1] != int(self.feature_dim):
            raise RuntimeError(
                "InternVideo2 patch token dim does not match the fine-tuned relevance head: "
                f"tokens={tokens.shape[-1]}, head={self.feature_dim}. "
                "The adapter could not project the tokens through vision_encoder.clip_projector; "
                "inspect scripts/inspect_model.py before training/exporting q_sem."
            )
        return tokens, grid

    def _project_clip_tokens(self, tokens):
        projector = _module_by_path(self.adapter.model, "vision_encoder.clip_projector")
        cross_attn = getattr(projector, "cross_attn", None) if projector is not None else None
        value_proj = getattr(cross_attn, "v", None)
        out_proj = getattr(cross_attn, "proj", None)
        if value_proj is None or out_proj is None:
            return tokens
        norm = getattr(projector, "norm1_v", None)
        x = norm(tokens) if norm is not None else tokens
        x = value_proj(x)
        x = out_proj(x)
        drop = getattr(cross_attn, "proj_drop", None)
        return drop(x) if drop is not None else x


def build_video_encoder(resources, config: dict[str, Any]):
    model_cfg = config.get("model", {})
    if bool(model_cfg.get("mock", False)):
        return MockVideoBackbone(
            embed_dim=int(model_cfg.get("embed_dim", 32)),
            token_grid_shape=tuple(model_cfg.get("patch_grid", (2, 4, 4))),
        )
    encoder = InternVideo2VideoEncoder(get_data_root=resources.get_data_root, resources=resources, config=config)
    encoder.load()
    return encoder


def _module_by_path(module, path: str):
    current = module
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _as_global_feature(tensor):
    if tensor.ndim == 3:
        if tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor.mean(dim=1)
    return tensor


def _first_token_sequence(tensors):
    for tensor in tensors:
        if tensor.ndim == 3 and tensor.shape[1] > 1:
            return tensor
    return None


def _last_global_tensor(tensors):
    for tensor in reversed(tensors):
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor[:, 0]
    return None


def _pool_global_from_tokens(tokens):
    if tokens is None:
        raise RuntimeError("InternVideo2 returned neither a global feature nor patch tokens.")
    if tokens.ndim == 3 and tokens.shape[1] > 1:
        return tokens[:, 1:].mean(dim=1)
    return _as_global_feature(tokens)


def _infer_token_grid(config, model, *, use_image: bool) -> tuple[int, int, int] | None:
    vision = getattr(getattr(config, "model", None), "vision_encoder", None)
    encoder = getattr(model, "vision_encoder", None)
    patch_embed = getattr(encoder, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", None)
    if grid_size is not None and len(tuple(grid_size)) == 3:
        t_grid, h_grid, w_grid = (int(x) for x in tuple(grid_size))
        return (1 if use_image else t_grid, h_grid, w_grid)
    if vision is None:
        return None
    frames = int(getattr(vision, "num_frames", 1))
    tubelet = max(1, int(getattr(vision, "tubelet_size", 1)))
    image = int(getattr(vision, "img_size", 224))
    patch = max(1, int(getattr(vision, "patch_size", 14)))
    return (1 if use_image else max(1, frames // tubelet), max(1, image // patch), max(1, image // patch))


def _strip_cls_token(tokens, grid):
    if grid is None:
        return tokens, _factor_grid(tokens.shape[1])
    expected = int(grid[0]) * int(grid[1]) * int(grid[2])
    if tokens.shape[1] == expected + 1:
        return tokens[:, 1:], grid
    if tokens.shape[1] == expected:
        return tokens, grid
    return tokens, _factor_grid(tokens.shape[1], preferred_t=int(grid[0]))


def _factor_grid(length: int, preferred_t: int | None = None) -> tuple[int, int, int]:
    length = int(length)
    preferred = [preferred_t] if preferred_t else []
    candidates = [x for x in preferred + [1, 2, 4, 8, 16] if x and length % x == 0]
    for t_grid in candidates:
        spatial = length // int(t_grid)
        side = int(round(spatial ** 0.5))
        if side * side == spatial:
            return int(t_grid), side, side
    return 1, 1, length


def _temporal_tokens(tokens, grid):
    if grid is None:
        return tokens.mean(dim=1, keepdim=True)
    t_grid, h_grid, w_grid = (int(x) for x in grid)
    if tokens.shape[1] != t_grid * h_grid * w_grid:
        return tokens.mean(dim=1, keepdim=True)
    return tokens.reshape(tokens.shape[0], t_grid, h_grid * w_grid, tokens.shape[-1]).mean(dim=2)
