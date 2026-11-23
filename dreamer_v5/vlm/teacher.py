from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..vcfa.grid import resolve_vcfa_grid


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VLMOutput:
    q_sem: torch.Tensor
    raw_grid: torch.Tensor
    token_grid_shape: tuple[int, int]
    source: str
    q_sem_logits: torch.Tensor | None = None
    patch_features: torch.Tensor | None = None
    patch_logits: torch.Tensor | None = None
    patch_probs: torch.Tensor | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    checkpoint_fingerprint: str | None = None
    preprocessing_fingerprint: str | None = None


class MockSemanticTeacher(torch.nn.Module):
    def __init__(self, grid_shape: tuple[int, int] = (8, 8), *, patch_feature_dim: int = 32):
        super().__init__()
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.patch_feature_dim = int(patch_feature_dim)
        self.clip_length = 1
        self.semantic_score_source = "spatial_feature_map"
        self.load_report = {"kind": "mock", "semantic_score_source": self.semantic_score_source}

    def forward_sequence(self, video: torch.Tensor, *, is_first: torch.Tensor | None = None) -> VLMOutput:
        x = _as_bthwc(video).float()
        if x.max() > 2:
            x = x / 255.0
        saliency = (x[..., 0] * 0.45 + x[..., 1] * 0.35 + (1.0 - x[..., 2]) * 0.20).reshape(*x.shape[:2], 1, x.shape[2], x.shape[3])
        grid = F.adaptive_avg_pool2d(saliency.reshape(-1, 1, x.shape[2], x.shape[3]), self.grid_shape).reshape(x.shape[0], x.shape[1], *self.grid_shape)
        grid = _normalize01(grid)
        logits = torch.logit(grid.clamp(1e-4, 1.0 - 1e-4))
        patch_features = _mock_patch_features(grid, self.patch_feature_dim)
        return VLMOutput(
            q_sem=grid.reshape(grid.shape[0], grid.shape[1], -1),
            q_sem_logits=logits.reshape(grid.shape[0], grid.shape[1], -1),
            patch_features=patch_features,
            patch_logits=logits.reshape(grid.shape[0], grid.shape[1], -1),
            patch_probs=grid.reshape(grid.shape[0], grid.shape[1], -1),
            raw_grid=grid,
            token_grid_shape=self.grid_shape,
            source="mock_spatial_feature_map",
            trace={
                "input_shape": list(video.shape),
                "semantic_score_source": "spatial_feature_map",
                "current_frame_output": "current_frame_mock",
                "aligned_logits_shape": list(logits.shape),
                "q_sem_shape": [grid.shape[0], grid.shape[1], grid.shape[2] * grid.shape[3]],
                "patch_features_shape": list(patch_features.shape),
            },
            checkpoint_fingerprint="mock",
            preprocessing_fingerprint="mock",
        )

    def forward(self, video: torch.Tensor) -> VLMOutput:
        return self.forward_sequence(video)

    def forward_latest_clip(self, video: torch.Tensor) -> VLMOutput:
        x = _latest_clip_bthwc(video, self.clip_length)
        out = self.forward_sequence(x)
        return _last_semantic_output(out)


class FinetuneV2InternVideo2SemanticTeacher(torch.nn.Module):
    """Frozen online q_sem teacher backed by ball_in_cup_vlm_finetune_v2 InternVideo2."""

    def __init__(self, resources, config: dict[str, Any]):
        super().__init__()
        self.resources = resources
        self.config = config
        semantic_cfg = config.get("semantic", {})
        tokens_cfg = config.get("tokens", {})
        token_grid = (int(tokens_cfg.get("grid_height", 8)), int(tokens_cfg.get("grid_width", 8)))
        self.grid_shape = resolve_vcfa_grid(config, token_grid)
        self.clip_length = int(semantic_cfg.get("clip_length") or semantic_cfg.get("num_frames") or 8)
        self.image_size = int(semantic_cfg.get("image_size") or config.get("environment", {}).get("vlm_image_size", 224) or 224)
        self.temperature = float(semantic_cfg.get("temperature", 1.0))
        self.inference_dtype = str(semantic_cfg.get("inference_dtype", "bfloat16"))
        self.internvideo2_backend = str(semantic_cfg.get("internvideo2_backend", "naive")).strip().lower()
        if self.internvideo2_backend not in {"naive", "native_flash"}:
            raise ValueError("semantic.internvideo2_backend must be naive or native_flash")
        self.causal_padding = str(semantic_cfg.get("causal_padding", "repeat_first"))
        self.current_frame_output = str(semantic_cfg.get("current_frame_output", "last_temporal_slice"))
        if self.current_frame_output != "last_temporal_slice":
            raise ValueError("finetune_v2 q_sem must use current_frame_output=last_temporal_slice.")
        object.__setattr__(self, "_teacher_model", None)
        self.loaded_checkpoint: Path | None = None
        self.load_report: dict[str, Any] = {}
        self._printed_load_report = False
        self.semantic_score_source = "unknown"

    def load(self):
        if self.model is not None:
            return self
        semantic_cfg = dict(self.config.get("semantic", {}) or {})
        checkpoint = _resolve_checkpoint(self.resources, self.config)
        fine_root = _resolve_fine_turn_root(self.resources, self.config)
        _prepend_sys_path(fine_root.parent)
        _prepend_sys_path(fine_root)
        _evict_module_prefixes(("vlm", "adapters.internvideo2_adapter"))
        from fine_turn.config import load_yaml as load_fine_yaml
        from fine_turn.models.internvideo2_adapter import build_video_encoder
        from fine_turn.models.internvideo2_task_relevance import InternVideo2TaskRelevanceModel
        from fine_turn.models.lora_adapter import LoRAConfig, apply_lora_to_named_linears
        from fine_turn.resources import resolve_resources

        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"finetune_v2 checkpoint must be a mapping: {checkpoint}")
        fine_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        if not fine_cfg:
            cfg_path = fine_root / "configs" / "ball_in_cup_patch_relevance.yaml"
            fine_cfg = load_fine_yaml(cfg_path) if cfg_path.exists() else {}
        fine_cfg = dict(fine_cfg)
        fine_cfg.setdefault("resources", {})
        if isinstance(payload.get("base_model_checkpoints"), Mapping):
            for key, value in payload["base_model_checkpoints"].items():
                if value and not fine_cfg["resources"].get(key):
                    fine_cfg["resources"][key] = value
        base_checkpoint = _resolve_base_checkpoint(self.resources, self.config)
        if base_checkpoint is not None:
            fine_cfg["resources"]["clip_checkpoint"] = str(base_checkpoint)
        fine_cfg.setdefault("device", {})["device"] = str(_semantic_device(self.config))
        fine_cfg.setdefault("model", {})["return_patch_tokens"] = True
        fine_cfg["model"]["attention_backend"] = self.internvideo2_backend
        args = SimpleNamespace(
            config=None,
            dry_run=False,
            get_data_root=fine_root.parent,
            data_root=None,
            background_root=None,
            interaction_root=None,
            internvideo_root=None,
            internvideo2_root=None,
            weights_root=None,
            clip_checkpoint=None,
            stage2_checkpoint=None,
            internvl_checkpoint=None,
            output_root=None,
            resume=None,
        )
        fine_resources = resolve_resources(args, fine_cfg, write_resolved=False)
        backbone = build_video_encoder(fine_resources, fine_cfg)
        backbone_backend_report = dict(getattr(backbone, "backend_report", {}) or {})
        actual_backend = str(backbone_backend_report.get("actual_backend", "naive"))
        if actual_backend != self.internvideo2_backend:
            raise RuntimeError(
                "InternVideo2 backend mismatch after model construction: "
                f"requested={self.internvideo2_backend}, actual={actual_backend}"
            )
        token_dim = int(getattr(backbone, "embed_dim", None) or getattr(backbone, "feature_dim", None) or fine_cfg.get("model", {}).get("embed_dim", 32))
        model_cfg = fine_cfg.get("model", {})
        model = InternVideo2TaskRelevanceModel(
            backbone,
            token_dim=token_dim,
            hidden_dim=int(model_cfg.get("token_hidden_dim") or token_dim),
            target_grid=tuple(model_cfg.get("target_grid", self.grid_shape)),
            pooling=str(model_cfg.get("pooling", "logsumexp")),
            dropout=0.0,
        )
        lora_modules = list(payload.get("lora_applied_modules") or [])
        lora_loaded = False
        if lora_modules:
            lora_cfg = payload.get("config", {}).get("lora", {}) if isinstance(payload.get("config"), Mapping) else fine_cfg.get("lora", {})
            applied = apply_lora_to_named_linears(
                model.backbone,
                lora_modules,
                LoRAConfig(
                    rank=int(lora_cfg.get("rank", 8)),
                    alpha=float(lora_cfg.get("alpha", 16)),
                    dropout=float(lora_cfg.get("dropout", 0.05)),
                ),
            )
            lora_loaded = len(applied) == len(lora_modules)
        state = payload.get("task_relevance")
        if not isinstance(state, Mapping):
            raise RuntimeError(f"finetune_v2 checkpoint has no task_relevance state. top-level keys={list(payload.keys())}")
        model = model.to(_semantic_device(self.config))
        model_state = model.state_dict()
        matched_keys, shape_mismatches = _state_match_report(state, model_state)
        if shape_mismatches and bool(semantic_cfg.get("strict_load", True)):
            raise RuntimeError(f"Shape mismatches while loading finetune_v2 checkpoint: {shape_mismatches[:20]}")
        result = model.load_state_dict(state, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        relevance_missing = [key for key in missing if key.startswith("relevance_head.")]
        if relevance_missing:
            raise RuntimeError(f"Patch relevance head did not load completely: {relevance_missing[:20]}")
        matched_numel = int(sum(int(state[key].numel()) for key in matched_keys if torch.is_tensor(state[key])))
        expected_numel = int(sum(int(value.numel()) for value in model_state.values() if torch.is_tensor(value)))
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        object.__setattr__(self, "_teacher_model", model)
        self.loaded_checkpoint = checkpoint
        self.semantic_score_source = "patch_tokens"
        self.load_report = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_fingerprint": None,
            "checkpoint_type": _checkpoint_type(payload),
            "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
            "base_fingerprint": None,
            "matched_keys": matched_keys[:50],
            "matched_total": len(matched_keys),
            "matched_parameter_count": len(matched_keys),
            "matched_numel": matched_numel,
            "expected_numel": expected_numel,
            "load_coverage": matched_numel / max(expected_numel, 1),
            "missing_keys": missing[:50],
            "unexpected_keys": unexpected[:50],
            "shape_mismatch_keys": shape_mismatches[:50],
            "missing_total": len(missing),
            "unexpected_total": len(unexpected),
            "shape_mismatch_total": len(shape_mismatches),
            "lora_loaded": bool(lora_loaded),
            "relevance_head_loaded": True,
            "output_8x8_head_loaded": tuple(getattr(model, "target_grid", self.grid_shape)) == self.grid_shape,
            "semantic_score_source": self.semantic_score_source,
            "internvideo2_backend_requested": self.internvideo2_backend,
            "internvideo2_backend_actual": actual_backend,
            "internvideo2_backend_report": backbone_backend_report,
            "clip_length": self.clip_length,
            "causal_padding": self.causal_padding,
            "image_size": self.image_size,
            "output_grid": list(self.grid_shape),
            "preprocessing_fingerprint": self.preprocessing_fingerprint(),
            "fine_turn_root": str(fine_root),
            "fine_resources": fine_resources.to_serializable(),
        }
        if bool(semantic_cfg.get("print_load_report", True)) and not self._printed_load_report:
            print("FinetuneV2 InternVideo2 q_sem load report:")
            print(json.dumps(self.load_report, indent=2, ensure_ascii=False, default=str))
            self._printed_load_report = True
        return self

    def forward_sequence(self, video: torch.Tensor, *, is_first: torch.Tensor | None = None) -> VLMOutput:
        self.load()
        x = _as_bthwc(video)
        clips = causal_clips(x, self.clip_length, padding=self.causal_padding, is_first=is_first)
        bsz, time = x.shape[:2]
        flat = clips.reshape(bsz * time, self.clip_length, *x.shape[2:])
        model_input = _preprocess_for_internvideo2(flat, self.image_size).to(_semantic_device(self.config))
        use_amp = _semantic_device(self.config).type == "cuda" and self.inference_dtype in {"bf16", "bfloat16", "float16", "fp16"}
        dtype = torch.bfloat16 if self.inference_dtype in {"bf16", "bfloat16"} else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            out = self.model(model_input)
        if self.semantic_score_source not in {"patch_tokens", "spatial_feature_map"}:
            raise RuntimeError(f"Invalid semantic_score_source={self.semantic_score_source}; refusing global-token q_sem.")
        logits_grid = out.q_sem_grid.detach().float().clone()[:, -1]
        if tuple(logits_grid.shape[-2:]) != self.grid_shape:
            logits_grid = F.interpolate(logits_grid[:, None], size=self.grid_shape, mode="bilinear", align_corners=False).squeeze(1)
        logits_grid = logits_grid / max(self.temperature, 1e-6)
        q_sem = torch.sigmoid(logits_grid).reshape(bsz, time, -1)
        patch_features = _last_temporal_patch_features(
            out.patch_tokens.detach().float().clone(),
            out.token_grid_shape,
            target_grid=self.grid_shape,
        ).reshape(bsz, time, self.grid_shape[0] * self.grid_shape[1], -1)
        trace = {
            "input_shape": list(video.shape),
            "causal_clip_shape": list(clips.shape),
            "model_input_shape": list(model_input.shape),
            "patch_tokens_shape": list(out.patch_tokens.shape),
            "patch_features_shape": list(patch_features.shape),
            "native_temporal_grid": int(out.token_grid_shape[0]),
            "native_spatial_grid": [int(out.token_grid_shape[-2]), int(out.token_grid_shape[-1])],
            "special_token_count": 0,
            "relevance_head_input_shape": list(out.patch_tokens.shape),
            "native_logits_shape": list(out.native_logits.shape),
            "current_frame_logits_shape": list(logits_grid.shape),
            "aligned_8x8_logits_shape": [bsz, time, *self.grid_shape],
            "q_sem_shape": list(q_sem.shape),
            "semantic_score_source": self.semantic_score_source,
            "temporal_selection": "last_temporal_slice",
        }
        return VLMOutput(
            q_sem=q_sem.to(video.device),
            q_sem_logits=logits_grid.reshape(bsz, time, -1).to(video.device),
            patch_features=patch_features.to(video.device),
            patch_logits=logits_grid.reshape(bsz, time, -1).to(video.device),
            patch_probs=q_sem.to(video.device),
            raw_grid=torch.sigmoid(logits_grid).reshape(bsz, time, *self.grid_shape).to(video.device),
            token_grid_shape=self.grid_shape,
            source="finetune_v2_internvideo2_online",
            trace=trace,
            checkpoint_fingerprint=self.load_report.get("checkpoint_fingerprint"),
            preprocessing_fingerprint=self.preprocessing_fingerprint(),
        )

    def forward(self, video: torch.Tensor) -> VLMOutput:
        return self.forward_sequence(video)

    def forward_latest_clip(self, video: torch.Tensor) -> VLMOutput:
        self.load()
        x = _latest_clip_bthwc(video, self.clip_length)
        input_batch_size = int(x.shape[0])
        model_input = _preprocess_for_internvideo2(x, self.image_size).to(_semantic_device(self.config))
        if int(model_input.shape[0]) != input_batch_size:
            raise RuntimeError(
                "forward_latest_clip expanded the online batch unexpectedly: "
                f"input_batch_size={input_batch_size}, model_input={tuple(model_input.shape)}"
            )
        use_amp = _semantic_device(self.config).type == "cuda" and self.inference_dtype in {"bf16", "bfloat16", "float16", "fp16"}
        dtype = torch.bfloat16 if self.inference_dtype in {"bf16", "bfloat16"} else torch.float16
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            out = self.model(model_input)
        logits_grid = out.q_sem_grid.detach().float().clone()[:, -1]
        if tuple(logits_grid.shape[-2:]) != self.grid_shape:
            logits_grid = F.interpolate(logits_grid[:, None], size=self.grid_shape, mode="bilinear", align_corners=False).squeeze(1)
        logits_grid = logits_grid / max(self.temperature, 1e-6)
        q_sem = torch.sigmoid(logits_grid).reshape(input_batch_size, 1, -1)
        patch_features = _last_temporal_patch_features(
            out.patch_tokens.detach().float().clone(),
            out.token_grid_shape,
            target_grid=self.grid_shape,
        ).reshape(input_batch_size, 1, self.grid_shape[0] * self.grid_shape[1], -1)
        trace = {
            "input_shape": list(video.shape),
            "latest_clip_shape": list(x.shape),
            "model_input_shape": list(model_input.shape),
            "patch_tokens_shape": list(out.patch_tokens.shape),
            "patch_features_shape": list(patch_features.shape),
            "q_sem_shape": list(q_sem.shape),
            "semantic_score_source": self.semantic_score_source,
            "temporal_selection": "latest_clip_only",
        }
        return VLMOutput(
            q_sem=q_sem.to(video.device),
            q_sem_logits=logits_grid.reshape(input_batch_size, 1, -1).to(video.device),
            patch_features=patch_features.to(video.device),
            patch_logits=logits_grid.reshape(input_batch_size, 1, -1).to(video.device),
            patch_probs=q_sem.to(video.device),
            raw_grid=torch.sigmoid(logits_grid).reshape(input_batch_size, 1, *self.grid_shape).to(video.device),
            token_grid_shape=self.grid_shape,
            source="finetune_v2_internvideo2_latest_clip",
            trace=trace,
            checkpoint_fingerprint=self.load_report.get("checkpoint_fingerprint"),
            preprocessing_fingerprint=self.preprocessing_fingerprint(),
        )

    @property
    def model(self):
        return object.__getattribute__(self, "_teacher_model")

    def preprocessing_fingerprint(self) -> str:
        return "v5_standard_internvideo2_preprocessing"


def align_relevance_to_grid(
    scores: torch.Tensor,
    source_grid: tuple[int, int] | tuple[int, int, int] | None,
    target_grid: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    if scores.ndim == 2:
        if source_grid is None:
            side = int(scores.shape[1] ** 0.5)
            source_grid = (side, side)
        if len(source_grid) == 3:
            source_grid = source_grid[-2:]
        scores = scores.reshape(scores.shape[0], 1, int(source_grid[0]), int(source_grid[1]))
    elif scores.ndim == 3:
        scores = scores.unsqueeze(1)
    elif scores.ndim != 4:
        raise ValueError(f"Expected q_sem scores [B,K], [B,H,W], or [B,1,H,W], got {tuple(scores.shape)}")
    aligned = F.interpolate(scores.float(), size=target_grid, mode="bilinear", align_corners=False).squeeze(1)
    return aligned.clamp(0, 1).reshape(aligned.shape[0], -1)


def build_semantic_teacher(resources, config: dict[str, Any]):
    semantic_cfg = config.get("semantic", {})
    tokens_cfg = config.get("tokens", {})
    token_grid = (int(tokens_cfg.get("grid_height", 8)), int(tokens_cfg.get("grid_width", 8)))
    grid = resolve_vcfa_grid(config, token_grid)
    if bool(semantic_cfg.get("mock", False)):
        return MockSemanticTeacher(grid, patch_feature_dim=int(semantic_cfg.get("mock_patch_feature_dim", 32)))
    if semantic_cfg.get("mode") != "finetune_v2_internvideo2_online":
        raise RuntimeError("semantic.mode must be finetune_v2_internvideo2_online; semantic student modes are forbidden.")
    return FinetuneV2InternVideo2SemanticTeacher(resources, config)


def causal_clips(
    video: torch.Tensor,
    clip_length: int,
    *,
    padding: str = "repeat_first",
    is_first: torch.Tensor | None = None,
) -> torch.Tensor:
    x = _as_bthwc(video)
    bsz, time = x.shape[:2]
    latest_reset = None
    if is_first is not None:
        resets = is_first.to(device=x.device).bool().reshape(bsz, time)
        reset_indices = torch.arange(time, device=x.device).reshape(1, time).expand(bsz, -1)
        reset_indices = reset_indices.masked_fill(~resets, -1)
        latest_reset = torch.cummax(reset_indices, dim=1).values.clamp_min(0)
    clips = []
    batch_indices = torch.arange(bsz, device=x.device)
    for t in range(time):
        start = max(0, t - int(clip_length) + 1)
        window = x[:, start : t + 1]
        fresh = window[:, :1]
        if latest_reset is not None:
            reset_t = latest_reset[:, t]
            fresh = x[batch_indices, reset_t][:, None]
            indices = torch.arange(start, t + 1, device=x.device).reshape(1, -1)
            valid = indices >= reset_t.reshape(-1, 1)
            window = torch.where(valid.reshape(bsz, -1, 1, 1, 1), window, fresh)
        if window.shape[1] < int(clip_length):
            if padding != "repeat_first":
                raise ValueError(f"Unsupported causal padding: {padding}")
            pad = fresh.expand(-1, int(clip_length) - window.shape[1], -1, -1, -1)
            window = torch.cat([pad, window], dim=1)
        clips.append(window)
    return torch.stack(clips, dim=1)


def _latest_clip_bthwc(video: torch.Tensor, clip_length: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected latest clip [B,T,H,W,C], [B,T,C,H,W], or [B,C,T,H,W], got {tuple(video.shape)}")
    if video.shape[-1] == 3:
        x = video
    elif video.shape[2] == 3:
        x = video.permute(0, 1, 3, 4, 2)
    elif video.shape[1] == 3:
        x = video.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Cannot infer latest clip layout from {tuple(video.shape)}")
    clip_length = int(clip_length)
    if x.shape[1] > clip_length:
        x = x[:, -clip_length:]
    if x.shape[1] < clip_length:
        pad = x[:, :1].expand(-1, clip_length - x.shape[1], -1, -1, -1)
        x = torch.cat([pad, x], dim=1)
    return x


def _last_semantic_output(out: VLMOutput) -> VLMOutput:
    return VLMOutput(
        q_sem=out.q_sem[:, -1:],
        raw_grid=out.raw_grid[:, -1:],
        token_grid_shape=out.token_grid_shape,
        source=out.source,
        q_sem_logits=None if out.q_sem_logits is None else out.q_sem_logits[:, -1:],
        patch_features=None if out.patch_features is None else out.patch_features[:, -1:],
        patch_logits=None if out.patch_logits is None else out.patch_logits[:, -1:],
        patch_probs=None if out.patch_probs is None else out.patch_probs[:, -1:],
        trace=out.trace,
        checkpoint_fingerprint=out.checkpoint_fingerprint,
        preprocessing_fingerprint=out.preprocessing_fingerprint,
    )


def _resolve_checkpoint(resources, config: Mapping[str, Any]) -> Path:
    semantic = config.get("semantic", {})
    vcfa = config.get("vcfa", {})
    model = config.get("model", {})
    model_vlm = model.get("vlm", {}) if isinstance(model, Mapping) else {}
    model_vcfa = model.get("vcfa", {}) if isinstance(model, Mapping) else {}
    candidates = [
        config.get("checkpoint"),
        semantic.get("checkpoint"),
        vcfa.get("checkpoint") if isinstance(vcfa, Mapping) else None,
        model_vlm.get("checkpoint") if isinstance(model_vlm, Mapping) else None,
        model_vcfa.get("checkpoint") if isinstance(model_vcfa, Mapping) else None,
        os.environ.get("VLM_FINETUNE_V2_CHECKPOINT"),
        getattr(resources, "vlm_finetune_v2_checkpoint", None),
        getattr(resources, "vlm_checkpoint", None),
    ]
    checked = []
    for raw in candidates:
        if raw:
            path = Path(raw).expanduser()
            checked.append(str(path))
            if path.exists():
                return path.resolve()
    raise FileNotFoundError(
        "Set VLM_FINETUNE_V2_CHECKPOINT, checkpoint, vcfa.checkpoint, "
        "or semantic.checkpoint to an existing finetune_v2 InternVideo2 checkpoint. "
        f"Checked: {checked}"
    )


def _resolve_base_checkpoint(resources, config: Mapping[str, Any]) -> Path | None:
    semantic = config.get("semantic", {})
    vcfa = config.get("vcfa", {})
    model = config.get("model", {})
    model_vlm = model.get("vlm", {}) if isinstance(model, Mapping) else {}
    model_vcfa = model.get("vcfa", {}) if isinstance(model, Mapping) else {}
    raw = (
        semantic.get("base_checkpoint")
        or (vcfa.get("base_checkpoint") if isinstance(vcfa, Mapping) else None)
        or config.get("base_checkpoint")
        or (model_vlm.get("base_checkpoint") if isinstance(model_vlm, Mapping) else None)
        or (model_vcfa.get("base_checkpoint") if isinstance(model_vcfa, Mapping) else None)
        or os.environ.get("INTERNVIDEO2_BASE_CHECKPOINT")
        or getattr(resources, "internvideo2_base_checkpoint", None)
    )
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.exists() else path


def _resolve_fine_turn_root(resources, config: Mapping[str, Any]) -> Path:
    semantic = config.get("semantic", {})
    candidates = [
        semantic.get("fine_turn_root"),
        os.environ.get("FINE_TURN_ROOT"),
        None if getattr(resources, "get_data_root", None) is None else Path(resources.get_data_root) / "fine-turn",
        Path(__file__).resolve().parents[2].parent / "get_data" / "fine-turn",
    ]
    for raw in candidates:
        if raw:
            path = Path(raw).expanduser()
            if (path / "fine_turn").exists():
                return path.resolve()
    raise FileNotFoundError("Could not find get_data/fine-turn. Set FINE_TURN_ROOT explicitly.")


def _semantic_device(config: Mapping[str, Any]) -> torch.device:
    raw = config.get("device", {}).get("vlm") or config.get("device", {}).get("train") or "cpu"
    device = torch.device(str(raw))
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _checkpoint_type(payload: Mapping[str, Any]) -> str:
    has_task = isinstance(payload.get("task_relevance"), Mapping)
    has_lora = bool(payload.get("lora_applied_modules"))
    if has_task and has_lora:
        return "internvideo2_base_plus_lora_plus_relevance_head"
    if has_task:
        return "internvideo2_base_plus_relevance_head"
    return "unknown"


def _state_match_report(state: Mapping[str, Any], model_state: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    matched: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for key, value in state.items():
        if key not in model_state or not torch.is_tensor(value) or not torch.is_tensor(model_state[key]):
            continue
        if tuple(value.shape) == tuple(model_state[key].shape):
            matched.append(str(key))
        else:
            mismatches.append(
                {
                    "key": str(key),
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
    return matched, mismatches


def _prepend_sys_path(path: Path) -> None:
    text = str(Path(path).resolve())
    sys.path[:] = [item for item in sys.path if str(Path(item).resolve()) != text]
    sys.path.insert(0, text)


def _evict_module_prefixes(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def _preprocess_for_internvideo2(video: torch.Tensor, image_size: int) -> torch.Tensor:
    x = _as_bthwc(video).float()
    if x.max() > 2:
        x = x / 255.0
    bsz, time, height, width, channels = x.shape
    x = x.permute(0, 1, 4, 2, 3).reshape(bsz * time, channels, height, width)
    x = F.interpolate(x, size=(int(image_size), int(image_size)), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.reshape(bsz, time, channels, int(image_size), int(image_size))


def _as_bthwc(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video [B,T,H,W,C] or [B,T,C,H,W], got {tuple(video.shape)}")
    if video.shape[-1] == 3:
        return video
    if video.shape[2] == 3:
        return video.permute(0, 1, 3, 4, 2)
    raise ValueError(f"Cannot infer video layout from {tuple(video.shape)}")


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(2)
    lo = flat.min(dim=2).values.reshape(x.shape[0], x.shape[1], 1, 1)
    hi = flat.max(dim=2).values.reshape(x.shape[0], x.shape[1], 1, 1)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def _mock_patch_features(grid: torch.Tensor, dim: int) -> torch.Tensor:
    bsz, time, height, width = grid.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=grid.dtype, device=grid.device),
        torch.linspace(-1.0, 1.0, width, dtype=grid.dtype, device=grid.device),
        indexing="ij",
    )
    pos = torch.stack([xx, yy, xx * yy], dim=-1).reshape(1, 1, height * width, 3).expand(bsz, time, -1, -1)
    score = grid.reshape(bsz, time, height * width, 1)
    base = torch.cat([score, pos, score * pos], dim=-1)
    repeat = (int(dim) + base.shape[-1] - 1) // base.shape[-1]
    return base.repeat_interleave(repeat, dim=-1)[..., : int(dim)]


def _last_temporal_patch_features(
    patch_tokens: torch.Tensor,
    token_grid_shape: tuple[int, int, int] | list[int] | None,
    *,
    target_grid: tuple[int, int],
) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens [B*T,N,D], got {tuple(patch_tokens.shape)}")
    if token_grid_shape is None:
        spatial = int(round(float(patch_tokens.shape[1]) ** 0.5))
        token_grid_shape = (1, spatial, spatial)
    t_grid, h_grid, w_grid = tuple(int(x) for x in token_grid_shape)
    expected = t_grid * h_grid * w_grid
    if expected != int(patch_tokens.shape[1]):
        raise ValueError(f"patch token count {patch_tokens.shape[1]} does not match token_grid_shape={token_grid_shape}")
    current = patch_tokens.reshape(patch_tokens.shape[0], t_grid, h_grid, w_grid, patch_tokens.shape[-1])[:, -1]
    current = current.permute(0, 3, 1, 2)
    if tuple(current.shape[-2:]) != tuple(int(x) for x in target_grid):
        current = F.interpolate(current, size=tuple(int(x) for x in target_grid), mode="area")
    return current.permute(0, 2, 3, 1).reshape(patch_tokens.shape[0], int(target_grid[0]) * int(target_grid[1]), patch_tokens.shape[-1])
