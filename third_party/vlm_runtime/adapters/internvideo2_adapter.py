from __future__ import annotations

import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vlm.video_preprocess import frames_to_internvideo2_tensor


DEFAULT_PATTERNS = {
    "pretrained": ("*1B*clip*.pth", "*InternVideo2*CLIP*1B*.pth"),
    "vision": ("InternVideo2*stage2*1b*224p*f4*.pt", "InternVideo2*stage2*1b*224p*f4*.pth", "InternVideo2_Stage2_1B.pth"),
    "text": ("internvl_c_13b_224px.pth",),
    "tokenizer": ("chinese_alpaca_lora_7b",),
}

SUPPORTED_ATTENTION_BACKENDS = ("naive", "native_flash")


@dataclass
class InternVideo2Discovery:
    multi_modality_root: Path | None
    config_candidates: list[Path]
    weight_candidates: dict[str, list[Path]]


class InternVideo2Adapter:
    def __init__(
        self,
        *,
        internvideo2_root: Path,
        model_weights_root: Path | None,
        checkpoint: Path | None,
        config: Path | None,
        device: str = "cpu",
        num_frames: int = 8,
        image_size: int = 224,
        dtype: str = "fp32",
        attention_backend: str = "naive",
        disable_text_lora: bool = False,
        patterns: Mapping[str, list[str]] | None = None,
    ):
        self.internvideo2_root = Path(internvideo2_root).expanduser().resolve()
        self.model_weights_root = None if model_weights_root is None else Path(model_weights_root).expanduser().resolve()
        self.explicit_checkpoint = None if checkpoint is None else Path(checkpoint).expanduser().resolve()
        self.explicit_config = None if config is None else Path(config).expanduser().resolve()
        self.device_text = device
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.dtype = dtype
        self.attention_backend = _normalize_attention_backend(attention_backend)
        self.disable_text_lora = bool(disable_text_lora)
        self.patterns = {key: tuple(value) for key, value in (patterns or DEFAULT_PATTERNS).items()}
        self.model = None
        self.config = None
        self.tokenizer = None
        self.feature_dim = None
        self.load_report = None
        self.backend_report: dict[str, Any] = {}

    def discover(self) -> InternVideo2Discovery:
        multi_root = resolve_multi_modality_root(self.internvideo2_root)
        config_candidates = []
        if self.explicit_config is not None:
            config_candidates.append(self.explicit_config)
        default = multi_root / "scripts" / "evaluation" / "clip" / "zero_shot" / "1B" / "config_msrvtt.py"
        if default.exists():
            config_candidates.append(default)
        weight_candidates = {key: [] for key in DEFAULT_PATTERNS}
        if self.model_weights_root is not None and self.model_weights_root.exists():
            for key, patterns in self.patterns.items():
                weight_candidates[key] = discover_candidates(self.model_weights_root, patterns)
        if self.explicit_checkpoint is not None:
            weight_candidates.setdefault("pretrained", [])
            weight_candidates["pretrained"] = [self.explicit_checkpoint, *weight_candidates["pretrained"]]
        return InternVideo2Discovery(multi_root, config_candidates, weight_candidates)

    def load(self) -> None:
        discovery = self.discover()
        if discovery.multi_modality_root is None:
            raise FileNotFoundError(f"InternVideo2 multi_modality not found under {self.internvideo2_root}")
        config_path = _single_or_raise("InternVideo2 config", discovery.config_candidates)
        pretrained = _single_or_raise("InternVideo2 pretrained checkpoint", discovery.weight_candidates.get("pretrained", []))
        tokenizer_path = _single_or_raise("InternVideo2 tokenizer", discovery.weight_candidates.get("tokenizer", []))
        vision = _single_or_raise("InternVideo2 vision checkpoint", discovery.weight_candidates.get("vision", []))
        text = _single_or_raise("InternVideo2 text checkpoint", discovery.weight_candidates.get("text", []))

        import torch

        package_root, package_prefix = resolve_import_package(discovery.multi_modality_root)
        _ensure_path(package_root)
        _ensure_path(discovery.multi_modality_root)
        if package_prefix:
            models_pkg = importlib.import_module(package_prefix + ".models")
            utils_pkg = importlib.import_module(package_prefix + ".utils")
            sys.modules["models"] = models_pkg
            sys.modules["utils"] = utils_pkg
            config_mod = importlib.import_module(package_prefix + ".utils.config")
            models_mod = importlib.import_module(package_prefix + ".models")
        else:
            config_mod = importlib.import_module("utils.config")
            models_mod = importlib.import_module("models")
        cfg = config_mod.eval_dict_leaf(config_mod.Config.from_file(str(config_path)))
        cfg.device = _normalize_device(self.device_text, torch)
        cfg.num_frames = self.num_frames
        cfg.num_frames_test = self.num_frames
        cfg.model.vision_encoder.num_frames = self.num_frames
        cfg.model.tokenizer_path = str(tokenizer_path)
        cfg.model.text_encoder.llama_path = str(tokenizer_path)
        cfg.model.vision_ckpt_path = str(vision)
        cfg.model.text_ckpt_path = str(text)
        # Stage2 V2 uses visual LoRA only.  The upstream InternVideo2
        # constructor enables its text-side PEFT adapter by default; disable
        # it explicitly before construction so no text LoRA parameters are
        # present in the model at all.
        if self.disable_text_lora:
            cfg.model.text_encoder.use_lora = False
        cfg.pretrained_path = str(pretrained)
        cfg.model.extra_ckpt_path = None
        cfg.deepspeed.enable = False
        cfg.use_half_precision = self.dtype == "fp16"
        cfg.use_bf16 = self.dtype == "bf16"
        cfg.gradient_checkpointing = False
        self.backend_report = _configure_optional_cuda_extensions(cfg, self.attention_backend)

        model_cls_name = cfg.model.get("model_cls", "InternVideo2_CLIP")
        model_cls = getattr(models_mod, model_cls_name)
        model = model_cls(config=cfg, is_pretrain=True)
        self.backend_report.update(_inspect_runtime_backend(model, self.attention_backend))
        checkpoint = torch.load(str(pretrained), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "module" in checkpoint:
            state_dict = checkpoint["module"]
        else:
            state_dict = checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        self.load_report = _load_key_report(
            msg,
            base_paths={
                "tokenizer_path": tokenizer_path,
                "vision_ckpt_path": vision,
                "text_ckpt_path": text,
                "pretrained_path": pretrained,
            },
        )
        self.load_report["attention_backend"] = dict(self.backend_report)
        print(
            "InternVideo2 load_state_dict "
            f"missing={self.load_report['missing_total']} "
            f"unexpected={self.load_report['unexpected_total']}"
        )
        print("InternVideo2 load_state_dict report:")
        print(json.dumps(self.load_report, indent=2))
        for param in model.parameters():
            param.requires_grad_(False)
        model = model.to(torch.device(cfg.device))
        if self.dtype == "fp16":
            model = model.to(torch.float16)
        elif self.dtype == "bf16":
            model = model.to(torch.bfloat16)
        else:
            model = model.to(torch.float32)
        model.eval()
        self.model = model
        self.config = cfg
        self.tokenizer = getattr(model, "tokenizer", None)
        probe = self.encode_video(np.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=np.uint8))
        self.feature_dim = int(probe.shape[-1])

    def encode_video(self, frames: np.ndarray) -> np.ndarray:
        if self.model is None or self.config is None:
            raise RuntimeError("InternVideo2Adapter.load() must be called before encode_video().")
        import torch

        tensor, indices = frames_to_internvideo2_tensor(
            frames,
            num_frames=self.num_frames,
            image_size=self.image_size,
            device=self.config.device,
        )
        print(f"raw frames: {tuple(np.asarray(frames).shape)}")
        print(f"sampled frame indices: {indices}")
        print(f"model input: {tuple(tensor.shape)}")
        with torch.inference_mode():
            feat = self.model.encode_vision(tensor, test=True)
            if isinstance(feat, tuple):
                feat = feat[-1]
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.detach().float().cpu().numpy().astype(np.float32)


def resolve_multi_modality_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = [root, root / "multi_modality", root / "InternVideo2" / "multi_modality"]
    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "utils" / "config.py").exists():
            return candidate
    raise FileNotFoundError(f"Could not find InternVideo2 multi_modality under {root}")


def resolve_import_package(multi_root: Path) -> tuple[Path, str]:
    multi_root = Path(multi_root).resolve()
    if multi_root.name == "multi_modality" and multi_root.parent.name == "InternVideo2":
        return multi_root.parent.parent, "InternVideo2.multi_modality"
    if multi_root.name == "multi_modality":
        return multi_root.parent, "multi_modality"
    return multi_root, ""


def discover_candidates(root: Path, patterns: tuple[str, ...] | list[str]) -> list[Path]:
    root = Path(root).expanduser().resolve()
    found: list[Path] = []
    for pattern in patterns:
        found.extend(path.resolve() for path in root.rglob(pattern))
    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return sorted(unique)


def _single_or_raise(label: str, candidates: list[Path]) -> Path:
    existing = []
    seen = set()
    for path in candidates:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    if not existing:
        raise FileNotFoundError(f"Missing {label}. Candidates: " + ", ".join(str(path) for path in candidates))
    if len(existing) > 1:
        raise RuntimeError(f"Multiple candidates for {label}; specify explicitly: " + ", ".join(str(path) for path in existing))
    return existing[0]


def _load_key_report(
    message: Any,
    *,
    sample_limit: int = 20,
    base_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    missing = list(getattr(message, "missing_keys", []) or [])
    unexpected = list(getattr(message, "unexpected_keys", []) or [])
    base_paths = base_paths or {}
    base_files_present = {
        key: Path(value).exists()
        for key, value in base_paths.items()
    }
    missing_categories = _category_counts(missing)
    unexpected_categories = _category_counts(unexpected)
    return {
        "missing_total": len(missing),
        "unexpected_total": len(unexpected),
        "base_checkpoint_paths": {key: str(value) for key, value in base_paths.items()},
        "base_files_present": base_files_present,
        "missing_by_category": missing_categories,
        "unexpected_by_category": unexpected_categories,
        "missing_prefix_counts": _prefix_counts(missing),
        "unexpected_prefix_counts": _prefix_counts(unexpected),
        "missing_samples": missing[:sample_limit],
        "unexpected_samples": unexpected[:sample_limit],
        "interpretation": _load_interpretation(missing_categories, unexpected_categories, base_files_present),
    }


def _category_counts(keys: list[str]) -> dict[str, int]:
    counts = {"vision_backbone": 0, "text_encoder": 0, "projection": 0, "head_or_task": 0, "other": 0}
    for key in keys:
        counts[_classify_key(key)] += 1
    return counts


def _classify_key(key: str) -> str:
    lower = key.lower()
    if lower.startswith("vision_encoder") or ".vision_encoder." in lower or "videomae" in lower or "internvideo2" in lower:
        return "vision_backbone"
    if lower.startswith("text_encoder") or ".text_encoder." in lower or "llama" in lower or "tokenizer" in lower:
        return "text_encoder"
    if "proj" in lower or "projection" in lower or "logit_scale" in lower:
        return "projection"
    if "head" in lower or "classifier" in lower or "loss" in lower or "temp" in lower:
        return "head_or_task"
    return "other"


def _prefix_counts(keys: list[str], *, limit: int = 20) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        parts = key.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit])


def _load_interpretation(
    missing_categories: dict[str, int],
    unexpected_categories: dict[str, int],
    base_files_present: dict[str, bool],
) -> str:
    vision_base_ready = bool(base_files_present.get("vision_ckpt_path"))
    text_base_ready = bool(base_files_present.get("text_ckpt_path"))
    if missing_categories["vision_backbone"] > 0 and not vision_base_ready:
        return (
            "Final CLIP checkpoint is missing vision backbone keys and no base "
            "vision checkpoint is configured. Do not trust training until fixed."
        )
    if missing_categories["text_encoder"] > 0 and not text_base_ready:
        return (
            "Final CLIP checkpoint is missing text encoder keys and no base text "
            "checkpoint is configured. Text-conditioned training is unsafe."
        )
    if missing_categories["vision_backbone"] > 0 or missing_categories["text_encoder"] > 0:
        return (
            "Final 1B CLIP checkpoint omits some base vision/text keys, while "
            "the corresponding base checkpoint files are configured. This matches "
            "the dreamer-master/official InternVideo2 flow: constructor loads "
            "base checkpoints first, then load_state_dict(strict=False) applies "
            "the final CLIP checkpoint. Forward probing is still required."
        )
    if sum(missing_categories.values()) or sum(unexpected_categories.values()):
        return "Only non-core or prefix-mismatch keys appear from this classifier; inspect samples."
    return "All checkpoint keys matched."


def _ensure_path(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def _normalize_device(device: str, torch: Any) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def _normalize_attention_backend(value: str) -> str:
    backend = str(value or "naive").strip().lower()
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"attention_backend must be one of {SUPPORTED_ATTENTION_BACKENDS}, got {value!r}"
        )
    return backend


def _configure_optional_cuda_extensions(config: Any, backend: str) -> dict[str, Any]:
    backend = _normalize_attention_backend(backend)
    enabled = backend == "native_flash"
    import_report: dict[str, Any] = {}
    if enabled:
        try:
            import flash_attn
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
            from flash_attn.modules.mlp import FusedMLP
            from flash_attn.ops.rms_norm import DropoutAddRMSNorm
        except Exception as exc:
            raise RuntimeError(
                "InternVideo2 native_flash requires FlashAttention, FusedMLP, and "
                "DropoutAddRMSNorm from one compatible flash-attn installation."
            ) from exc
        required = {
            "flash_attn_varlen_qkvpacked_func": flash_attn_varlen_qkvpacked_func,
            "unpad_input": unpad_input,
            "pad_input": pad_input,
            "FusedMLP": FusedMLP,
            "DropoutAddRMSNorm": DropoutAddRMSNorm,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"InternVideo2 native_flash imports resolved to None: {missing}")
        import_report = {
            "flash_attn_version": importlib_metadata.version("flash-attn"),
            "flash_attn_path": str(Path(inspect.getfile(flash_attn)).resolve()),
            "flash_attention_op_path": str(Path(inspect.getfile(flash_attn_varlen_qkvpacked_func)).resolve()),
            "fused_mlp_path": str(Path(inspect.getfile(FusedMLP)).resolve()),
            "fused_rmsnorm_path": str(Path(inspect.getfile(DropoutAddRMSNorm)).resolve()),
        }

    for path in (
        ("model", "vision_encoder", "use_flash_attn"),
        ("model", "vision_encoder", "use_fused_rmsnorm"),
        ("model", "vision_encoder", "use_fused_mlp"),
    ):
        _set_config_value(config, path, enabled, required=True)
    _set_config_value(config, ("model", "text_encoder", "use_flash_attn"), False, required=False)
    return {
        "requested_backend": backend,
        "configured_backend": "native_flash" if enabled else "naive",
        "vision_use_flash_attn": enabled,
        "vision_use_fused_mlp": enabled,
        "vision_use_fused_rmsnorm": enabled,
        "text_use_flash_attn": False,
        **import_report,
    }


def _set_config_value(config: Any, path: tuple[str, ...], value: Any, *, required: bool) -> None:
    target = config
    for part in path[:-1]:
        if isinstance(target, Mapping):
            target = target.get(part)
        else:
            target = getattr(target, part, None)
        if target is None:
            if required:
                raise RuntimeError(f"InternVideo2 config is missing {'.'.join(path[:-1])}")
            return
    key = path[-1]
    if isinstance(target, dict):
        if required and key not in target:
            raise RuntimeError(f"InternVideo2 config is missing {'.'.join(path)}")
        target[key] = value
        return
    if not hasattr(target, key):
        if required:
            raise RuntimeError(f"InternVideo2 config is missing {'.'.join(path)}")
        return
    setattr(target, key, value)


def _inspect_runtime_backend(model: Any, requested_backend: str) -> dict[str, Any]:
    attention_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "Attention"
        and module.__class__.__module__.endswith("internvideo2_clip_vision")
    ]
    flash_attention_modules = [module for module in attention_modules if bool(getattr(module, "use_flash_attn", False))]
    fused_mlp_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "FusedMLP" and module.__class__.__module__.startswith("flash_attn.")
    ]
    fused_rmsnorm_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "DropoutAddRMSNorm" and module.__class__.__module__.startswith("flash_attn.")
    ]
    native_active = bool(
        attention_modules
        and len(flash_attention_modules) == len(attention_modules)
        and fused_mlp_modules
        and fused_rmsnorm_modules
    )
    if requested_backend == "native_flash" and not native_active:
        raise RuntimeError(
            "InternVideo2 native_flash was requested but the constructed model is not fully fused: "
            f"attention={len(flash_attention_modules)}/{len(attention_modules)}, "
            f"fused_mlp={len(fused_mlp_modules)}, fused_rmsnorm={len(fused_rmsnorm_modules)}"
        )
    if requested_backend == "naive" and flash_attention_modules:
        raise RuntimeError("InternVideo2 naive backend unexpectedly constructed FlashAttention modules.")
    return {
        "actual_backend": "native_flash" if native_active else "naive",
        "vision_attention_module_count": len(attention_modules),
        "vision_flash_attention_module_count": len(flash_attention_modules),
        "vision_fused_mlp_module_count": len(fused_mlp_modules),
        "vision_fused_rmsnorm_module_count": len(fused_rmsnorm_modules),
        "attention_class_path": (
            str(Path(inspect.getfile(attention_modules[0].__class__)).resolve())
            if attention_modules
            else None
        ),
        "inner_attention_class": (
            f"{flash_attention_modules[0].inner_attn.__class__.__module__}."
            f"{flash_attention_modules[0].inner_attn.__class__.__name__}"
            if flash_attention_modules
            else None
        ),
    }
