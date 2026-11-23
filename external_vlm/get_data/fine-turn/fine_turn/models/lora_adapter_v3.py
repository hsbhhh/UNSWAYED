from __future__ import annotations

from dataclasses import dataclass
import re

import torch


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"


class LoRALinear(torch.nn.Module):
    def __init__(self, base, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / max(1, self.rank)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.lora_a = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = torch.nn.Linear(self.rank, base.out_features, bias=False)
        # The official InternVideo2 backbone is loaded in bf16/fp16.  Match
        # the adapter parameters to the wrapped projection immediately so
        # mixed-precision forwards do not promote or reject the LoRA path.
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale

    @property
    def weight(self) -> torch.Tensor:
        """Expose a differentiable merged weight for FlashAttention FusedMLP."""
        delta = self.lora_b.weight @ self.lora_a.weight
        return self.base.weight + delta * self.scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def merged_linear(self) -> torch.nn.Linear:
        """Return an inference-only Linear with the LoRA delta folded in."""
        if self.training and float(self.dropout.p) > 0.0:
            raise RuntimeError("LoRA must be in eval mode before it can be merged for inference.")
        base = self.base
        merged = torch.nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        with torch.no_grad():
            delta = self.lora_b.weight.float() @ self.lora_a.weight.float()
            merged.weight.copy_((base.weight.float() + delta * self.scale).to(base.weight.dtype))
            if base.bias is not None:
                merged.bias.copy_(base.bias)
        merged.requires_grad_(False)
        merged.train(self.training)
        return merged


def candidate_lora_modules(model) -> list[str]:
    wanted = ("qkv", "attn", "proj", "fc", "mlp")
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(part in name.lower() for part in wanted):
            names.append(name)
    return names


def candidate_visual_lora_modules(model, *, last_n_blocks: int = 4) -> list[str]:
    """Return attention/MLP linears in the last N vision encoder blocks only."""
    block_pattern = re.compile(r"(?:^|\.)vision_encoder\.blocks\.(\d+)\.")
    allowed_suffixes = (".attn.qkv", ".attn.proj", ".mlp.fc1", ".mlp.fc2")
    candidates: list[tuple[str, int]] = []
    for name, module in model.named_modules():
        lower = name.lower()
        if not isinstance(module, torch.nn.Linear) or "text_encoder" in lower:
            continue
        match = block_pattern.search(lower)
        if match is None or not lower.endswith(allowed_suffixes):
            continue
        candidates.append((name, int(match.group(1))))
    if not candidates:
        return []
    blocks = sorted({block for _, block in candidates})
    selected = set(blocks[-max(1, int(last_n_blocks)) :])
    return [name for name, block in candidates if block in selected]


def apply_lora_to_named_linears(model, names: list[str], config: LoRAConfig) -> list[str]:
    import torch

    modules = dict(model.named_modules())
    applied = []
    for name in names:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, LoRALinear(modules[name], rank=config.rank, alpha=config.alpha, dropout=config.dropout))
        applied.append(name)
    return applied


def merge_lora_linears_for_inference(model) -> list[str]:
    """Replace every LoRALinear with its equivalent plain Linear in-place."""
    modules = dict(model.named_modules())
    names = [name for name, module in modules.items() if isinstance(module, LoRALinear)]
    for name in names:
        module = modules[name]
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, module.merged_linear())
    return names
