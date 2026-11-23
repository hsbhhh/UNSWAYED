from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CapturedToken:
    module_name: str
    value: Any


def list_module_names(model) -> list[str]:
    return [name for name, _ in model.named_modules()]


class TokenExtractor:
    def __init__(self, model, module_name: str):
        self.model = model
        self.module_name = module_name
        self.captured: CapturedToken | None = None
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"Module not found: {module_name}")
        self.handle = modules[module_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.captured = CapturedToken(self.module_name, output)

    def close(self) -> None:
        self.handle.remove()

