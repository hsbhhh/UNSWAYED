from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class JsonlMetricsWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | bool | None]:
    out: dict[str, float | int | str | bool | None] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 0:
                out[key] = None
            elif value.dtype == torch.bool:
                out[key] = bool(value.detach().flatten()[0].cpu())
            else:
                out[key] = float(value.detach().float().mean().cpu())
        elif isinstance(value, (str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, int):
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _to_jsonable(value.detach().cpu().reshape(()).item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
