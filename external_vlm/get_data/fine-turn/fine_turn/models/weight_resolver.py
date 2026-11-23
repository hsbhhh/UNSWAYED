from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WeightCandidates:
    clip: list[Path]
    stage2: list[Path]
    internvl: list[Path]


def discover_weight_candidates(root: Path | None) -> WeightCandidates:
    if root is None or not Path(root).exists():
        return WeightCandidates([], [], [])
    root = Path(root).expanduser().resolve()
    return WeightCandidates(
        clip=_unique(root.rglob("*clip*.pth")),
        stage2=_unique([*root.rglob("*stage2*.pt"), *root.rglob("*Stage2*.pth")]),
        internvl=_unique(root.rglob("internvl*.pth")),
    )


def resolve_single(label: str, explicit: Path | None, candidates: Iterable[Path]) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    existing = [Path(path).expanduser().resolve() for path in candidates if Path(path).exists()]
    if not existing:
        return None
    if len(existing) > 1:
        raise RuntimeError(f"Multiple candidates for {label}; pass explicit path: " + ", ".join(str(p) for p in existing))
    return existing[0]


def _unique(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return sorted(out)

