from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from .schema import ReplaySchema, validate_record


@dataclass
class CollectedShard:
    shard_id: int
    policy_version: int
    records: list[dict[str, Any]] = field(default_factory=list)
    schema: ReplaySchema = field(default_factory=ReplaySchema)


@dataclass
class VLMProcessedShard:
    shard_id: int
    policy_version: int
    records: list[dict[str, Any]]
    schema: ReplaySchema = field(default_factory=ReplaySchema)


@dataclass
class ReadyShard:
    shard_id: int
    policy_version: int
    records: list[dict[str, Any]]
    schema: ReplaySchema = field(default_factory=ReplaySchema)
    storage_path: str | None = None


class ShardStore:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_collected_shards: int = 2,
        max_ready_shards: int = 4,
        retain_ready_shards: int = 4,
        schema: ReplaySchema | None = None,
    ):
        self.root = None if root is None else Path(root)
        self.max_collected_shards = max(1, int(max_collected_shards))
        self.max_ready_shards = max(0, int(max_ready_shards))
        self.retain_ready_shards = max(0, int(retain_ready_shards))
        self.schema = schema or ReplaySchema()
        self.next_shard_id = 0
        self.collected: list[CollectedShard] = []
        self.ready: list[ReadyShard] = []
        self.released_collected_shards = 0
        self.released_ready_shards = 0

    def create_collected(self, *, policy_version: int) -> CollectedShard:
        shard = CollectedShard(self.next_shard_id, int(policy_version), schema=self.schema)
        self.next_shard_id += 1
        self.collected.append(shard)
        self._trim_collected()
        return shard

    def mark_ready(self, shard: CollectedShard | VLMProcessedShard) -> ReadyShard:
        for record in shard.records:
            validate_record(record, require_vlm=True, expected_schema=self.schema)
        records = shard.records
        shard.records = []
        self.collected = [item for item in self.collected if int(item.shard_id) != int(shard.shard_id)]
        ready = ReadyShard(shard.shard_id, shard.policy_version, records, shard.schema)
        self.ready.append(ready)
        self._trim_ready(protected=ready)
        return ready

    def save_ready(self, shard: ReadyShard) -> Path | None:
        if self.root is None:
            return None
        path = self.root / "shards" / f"shard_{int(shard.shard_id):06d}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": shard.schema.to_dict(), "records": shard.records}, path / "records.pt")
        shard.storage_path = str((path / "records.pt").resolve())
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "shard_id": int(shard.shard_id),
                    "policy_version": int(shard.policy_version),
                    "record_count": len(shard.records),
                    "schema": shard.schema.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def state_dict(self) -> dict[str, int]:
        return {
            "next_shard_id": int(self.next_shard_id),
            "released_collected_shards": int(self.released_collected_shards),
            "released_ready_shards": int(self.released_ready_shards),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for shard in self.collected:
            shard.records.clear()
        for shard in self.ready:
            shard.records.clear()
        self.collected.clear()
        self.ready.clear()
        self.next_shard_id = int(state.get("next_shard_id", 0))
        self.released_collected_shards = int(state.get("released_collected_shards", 0))
        self.released_ready_shards = int(state.get("released_ready_shards", 0))

    def retention_snapshot(self) -> dict[str, int]:
        return {
            "max_collected_shards": self.max_collected_shards,
            "max_ready_shards": self.max_ready_shards,
            "retain_ready_shards": self.retain_ready_shards,
            "collected_shards": len(self.collected),
            "ready_shards": len(self.ready),
            "released_collected_shards": self.released_collected_shards,
            "released_ready_shards": self.released_ready_shards,
            "collected_record_references": sum(len(shard.records) for shard in self.collected),
            "ready_record_references": sum(len(shard.records) for shard in self.ready),
        }

    def _trim_collected(self) -> None:
        while len(self.collected) > self.max_collected_shards:
            released = self.collected.pop(0)
            released.records.clear()
            self.released_collected_shards += 1

    def _trim_ready(self, *, protected: ReadyShard | None = None) -> None:
        limit = min(self.max_ready_shards, self.retain_ready_shards)
        while len(self.ready) > limit:
            released = self.ready.pop(0)
            if released is not protected:
                released.records.clear()
            self.released_ready_shards += 1
