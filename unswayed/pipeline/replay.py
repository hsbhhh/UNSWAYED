from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import ReplaySchema, validate_record


StreamKey = tuple[int, int]
SourceRef = tuple[str, int] | None


class ReplayBuffer:
    """Replay with incrementally maintained per-episode sequence indexes."""

    def __init__(
        self,
        *,
        capacity: int,
        sequence_length: int,
        seed: int = 0,
        expected_schema: ReplaySchema | None = None,
    ):
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.records: deque[dict[str, Any]] = deque()
        self._sources: deque[SourceRef] = deque()
        self._streams: OrderedDict[StreamKey, list[dict[str, Any]]] = OrderedDict()
        self.rng = np.random.default_rng(int(seed))
        self.expected_schema = expected_schema
        self.last_sample_identities: list[tuple[int, int, tuple[int, ...]]] = []

    def append(self, record: dict[str, Any], *, source: SourceRef = None) -> None:
        validate_record(record, require_vlm=True, expected_schema=self.expected_schema)
        self.records.append(record)
        self._sources.append(source)
        self._insert_stream_record(record)
        while len(self.records) > self.capacity:
            removed = self.records.popleft()
            self._sources.popleft()
            self._remove_stream_record(removed)

    def append_shard(self, shard) -> None:
        for record_index, record in enumerate(shard.records):
            source = None
            if getattr(shard, "storage_path", None):
                source = (str(Path(shard.storage_path).resolve()), int(record_index))
            self.append(record, source=source)

    def __len__(self) -> int:
        return len(self.records)

    def valid_window_identities(self, sequence_length: int | None = None) -> list[tuple[int, int, tuple[int, ...]]]:
        length = int(sequence_length or self.sequence_length)
        identities: list[tuple[int, int, tuple[int, ...]]] = []
        for (env_id, episode_id), rows in self._streams.items():
            for start in range(max(0, len(rows) - length + 1)):
                indices = tuple(int(row["observation_index"]) for row in rows[start : start + length])
                identities.append((env_id, episode_id, indices))
        return identities

    def sample(self, *, batch_size: int | None = None, sequence_length: int | None = None) -> dict[str, torch.Tensor]:
        batch_size = int(batch_size or 1)
        sequence_length = int(sequence_length or self.sequence_length)
        streams: list[tuple[StreamKey, list[dict[str, Any]], int]] = []
        candidate_count = 0
        for key, rows in self._streams.items():
            count = max(0, len(rows) - sequence_length + 1)
            if count:
                streams.append((key, rows, count))
                candidate_count += count
        if candidate_count == 0:
            raise RuntimeError("ReplayBuffer has no contiguous sequence candidates")

        chosen: list[list[dict[str, Any]]] = []
        identities: list[tuple[int, int, tuple[int, ...]]] = []
        for _ in range(batch_size):
            candidate_index = int(self.rng.integers(0, candidate_count))
            for (env_id, episode_id), rows, count in streams:
                if candidate_index < count:
                    sequence = rows[candidate_index : candidate_index + sequence_length]
                    chosen.append(sequence)
                    identities.append(
                        (env_id, episode_id, tuple(int(row["observation_index"]) for row in sequence))
                    )
                    break
                candidate_index -= count
        self.last_sample_identities = identities
        return _stack_batch(chosen)

    def state_dict(self, *, persist_records: bool) -> dict[str, Any]:
        state: dict[str, Any] = {
            "capacity": self.capacity,
            "sequence_length": self.sequence_length,
            "record_count": len(self.records),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "schema": None if self.expected_schema is None else self.expected_schema.to_dict(),
        }
        if not persist_records:
            state["storage"] = "not_persisted"
            return state
        if all(source is not None for source in self._sources):
            state["storage"] = "shard_manifest"
            state["source_refs"] = [list(source) for source in self._sources if source is not None]
        else:
            state["storage"] = "embedded_records"
            state["records"] = list(self.records)
        return state

    def load_state_dict(self, state: dict[str, Any], *, shard_root: str | Path | None = None) -> None:
        saved_capacity = int(state.get("capacity", self.capacity))
        saved_sequence_length = int(state.get("sequence_length", self.sequence_length))
        if saved_capacity != self.capacity or saved_sequence_length != self.sequence_length:
            raise RuntimeError(
                "Replay resume configuration mismatch: "
                f"checkpoint=(capacity={saved_capacity}, sequence_length={saved_sequence_length}), "
                f"runtime=(capacity={self.capacity}, sequence_length={self.sequence_length})"
            )
        saved_schema = state.get("schema")
        if self.expected_schema is not None:
            if not isinstance(saved_schema, dict):
                raise RuntimeError("Replay resume is missing required schema metadata")
            if str(saved_schema.get("schema_version")) != self.expected_schema.schema_version:
                raise RuntimeError(
                    "Replay resume schema mismatch: "
                    f"checkpoint={saved_schema.get('schema_version')!r}, "
                    f"runtime={self.expected_schema.schema_version!r}"
                )
            if int(saved_schema.get("counterfactual_K", 2)) != int(self.expected_schema.counterfactual_K):
                raise RuntimeError(
                    "Replay resume counterfactual K mismatch: "
                    f"checkpoint={saved_schema.get('counterfactual_K')}, "
                    f"runtime={self.expected_schema.counterfactual_K}"
                )
        self.clear()
        storage = str(state.get("storage", "not_persisted"))
        if storage == "shard_manifest":
            cache: dict[str, list[dict[str, Any]]] = {}
            for raw_path, raw_record_index in state.get("source_refs", []):
                path = Path(str(raw_path))
                if not path.is_absolute() and shard_root is not None:
                    path = Path(shard_root) / path
                path_key, record_index = str(path.resolve()), int(raw_record_index)
                if path_key not in cache:
                    if not path.exists():
                        raise FileNotFoundError(f"Replay restore is missing persisted shard: {path}")
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    cache[path_key] = list(payload["records"])
                rows = cache[path_key]
                if not 0 <= record_index < len(rows):
                    raise RuntimeError(
                        f"Replay manifest record index is invalid: path={path}, index={record_index}, count={len(rows)}"
                    )
                self.append(rows[record_index], source=(path_key, record_index))
        elif storage == "embedded_records":
            for record in state.get("records", []):
                self.append(record)
        elif storage != "not_persisted":
            raise RuntimeError(f"unsupported Replay checkpoint storage={storage!r}")
        expected = int(state.get("record_count", len(self.records)))
        if storage != "not_persisted" and len(self.records) != expected:
            raise RuntimeError(f"Replay restore count mismatch: loaded={len(self.records)}, expected={expected}")
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = deepcopy(state["rng_state"])

    def clear(self) -> None:
        self.records.clear()
        self._sources.clear()
        self._streams.clear()
        self.last_sample_identities = []

    def _insert_stream_record(self, record: dict[str, Any]) -> None:
        key = (int(record["env_id"]), int(record["episode_id"]))
        rows = self._streams.setdefault(key, [])
        index = int(record["observation_index"])
        position = bisect_right([int(row["observation_index"]) for row in rows], index)
        rows.insert(position, record)

    def _remove_stream_record(self, record: dict[str, Any]) -> None:
        key = (int(record["env_id"]), int(record["episode_id"]))
        rows = self._streams[key]
        for index, candidate in enumerate(rows):
            if candidate is record:
                del rows[index]
                break
        else:
            raise RuntimeError("Replay incremental stream index lost record identity")
        if not rows:
            del self._streams[key]


def legacy_valid_window_identities(
    records: list[dict[str, Any]] | deque[dict[str, Any]], sequence_length: int
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Pre-hardening implementation retained only as a parity oracle."""

    streams: OrderedDict[StreamKey, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        streams.setdefault((int(record["env_id"]), int(record["episode_id"])), []).append(record)
    candidates = []
    for (env_id, episode_id), rows in streams.items():
        rows = sorted(rows, key=lambda item: int(item["observation_index"]))
        for start in range(max(0, len(rows) - int(sequence_length) + 1)):
            indices = tuple(int(row["observation_index"]) for row in rows[start : start + int(sequence_length)])
            candidates.append((env_id, episode_id, indices))
    return candidates


def _stack_batch(sequences: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
    keys = set.intersection(*(set(row.keys()) for seq in sequences for row in seq))
    skip = {
        "schema_version",
        "temporal_contract",
        "counterfactual_view_ids",
        "temporal_protocol_version",
        "pairing_contract",
        "background_stream_policy",
        "background_ids_b",
        "view_ids_b",
    }
    out = {}
    for key in sorted(keys - skip):
        values = [[row[key] for row in seq] for seq in sequences]
        try:
            arr = np.asarray(values)
        except Exception:
            continue
        if arr.dtype.kind in {"U", "S", "O"}:
            continue
        out[key] = torch.as_tensor(arr)
    # Preserve K-view identity as numeric, deterministic metadata.
    # The persisted replay schema remains unchanged; these tensors are derived
    # only while stacking a sampled batch so CAPC can catch view permutation or
    # duplicate-background errors instead of trusting raw pixel position.
    capc_identity_fields = (
        "view_ids_b",
        "background_ids_b",
        "physics_state_hash",
        "camera_id",
        "env_id",
        "episode_id",
        "observation_index",
        "step_id",
        "incoming_action_index",
        "outgoing_action_index",
        "action",
        "incoming_action",
        "outgoing_action",
        "is_first",
        "is_last",
        "is_terminal",
        "paired_view_valid",
        "multi_view_valid",
        "vlm_views_valid",
    )
    if all(all(name in row for name in capc_identity_fields) for seq in sequences for row in seq):
        identity_ids: dict[str, int] = {}
        identity_rows: dict[str, list[Any]] = {
            "capc_view_slot_id": [],
            "capc_background_id": [],
            "capc_physics_state_id": [],
            "capc_row_physics_state_id": [],
            "capc_camera_id": [],
            "capc_env_id": [],
            "capc_episode_id": [],
            "capc_observation_index": [],
            "capc_step_id": [],
            "capc_incoming_action_index": [],
            "capc_outgoing_action_index": [],
            "capc_incoming_action": [],
            "capc_outgoing_action": [],
            "capc_is_first": [],
            "capc_is_last": [],
            "capc_is_terminal": [],
            "capc_paired_view_valid": [],
            "capc_multi_view_valid": [],
            "capc_vlm_views_valid": [],
        }
        for sequence in sequences:
            sequence_rows = {name: [] for name in identity_rows}
            for row in sequence:
                configured = tuple(map(str, row.get("counterfactual_view_ids", ())))
                observed = tuple(map(str, row.get("view_ids_b", ())))
                backgrounds = tuple(map(str, row.get("background_ids_b", ())))
                view_count = len(observed)
                slots = []
                for value in observed:
                    label = value.split(":", 1)[0]
                    slots.append(configured.index(label) if label in configured else -1)
                sequence_rows["capc_view_slot_id"].append(slots)
                sequence_rows["capc_background_id"].append(
                    [_identity_id(value, identity_ids) for value in backgrounds]
                )
                physics = _identity_id(str(row["physics_state_hash"]), identity_ids)
                sequence_rows["capc_physics_state_id"].append([physics] * view_count)
                sequence_rows["capc_row_physics_state_id"].append(physics)
                scalar_fields = {
                    "capc_camera_id": int(row["camera_id"]),
                    "capc_env_id": int(row["env_id"]),
                    "capc_episode_id": int(row["episode_id"]),
                    "capc_observation_index": int(row["observation_index"]),
                    "capc_step_id": int(row["step_id"]),
                    "capc_incoming_action_index": int(row["incoming_action_index"]),
                    "capc_outgoing_action_index": int(row["outgoing_action_index"]),
                    "capc_is_first": bool(row["is_first"]),
                    "capc_is_last": bool(row["is_last"]),
                    "capc_is_terminal": bool(row["is_terminal"]),
                    "capc_paired_view_valid": bool(row["paired_view_valid"]),
                    "capc_multi_view_valid": bool(row["multi_view_valid"]),
                    "capc_vlm_views_valid": bool(row["vlm_views_valid"]),
                }
                for name, value in scalar_fields.items():
                    sequence_rows[name].append([value] * view_count)
                for name, source in (
                    ("capc_incoming_action", "incoming_action"),
                    ("capc_outgoing_action", "outgoing_action"),
                ):
                    value = np.asarray(row[source], dtype=np.float32)
                    sequence_rows[name].append(
                        np.repeat(value[None], view_count, axis=0)
                    )
            for name in identity_rows:
                identity_rows[name].append(sequence_rows[name])
        for name, values in identity_rows.items():
            array = np.asarray(values)
            if name == "capc_row_physics_state_id":
                out[name] = torch.as_tensor(array, dtype=torch.int64)
            else:
                out[name] = torch.as_tensor(array)
    out["sequence_valid"] = torch.ones(len(sequences), len(sequences[0]), dtype=torch.bool)
    return out


def _identity_id(value: str, identities: dict[str, int]) -> int:
    """Return a batch-local equality label without computing a data hash."""

    key = str(value)
    if key not in identities:
        identities[key] = len(identities) + 1
    return identities[key]
