from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import to_plain
from ..pipeline.schema import ReplaySchema


# Compatibility marker required to load the final trained checkpoint. It is a
# payload schema value, not an import/path/runtime dependency on the UNSWAYED tree.
LEGACY_V4_CHECKPOINT_MARKER = "dreamerv3_vlm_v4"


@dataclass
class CheckpointLoadReport:
    path: str
    step: int
    loaded: bool
    model_version: str | None
    implementation_revision: str | None
    schema_version: str | None
    temporal_contract: str | None
    temporal_contract_validated: bool
    missing_keys: list[str]
    unexpected_keys: list[str]
    optimizer_loaded: bool
    behavior_loaded: bool
    world_optimizer_loaded: bool
    behavior_optimizer_loaded: bool
    world_scheduler_loaded: bool
    behavior_scheduler_loaded: bool
    replay_loaded: bool
    shard_store_loaded: bool
    counters_loaded: bool
    slow_value_counter_loaded: bool
    rng_loaded: bool
    training_state_restored: bool
    trainer_counters: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def save_checkpoint(
    path: str | Path,
    *,
    config,
    model,
    behavior,
    world_optimizer=None,
    behavior_optimizer=None,
    world_scheduler=None,
    behavior_scheduler=None,
    trainer_state=None,
    replay=None,
    shard_store=None,
    step: int | None = None,
    gradient_updates: int = 0,
    replay_records: int = 0,
    previous_path: str | Path | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counters = _trainer_counters(trainer_state, step, gradient_updates, replay_records)
    payload = {
        "marker": LEGACY_V4_CHECKPOINT_MARKER,
        "model_version": str(config.checkpoint.model_version),
        "implementation_revision": str(config.checkpoint.implementation_revision),
        "schema": ReplaySchema.from_config(config).to_dict(),
        "counterfactual_metadata": _counterfactual_metadata(config, model),
        "temporal_contract": str(config.replay.temporal_contract),
        "config": to_plain(config),
        "step": int(counters["global_env_steps"]),
        "global_env_steps": int(counters["global_env_steps"]),
        "gradient_updates": int(counters["gradient_updates"]),
        "replay_records": int(counters["replay_records"]),
        "trainer_state": counters,
        "slow_value_updates": int(getattr(behavior, "slow_value_updates", 0)) if behavior is not None else 0,
        "model": model.state_dict(),
        "behavior": behavior.state_dict() if behavior is not None else None,
        "world_optimizer": world_optimizer.state_dict() if world_optimizer is not None else None,
        "behavior_optimizer": behavior_optimizer.state_dict() if behavior_optimizer is not None else None,
        "world_scheduler": world_scheduler.state_dict() if world_scheduler is not None else None,
        "behavior_scheduler": behavior_scheduler.state_dict() if behavior_scheduler is not None else None,
        "replay": replay.state_dict(persist_records=bool(config.replay.persist_replay)) if replay is not None else None,
        "shard_store": shard_store.state_dict() if shard_store is not None else None,
        "rng_state": capture_rng_state(),
    }
    _atomic_save_checkpoint(
        path,
        payload,
        previous_path=None if previous_path is None else Path(previous_path),
    )


def validate_checkpoint_file(path: str | Path) -> set[Path]:
    """Round-trip load and structurally validate a checkpoint without mutation.

    The returned paths are the replay shard files protected by the checkpoint.
    Every manifest reference is checked for existence and against the actual
    record count in its shard, so callers may safely use the set as a GC root.
    """

    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"checkpoint round-trip load failed: {path}") from exc
    return _validate_checkpoint_payload(payload, checkpoint_root=path.parent, path=path)


def checkpoint_replay_references(path: str | Path) -> set[Path]:
    """Return fully validated replay shard references from ``path``."""

    path = Path(path)
    if not path.is_file():
        return set()
    return validate_checkpoint_file(path)


def _atomic_save_checkpoint(
    path: Path,
    payload: dict[str, Any],
    *,
    previous_path: Path | None,
) -> None:
    """Commit a validated checkpoint without exposing a partial destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if previous_path is not None:
        previous_path = Path(previous_path)
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        if previous_path.parent.resolve() != path.parent.resolve():
            raise ValueError("previous checkpoint must share the destination directory")
        if previous_path.resolve() == path.resolve():
            raise ValueError("previous checkpoint path must differ from destination")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        # This is deliberately before either durable name is changed. A
        # serialization or replay-manifest failure therefore cannot replace a
        # previously resumable latest.pt.
        validate_checkpoint_file(temporary)

        if previous_path is not None and path.is_file():
            _atomic_preserve_checkpoint(path, previous_path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_preserve_checkpoint(current: Path, previous: Path) -> None:
    """Atomically preserve ``current`` as ``previous`` after full validation."""

    # Refuse to rotate a corrupt or non-resumable current checkpoint. The
    # existing previous file remains untouched on every failure before replace.
    validate_checkpoint_file(current)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{previous.name}.", suffix=".tmp", dir=previous.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(current, temporary)
        except OSError:
            shutil.copyfile(current, temporary)
        _fsync_file(temporary)
        # Validate structure directly. UNSWAYED does not calculate checkpoint hashes.
        validate_checkpoint_file(temporary)
        os.replace(temporary, previous)
        _fsync_directory(previous.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_checkpoint_payload(
    payload: Any,
    *,
    checkpoint_root: Path,
    path: Path,
) -> set[Path]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"checkpoint is not a mapping: {path}")
    required = {
        "marker",
        "model_version",
        "implementation_revision",
        "schema",
        "counterfactual_metadata",
        "temporal_contract",
        "config",
        "step",
        "global_env_steps",
        "gradient_updates",
        "replay_records",
        "trainer_state",
        "slow_value_updates",
        "model",
        "behavior",
        "world_optimizer",
        "behavior_optimizer",
        "world_scheduler",
        "behavior_scheduler",
        "replay",
        "shard_store",
        "rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"checkpoint is missing required fields {missing}: {path}")
    if payload.get("marker") != LEGACY_V4_CHECKPOINT_MARKER:
        raise RuntimeError(f"checkpoint has invalid UNSWAYED marker: {path}")
    for name in ("schema", "counterfactual_metadata", "config", "trainer_state", "rng_state"):
        if not isinstance(payload.get(name), Mapping):
            raise RuntimeError(f"checkpoint field {name!r} is not a mapping: {path}")
    schema = payload["schema"]
    required_schema = {
        "schema_version",
        "temporal_contract",
        "counterfactual_K",
        "view_ids",
    }
    missing_schema = sorted(required_schema - set(schema))
    if missing_schema:
        raise RuntimeError(f"checkpoint schema is missing {missing_schema}: {path}")
    for name in ("model_version", "implementation_revision", "temporal_contract"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise RuntimeError(f"checkpoint field {name!r} is empty or malformed: {path}")
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RuntimeError(f"checkpoint model state is empty or malformed: {path}")
    if any(not torch.is_tensor(value) for value in model_state.values()):
        raise RuntimeError(f"checkpoint model state contains non-tensor values: {path}")
    optional_mappings = (
        "behavior",
        "world_optimizer",
        "behavior_optimizer",
        "world_scheduler",
        "behavior_scheduler",
        "shard_store",
    )
    for name in optional_mappings:
        if payload.get(name) is not None and not isinstance(payload[name], Mapping):
            raise RuntimeError(f"checkpoint field {name!r} is malformed: {path}")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing_rng = sorted(required_rng - set(payload["rng_state"]))
    if missing_rng:
        raise RuntimeError(f"checkpoint RNG state is missing {missing_rng}: {path}")
    for name in ("step", "global_env_steps", "gradient_updates", "replay_records"):
        try:
            value = int(payload[name])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"checkpoint counter {name!r} is not integral: {path}") from exc
        if value < 0:
            raise RuntimeError(f"checkpoint counter {name!r} is negative: {path}")
    try:
        slow_value_updates = int(payload["slow_value_updates"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"checkpoint slow_value_updates is not integral: {path}") from exc
    if slow_value_updates < 0:
        raise RuntimeError(f"checkpoint slow_value_updates is negative: {path}")
    if int(payload["step"]) != int(payload["global_env_steps"]):
        raise RuntimeError(f"checkpoint step/global_env_steps disagree: {path}")
    trainer_state = payload["trainer_state"]
    for name in (
        "global_env_steps",
        "gradient_updates",
        "replay_records",
        "shards_collected",
        "policy_version",
    ):
        if name not in trainer_state:
            raise RuntimeError(f"checkpoint trainer_state is missing {name!r}: {path}")
        try:
            trainer_value = int(trainer_state[name])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"checkpoint trainer_state counter {name!r} is not integral: {path}"
            ) from exc
        if trainer_value < 0:
            raise RuntimeError(
                f"checkpoint trainer_state counter {name!r} is negative: {path}"
            )
    for name in ("global_env_steps", "gradient_updates", "replay_records"):
        if name not in trainer_state or int(trainer_state[name]) != int(payload[name]):
            raise RuntimeError(f"checkpoint trainer_state counter {name!r} disagrees: {path}")
    replay = payload.get("replay")
    if isinstance(replay, Mapping) and int(replay.get("record_count", -1)) != int(
        payload["replay_records"]
    ):
        raise RuntimeError(f"checkpoint replay record count disagrees with trainer state: {path}")
    return _validate_replay_state(replay, checkpoint_root=checkpoint_root, path=path)


def _validate_replay_state(
    replay: Any,
    *,
    checkpoint_root: Path,
    path: Path,
) -> set[Path]:
    if replay is None:
        return set()
    if not isinstance(replay, Mapping):
        raise RuntimeError(f"checkpoint replay state is not a mapping: {path}")
    try:
        record_count = int(replay.get("record_count", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"checkpoint replay record_count is invalid: {path}") from exc
    if record_count < 0:
        raise RuntimeError(f"checkpoint replay record_count is negative: {path}")
    storage = str(replay.get("storage", ""))
    if storage == "not_persisted":
        return set()
    if storage == "embedded_records":
        records = replay.get("records")
        if not isinstance(records, (list, tuple)) or len(records) != record_count:
            raise RuntimeError(f"checkpoint embedded replay is incomplete: {path}")
        return set()
    if storage != "shard_manifest":
        raise RuntimeError(f"checkpoint replay storage is unsupported: {storage!r}: {path}")

    references = replay.get("source_refs")
    if not isinstance(references, (list, tuple)) or len(references) != record_count:
        raise RuntimeError(f"checkpoint replay manifest is incomplete: {path}")
    by_shard: dict[Path, set[int]] = {}
    for position, reference in enumerate(references):
        if not isinstance(reference, (list, tuple)) or len(reference) != 2:
            raise RuntimeError(
                f"checkpoint replay reference {position} is malformed: {path}"
            )
        raw_path, raw_index = reference
        shard_path = Path(str(raw_path))
        if not shard_path.is_absolute():
            shard_path = checkpoint_root / shard_path
        shard_path = shard_path.resolve()
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise RuntimeError(
                f"checkpoint replay reference {position} has a non-integral index: {path}"
            )
        record_index = int(raw_index)
        if record_index < 0:
            raise RuntimeError(
                f"checkpoint replay reference {position} has a negative index: {path}"
            )
        by_shard.setdefault(shard_path, set()).add(record_index)

    for shard_path, indices in by_shard.items():
        if not shard_path.is_file():
            raise RuntimeError(
                f"checkpoint replay source shard does not exist: {shard_path}"
            )
        try:
            shard_payload = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"checkpoint replay source shard cannot be loaded: {shard_path}"
            ) from exc
        if not isinstance(shard_payload, Mapping):
            raise RuntimeError(
                f"checkpoint replay source shard is not a mapping: {shard_path}"
            )
        records = shard_payload.get("records")
        if not isinstance(records, (list, tuple)):
            raise RuntimeError(
                f"checkpoint replay source shard has no verifiable records: {shard_path}"
            )
        if indices and max(indices) >= len(records):
            raise RuntimeError(
                "checkpoint replay source index is out of range: "
                f"shard={shard_path}, max_index={max(indices)}, records={len(records)}"
            )
    return set(by_shard)


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only CRT descriptor.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def load_checkpoint(
    path: str | Path,
    *,
    config,
    model,
    behavior=None,
    world_optimizer=None,
    behavior_optimizer=None,
    world_scheduler=None,
    behavior_scheduler=None,
    trainer_state=None,
    replay=None,
    shard_store=None,
    restore_training_state: bool = False,
) -> CheckpointLoadReport:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    marker = payload.get("marker")
    model_version = payload.get("model_version")
    implementation_revision = payload.get("implementation_revision")
    if bool(config.checkpoint.reject_legacy_without_v4_marker) and marker != LEGACY_V4_CHECKPOINT_MARKER:
        raise RuntimeError(
            "Refusing to load an incompatible checkpoint without the expected "
            f"compatibility marker={LEGACY_V4_CHECKPOINT_MARKER!r}: {path}"
        )
    expected_revision = str(config.checkpoint.implementation_revision)
    if implementation_revision != expected_revision:
        raise RuntimeError(
            "Refusing checkpoint from a different UNSWAYED implementation revision: "
            f"checkpoint={implementation_revision!r}, required={expected_revision!r}. "
            "Engineering-hardening changes require fresh from0 training."
        )
    if model_version != str(config.checkpoint.model_version):
        raise RuntimeError(
            f"Refusing checkpoint model_version={model_version!r}; required={str(config.checkpoint.model_version)!r}"
        )
    schema = dict(payload.get("schema") or {})
    schema_version = schema.get("schema_version")
    if schema_version != str(config.replay.schema_version):
        raise RuntimeError(
            f"Refusing checkpoint with incompatible replay schema: checkpoint={schema_version}, config={config.replay.schema_version}"
        )
    runtime_schema = ReplaySchema.from_config(config)
    checkpoint_k = int(schema.get("counterfactual_K", 2))
    if checkpoint_k != int(runtime_schema.counterfactual_K):
        raise RuntimeError(
            "Refusing checkpoint with incompatible counterfactual K: "
            f"checkpoint={checkpoint_k}, runtime={runtime_schema.counterfactual_K}"
        )
    checkpoint_view_ids = tuple(map(str, schema.get("view_ids", ["B1", "B2"])))
    if checkpoint_view_ids != tuple(runtime_schema.view_ids):
        raise RuntimeError(
            "Refusing checkpoint with incompatible counterfactual view IDs: "
            f"checkpoint={checkpoint_view_ids}, runtime={runtime_schema.view_ids}"
        )
    metadata = dict(payload.get("counterfactual_metadata") or {})
    if checkpoint_k == 6:
        required_metadata = {
            "counterfactual_K",
            "view_ids",
            "temporal_protocol_version",
            "replay_schema_version",
            "background_stream_policy",
            "task_relevance_update_counter",
            "msir_enabled",
            "msir_training_enabled",
            "msir_target_mode",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise RuntimeError(f"K=6 checkpoint is missing counterfactual metadata: {missing_metadata}")
        if int(metadata["counterfactual_K"]) != int(runtime_schema.counterfactual_K):
            raise RuntimeError("K=6 checkpoint counterfactual metadata does not match runtime")
        runtime_msir = config.model.msir
        expected_msir = (
            bool(runtime_msir.enabled),
            bool(runtime_msir.training_enabled),
            str(runtime_msir.target_mode),
        )
        checkpoint_msir = (
            bool(metadata["msir_enabled"]),
            bool(metadata["msir_training_enabled"]),
            str(metadata["msir_target_mode"]),
        )
        if checkpoint_msir != expected_msir:
            raise RuntimeError(
                "Refusing K=6 checkpoint from the other E4/E5 arm: "
                f"checkpoint={checkpoint_msir}, runtime={expected_msir}"
            )
    checkpoint_contract = payload.get("temporal_contract")
    schema_contract = schema.get("temporal_contract")
    runtime_contract = str(config.replay.temporal_contract)
    if checkpoint_contract != runtime_contract or schema_contract != runtime_contract:
        raise RuntimeError(
            "Refusing checkpoint with incompatible temporal contract: "
            f"checkpoint={checkpoint_contract!r}, schema={schema_contract!r}, runtime={runtime_contract!r}"
        )
    if restore_training_state:
        _validate_training_payload(payload, config=config, behavior=behavior, world_optimizer=world_optimizer,
                                   behavior_optimizer=behavior_optimizer, world_scheduler=world_scheduler,
                                   behavior_scheduler=behavior_scheduler, trainer_state=trainer_state,
                                   replay=replay, shard_store=shard_store)

    strict = bool(config.checkpoint.strict_model_load)
    model_result = model.load_state_dict(payload["model"], strict=strict)
    behavior_loaded = False
    if behavior is not None and payload.get("behavior") is not None:
        behavior.load_state_dict(payload["behavior"], strict=strict)
        behavior_loaded = True

    world_optimizer_loaded = False
    behavior_optimizer_loaded = False
    world_scheduler_loaded = False
    behavior_scheduler_loaded = False
    replay_loaded = False
    shard_store_loaded = False
    counters_loaded = False
    slow_counter_loaded = False
    rng_loaded = False
    if restore_training_state:
        world_optimizer.load_state_dict(payload["world_optimizer"])
        _move_optimizer_state_to_parameter_device(world_optimizer)
        world_optimizer_loaded = True
        behavior_optimizer.load_state_dict(payload["behavior_optimizer"])
        _move_optimizer_state_to_parameter_device(behavior_optimizer)
        behavior_optimizer_loaded = True
        world_scheduler.load_state_dict(payload["world_scheduler"])
        behavior_scheduler.load_state_dict(payload["behavior_scheduler"])
        world_scheduler_loaded = True
        behavior_scheduler_loaded = True

        replay.load_state_dict(payload["replay"], shard_root=path.parent)
        replay_loaded = True
        shard_store.load_state_dict(payload["shard_store"])
        shard_store_loaded = True
        counters = dict(payload["trainer_state"])
        _restore_trainer_counters(trainer_state, counters)
        counters_loaded = True
        behavior.slow_value_updates = int(payload["slow_value_updates"])
        slow_counter_loaded = True
        restore_rng_state(payload["rng_state"])
        rng_loaded = True
    else:
        counters = dict(payload.get("trainer_state") or {})

    return CheckpointLoadReport(
        path=str(path.resolve()),
        step=int(payload.get("global_env_steps", payload.get("step", 0))),
        loaded=True,
        model_version=None if model_version is None else str(model_version),
        implementation_revision=None if implementation_revision is None else str(implementation_revision),
        schema_version=None if schema_version is None else str(schema_version),
        temporal_contract=None if checkpoint_contract is None else str(checkpoint_contract),
        temporal_contract_validated=True,
        missing_keys=list(getattr(model_result, "missing_keys", [])),
        unexpected_keys=list(getattr(model_result, "unexpected_keys", [])),
        optimizer_loaded=bool(world_optimizer_loaded and behavior_optimizer_loaded),
        behavior_loaded=behavior_loaded,
        world_optimizer_loaded=world_optimizer_loaded,
        behavior_optimizer_loaded=behavior_optimizer_loaded,
        world_scheduler_loaded=world_scheduler_loaded,
        behavior_scheduler_loaded=behavior_scheduler_loaded,
        replay_loaded=replay_loaded,
        shard_store_loaded=shard_store_loaded,
        counters_loaded=counters_loaded,
        slow_value_counter_loaded=slow_counter_loaded,
        rng_loaded=rng_loaded,
        training_state_restored=bool(restore_training_state),
        trainer_counters={key: int(value) for key, value in counters.items()},
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable at resume")
        torch.cuda.set_rng_state_all(cuda_state)


def write_checkpoint_report(path: str | Path, report: CheckpointLoadReport) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _trainer_counters(trainer_state, step: int | None, gradient_updates: int, replay_records: int) -> dict[str, int]:
    if trainer_state is not None:
        return {key: int(value) for key, value in vars(trainer_state).items()}
    return {
        "global_env_steps": int(step or 0),
        "gradient_updates": int(gradient_updates),
        "shards_collected": 0,
        "replay_records": int(replay_records),
        "policy_version": int(gradient_updates),
    }


def _validate_training_payload(payload, *, config, behavior, world_optimizer, behavior_optimizer, world_scheduler,
                               behavior_scheduler, trainer_state, replay, shard_store) -> None:
    required_objects = {
        "behavior": behavior,
        "world_optimizer": world_optimizer,
        "behavior_optimizer": behavior_optimizer,
        "world_scheduler": world_scheduler,
        "behavior_scheduler": behavior_scheduler,
        "trainer_state": trainer_state,
        "replay": replay,
        "shard_store": shard_store,
    }
    missing_objects = [name for name, value in required_objects.items() if value is None]
    if missing_objects:
        raise RuntimeError(f"Full training resume requires runtime objects: {missing_objects}")
    required_payload = [
        "behavior", "world_optimizer", "behavior_optimizer", "world_scheduler", "behavior_scheduler",
        "trainer_state", "slow_value_updates",
        "replay", "shard_store", "rng_state",
    ]
    missing_payload = [name for name in required_payload if payload.get(name) is None]
    if missing_payload:
        raise RuntimeError(f"Checkpoint is incomplete for full training resume: {missing_payload}")
    if bool(config.replay.persist_replay) and str(payload["replay"].get("storage")) == "not_persisted":
        raise RuntimeError("Formal resume requires persisted Replay, but checkpoint marks Replay as not_persisted")


def _restore_trainer_counters(target, counters: dict[str, Any]) -> None:
    required = {"global_env_steps", "gradient_updates", "shards_collected", "replay_records", "policy_version"}
    missing = sorted(required - set(counters))
    if missing:
        raise RuntimeError(f"Checkpoint trainer_state is missing counters: {missing}")
    for key in required:
        setattr(target, key, int(counters[key]))


def _move_optimizer_state_to_parameter_device(optimizer) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=param.device)


def _counterfactual_metadata(config, model) -> dict[str, Any]:
    schema = ReplaySchema.from_config(config)
    estimator = getattr(model, "task_relevance", None)
    metadata = {
        "counterfactual_K": int(schema.counterfactual_K),
        "view_ids": list(schema.view_ids),
        "temporal_protocol_version": schema.temporal_protocol_version,
        "replay_schema_version": schema.schema_version,
        "pairing_contract": schema.pairing_contract,
        "background_stream_policy": schema.background_stream_policy,
        "task_relevance_ema_valid": bool(getattr(estimator, "ema_valid", False)),
        "task_relevance_update_counter": int(getattr(estimator, "update_counter", torch.zeros(())).item()),
        "psab_state": "stateless",
        "msir_enabled": bool(config.model.msir.enabled),
        "msir_training_enabled": bool(config.model.msir.get("training_enabled", config.model.msir.enabled)),
        "msir_target_mode": str(config.model.msir.get("target_mode", "legacy")),
    }
    capc = config.get("capc", {})
    metadata.update(
        {
            "capc_enabled": bool(capc.get("enabled", False)),
            "capc_implementation": str(capc.get("implementation", "disabled")),
            "capc_formula": str(capc.get("formula", "disabled")),
            "capc_probability_semantics": str(capc.get("probability_semantics", "disabled")),
            "capc_target_stopgrad": bool(capc.get("target_stopgrad", False)),
            "capc_weight": float(capc.get("weight", 0.0)),
        }
    )
    return metadata
