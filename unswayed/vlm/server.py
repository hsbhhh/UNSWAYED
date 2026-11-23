from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..vcfa.grid import compress_feature_dim
from .protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    STAT_COMPLETED,
    STAT_DROPPED,
    STAT_DROPPED_INTERMEDIATE_FRAMES,
    STAT_DUPLICATE_FRAME_SKIP,
    STAT_INFERENCE,
    STAT_EXACT_FRAME_MATCH,
    STAT_EXACT_VIEW_MATCH,
    STAT_CROSS_VIEW_CONTAMINATION,
    STAT_CROSS_VIEW_MISMATCH,
    STAT_FRAME_GAP,
    STAT_INFERENCE_FAILURE,
    STAT_INFERENCE_BATCH_COUNT,
    STAT_INFERENCE_BATCH_MAX,
    STAT_INFERENCE_BATCH_TOTAL,
    STAT_OLD_EPISODE_RESULT_REJECTED,
    STAT_OUT_OF_ORDER,
    STAT_REQUEST_WAIT_MS_MAX,
    STAT_REQUEST_WAIT_MS_TOTAL,
    STAT_SERVER_LOOPS,
    VLMSharedMemory,
    VLMSharedSpec,
    ensure_ready_file,
    publish_semantic_snapshot,
    write_heartbeat,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Persistent async VLM server for DreamerV3 external_async mode.")
    parser.add_argument("--shared-memory-name", required=True)
    parser.add_argument("--env-num", type=int, required=True)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--input-height", type=int, default=64)
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--q-sem-only", action="store_true", help="Use UNSWAYED dense patch-score/patch-feature protocol.")
    parser.add_argument("--disable-reference-features", action="store_true", help="Accepted for compatibility; UNSWAYED never publishes reference features.")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--internvideo2-backend",
        choices=["naive", "native_flash"],
        default="naive",
        help="Select the reversible InternVideo2 vision backend; native_flash enables FlashAttention, FusedMLP, and fused RMSNorm together.",
    )
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--fine-turn-root", type=Path, default=None)
    parser.add_argument("--get-data-root", type=Path, default=None)
    parser.add_argument("--heartbeat-path", type=Path, default=None)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.001)
    parser.add_argument("--max-batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--strict-per-frame", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.logdir.mkdir(parents=True, exist_ok=True)
    ready_path = args.heartbeat_path or (args.logdir / "vlm_server_ready.json")
    stop_path = args.logdir / "vlm_server.stop"
    spec = VLMSharedSpec(
        name=str(args.shared_memory_name),
        env_num=int(args.env_num),
        clip_len=int(args.clip_len),
        height=int(args.input_height),
        width=int(args.input_width),
        grid_size=int(args.grid_size),
        feature_dim=int(args.feature_dim),
        view_num=int(args.view_num),
    )
    shared = VLMSharedMemory(spec, create=True)
    try:
        teacher = _build_teacher(args, spec)
        teacher.eval()
        parameters = _teacher_parameters(teacher)
        for param in parameters:
            param.requires_grad_(False)
        param_count = sum(int(p.numel()) for p in parameters)
        dtype = next((p.dtype for p in parameters), torch.float32)
        device = next((p.device for p in parameters), torch.device(args.device if torch.cuda.is_available() else "cpu"))
        teacher_load_report = dict(getattr(teacher, "load_report", {}) or {})
        backend_report = dict(teacher_load_report.get("internvideo2_backend_report", {}) or {})
        actual_backend = "mock" if bool(args.mock) else str(
            teacher_load_report.get("internvideo2_backend_actual", backend_report.get("actual_backend", "naive"))
        )
        if not bool(args.mock) and actual_backend != str(args.internvideo2_backend):
            raise RuntimeError(
                "Resident VLM backend mismatch: "
                f"requested={args.internvideo2_backend}, actual={actual_backend}"
            )
        ready = {
            "status": "ready",
            "schema_version": PROTOCOL_VERSION,
            "shared_memory_name": spec.name,
            "env_num": spec.env_num,
            "view_num": spec.view_num,
            "stream_key": "(env_id, view_id)",
            "clip_len": spec.clip_len,
            "input_shape": [1, 3, spec.clip_len, spec.height, spec.width],
            "grid_size": spec.grid_size,
            "q_sem_only": True,
            "reference_feature_enabled": False,
            "reference_feature_bytes": 0,
            "feature_dim": spec.feature_dim,
            "micro_batch_size": int(args.micro_batch_size),
            "max_batch_wait_ms": float(args.max_batch_wait_ms),
            "strict_per_frame": bool(args.strict_per_frame),
            "param_count": param_count,
            "vlm_model_copies": 1,
            "shared_memory_bytes": int(sum(len(block.buf) for block in shared.blocks.values())),
            "clip_buffer_bytes": int(shared.arrays["frames"].nbytes),
            "model_dtype": str(dtype),
            "model_device": str(device),
            "checkpoint": str(args.checkpoint),
            "python_executable": sys.executable,
            "internvideo2_backend_requested": str(args.internvideo2_backend),
            "internvideo2_backend_actual": actual_backend,
            "internvideo2_backend_report": backend_report,
        }
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            ready["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            ready["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
        print(json.dumps(ready, indent=2), flush=True)
        ensure_ready_file(ready_path, ready)
        write_heartbeat(shared)
        first_shape_logged = False
        first_feature_adapter_logged = False
        latencies = []
        last_processed_generation = np.zeros((spec.env_num, spec.view_num), dtype=np.int64) - 1
        last_processed_episode = np.zeros((spec.env_num, spec.view_num), dtype=np.int64) - 1
        # Kept as zero-valued wire-compatibility fields. UNSWAYED does not calculate
        # or gate execution on checkpoint/preprocess hashes.
        checkpoint_hash = 0
        preprocess_hash = 0
        while not stop_path.exists():
            shared.arrays["stats"][STAT_SERVER_LOOPS] += 1
            write_heartbeat(shared)
            pending = _pending_stream_ids(shared, spec)
            if not pending:
                time.sleep(float(args.poll_sleep_sec))
                continue
            while pending:
                stream_ids = _collect_micro_batch(
                    shared,
                    spec,
                    max_batch=max(1, int(args.micro_batch_size)),
                    max_wait_ms=float(args.max_batch_wait_ms),
                )
                if not stream_ids:
                    break
                pending = [stream for stream in pending if stream not in set(stream_ids)]
                clips, snapshots = [], []
                for env_id, view_id in stream_ids:
                    frame = shared.read_consistent_frame_snapshot(env_id, view_id)
                    if frame is None:
                        continue
                    _verify_pending_request(shared, frame, strict=bool(args.strict_per_frame))
                    if (
                        int(frame.generation_id) == int(last_processed_generation[env_id, view_id])
                        and int(frame.episode_id) == int(last_processed_episode[env_id, view_id])
                    ):
                        shared.arrays["stats"][STAT_DUPLICATE_FRAME_SKIP] += 1
                        if bool(args.strict_per_frame):
                            raise RuntimeError(
                                f"Duplicate strict VLM request for env_id={env_id}, view_id={view_id}, frame={frame}"
                            )
                        if int(shared.arrays["request_frame_generation_id"][env_id, view_id]) == int(
                            frame.generation_id
                        ):
                            shared.arrays["request_pending"][env_id, view_id] = 0
                        continue
                    clips.append(frame.clip)
                    snapshots.append(frame)
                if not clips:
                    continue
                batch = torch.as_tensor(np.stack(clips), dtype=torch.uint8)
                inference_start_ns = int(time.time_ns())
                t0 = time.perf_counter()
                try:
                    with torch.inference_mode():
                        out = teacher.forward_latest_clip(batch.to(args.device if torch.cuda.is_available() else "cpu"))
                    shared.arrays["stats"][STAT_INFERENCE] += int(len(snapshots))
                except Exception:
                    shared.arrays["stats"][STAT_INFERENCE_FAILURE] += int(len(snapshots))
                    raise
                shared.arrays["stats"][STAT_INFERENCE_BATCH_TOTAL] += int(len(snapshots))
                shared.arrays["stats"][STAT_INFERENCE_BATCH_COUNT] += 1
                shared.arrays["stats"][STAT_INFERENCE_BATCH_MAX] = max(
                    int(shared.arrays["stats"][STAT_INFERENCE_BATCH_MAX]), int(len(snapshots))
                )
                inference_end_ns = int(time.time_ns())
                latency = time.perf_counter() - t0
                latencies.append(latency / max(1, len(clips)))
                if not first_shape_logged:
                    trace = out.trace or {}
                    print(
                        "VLM input shape:",
                        trace.get("model_input_shape", "<unknown>"),
                        "latest_clip_shape:",
                        trace.get("latest_clip_shape", list(batch.shape)),
                        flush=True,
                    )
                    first_shape_logged = True
                q_sem = out.q_sem[:, 0].detach().float().cpu()
                patch_features = out.patch_features[:, 0].detach().float().cpu()
                patch_features, feature_adapter_report = _adapt_patch_features_to_spec(patch_features, spec.feature_dim)
                if feature_adapter_report and not first_feature_adapter_logged:
                    print("[VLM_FEATURE_ADAPTER] " + json.dumps(feature_adapter_report, sort_keys=True), flush=True)
                    first_feature_adapter_logged = True
                _write_outputs(
                    shared,
                    snapshots,
                    q_sem,
                    patch_features,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                    checkpoint_hash=checkpoint_hash,
                    preprocess_hash=preprocess_hash,
                    last_processed_generation=last_processed_generation,
                    last_processed_episode=last_processed_episode,
                    strict=bool(args.strict_per_frame),
                )
            if len(latencies) and len(latencies) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "completed": int(shared.arrays["stats"][STAT_COMPLETED]),
                            "dropped": int(shared.arrays["stats"][STAT_DROPPED]),
                            "duplicate_frame_skip": int(shared.arrays["stats"][STAT_DUPLICATE_FRAME_SKIP]),
                            "dropped_intermediate_frames": int(shared.arrays["stats"][STAT_DROPPED_INTERMEDIATE_FRAMES]),
                            "avg_latency_s": float(np.mean(latencies[-100:])),
                            "clips_per_sec": float(1.0 / max(np.mean(latencies[-100:]), 1e-9)),
                            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                        }
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("VLM server interrupted.", flush=True)
    finally:
        shared.close(unlink=True)
        if ready_path.exists():
            ready_path.unlink(missing_ok=True)
    return 0


def _build_teacher(args, spec: VLMSharedSpec):
    _ensure_local_teacher_path()
    from .teacher import build_semantic_teacher

    config = {
        "tokens": {"grid_height": spec.grid_size, "grid_width": spec.grid_size},
        "model": {"vcfa": {"enabled": True, "vlm_grid_size": spec.grid_size}},
        "semantic": {
            "mode": "finetune_v2_internvideo2_online",
            "checkpoint": str(args.checkpoint),
            "base_checkpoint": None if args.base_checkpoint is None else str(args.base_checkpoint),
            "fine_turn_root": None if args.fine_turn_root is None else str(args.fine_turn_root),
            "mock": bool(args.mock),
            "mock_patch_feature_dim": spec.feature_dim,
            "clip_length": spec.clip_len,
            "current_frame_output": "last_temporal_slice",
            "inference_dtype": str(args.dtype),
            "internvideo2_backend": str(args.internvideo2_backend),
            "print_load_report": True,
            "task_name": args.task_name,
        },
        "device": {"vlm": str(args.device), "train": str(args.device)},
    }
    resources = SimpleNamespace(
        get_data_root=args.get_data_root,
        vlm_finetune_v2_checkpoint=args.checkpoint,
        vlm_checkpoint=None,
        internvideo2_base_checkpoint=args.base_checkpoint,
        internvideo2_weights_root=None,
    )
    teacher = build_semantic_teacher(resources, config)
    if hasattr(teacher, "load"):
        teacher.load()
    return teacher


def _pending_stream_ids(shared: VLMSharedMemory, spec: VLMSharedSpec) -> list[tuple[int, int]]:
    pending = np.argwhere(
        shared.arrays["request_pending"][: spec.env_num, : spec.view_num] > 0
    )
    return [(int(env_id), int(view_id)) for env_id, view_id in pending.tolist()]


def _collect_micro_batch(
    shared: VLMSharedMemory,
    spec: VLMSharedSpec,
    *,
    max_batch: int,
    max_wait_ms: float,
) -> list[tuple[int, int]]:
    deadline = time.perf_counter() + max(0.0, float(max_wait_ms)) / 1000.0
    while True:
        pending = _pending_stream_ids(shared, spec)
        if pending and (len(pending) >= max(1, int(max_batch)) or time.perf_counter() >= deadline):
            return pending[: max(1, int(max_batch))]
        if pending and max_wait_ms <= 0:
            return pending[: max(1, int(max_batch))]
        if time.perf_counter() >= deadline:
            return pending[: max(1, int(max_batch))]
        time.sleep(min(0.0005, max(0.0, deadline - time.perf_counter())))


def _verify_pending_request(shared: VLMSharedMemory, frame, *, strict: bool) -> None:
    arrays = shared.arrays
    env_id = int(frame.env_id)
    view_id = int(frame.view_id)
    req_env = int(arrays["request_env_id"][env_id, view_id])
    req_view = int(arrays["request_view_id"][env_id, view_id])
    req_episode = int(arrays["request_episode_id"][env_id, view_id])
    req_frame = int(arrays["request_frame_id"][env_id, view_id])
    req_generation = int(arrays["request_frame_generation_id"][env_id, view_id])
    if req_env != env_id or req_view != view_id:
        arrays["stats"][STAT_CROSS_VIEW_MISMATCH] += int(req_view != view_id)
        arrays["stats"][STAT_OUT_OF_ORDER] += 1
        if strict:
            raise RuntimeError(
                "VLM request stream mismatch "
                f"slot=({env_id},{view_id}), request=({req_env},{req_view})"
            )
    contamination = int(np.count_nonzero(np.asarray(frame.slot_view_ids) != view_id))
    if contamination:
        arrays["stats"][STAT_CROSS_VIEW_CONTAMINATION] += contamination
        if strict:
            raise RuntimeError(
                "VLM temporal clip contains cross-view frames: "
                f"env_id={env_id}, view_id={view_id}, slot_view_ids={frame.slot_view_ids.tolist()}"
            )
    if req_episode != int(frame.episode_id):
        arrays["stats"][STAT_OLD_EPISODE_RESULT_REJECTED] += 1
        if strict:
            raise RuntimeError(
                f"VLM request/frame episode mismatch env_id={env_id}, view_id={view_id}: "
                f"request_episode={req_episode}, frame_episode={frame.episode_id}"
            )
    if req_frame != int(frame.frame_id) or req_generation != int(frame.generation_id):
        arrays["stats"][STAT_OUT_OF_ORDER] += 1
        if strict:
            raise RuntimeError(
                "VLM request/frame mismatch "
                f"env_id={env_id}, view_id={view_id}: request_frame={req_frame}, frame_id={frame.frame_id}, "
                f"request_generation={req_generation}, frame_generation={frame.generation_id}"
            )


def _write_outputs(
    shared: VLMSharedMemory,
    snapshots,
    q_sem,
    patch_features,
    *,
    inference_start_ns: int,
    inference_end_ns: int,
    checkpoint_hash: int,
    preprocess_hash: int,
    last_processed_generation: np.ndarray,
    last_processed_episode: np.ndarray,
    strict: bool,
) -> None:
    arrays = shared.arrays
    for i, frame in enumerate(snapshots[: q_sem.shape[0]]):
        env_id = int(frame.env_id)
        view_id = int(frame.view_id)
        request_id = int(arrays["request_id"][env_id, view_id])
        request_env_id = int(arrays["request_env_id"][env_id, view_id])
        request_view_id = int(arrays["request_view_id"][env_id, view_id])
        request_frame_id = int(arrays["request_frame_id"][env_id, view_id])
        request_generation = int(arrays["request_frame_generation_id"][env_id, view_id])
        request_episode = int(arrays["request_episode_id"][env_id, view_id])
        if int(arrays["episode_version"][env_id, view_id]) != int(frame.episode_id):
            arrays["stats"][STAT_OLD_EPISODE_RESULT_REJECTED] += 1
            if strict:
                raise RuntimeError(
                    f"Stale VLM frame reached publisher env_id={env_id}, view_id={view_id}: "
                    f"current_episode={int(arrays['episode_version'][env_id, view_id])}, "
                    f"frame_episode={frame.episode_id}"
                )
            arrays["stats"][STAT_DROPPED] += 1
            continue
        exact = (
            int(frame.frame_id) == request_frame_id
            and int(frame.generation_id) == request_generation
            and int(frame.episode_id) == request_episode
            and int(frame.env_id) == request_env_id
            and int(frame.view_id) == request_view_id
            and request_id >= 0
        )
        if not exact:
            arrays["stats"][STAT_OUT_OF_ORDER] += 1
            if strict:
                raise RuntimeError(
                    "Refusing to publish non-exact VLM result: "
                    f"env_id={env_id}, view_id={view_id}, request_id={request_id}, "
                    f"request_stream=({request_env_id},{request_view_id}), request_frame={request_frame_id}, "
                    f"frame={frame.frame_id}, request_generation={request_generation}, frame_generation={frame.generation_id}, "
                    f"request_episode={request_episode}, frame_episode={frame.episode_id}"
                )
        generation = publish_semantic_snapshot(
            shared,
            env_id,
            view_id,
            frame,
            q_sem[i].reshape(arrays["semantic_map"][env_id, view_id].shape).numpy(),
            patch_features[i].reshape(arrays["patch_features"][env_id, view_id].shape).numpy(),
            inference_start_ns=int(inference_start_ns),
            inference_end_ns=int(inference_end_ns),
            checkpoint_hash=int(checkpoint_hash),
            preprocess_hash=int(preprocess_hash),
            request_id=request_id,
        )
        if exact:
            arrays["stats"][STAT_EXACT_FRAME_MATCH] += 1
            arrays["stats"][STAT_EXACT_VIEW_MATCH] += 1
            previous_episode = int(arrays["last_completed_episode_id"][env_id, view_id])
            previous_frame = int(arrays["last_completed_frame_id"][env_id, view_id])
            if previous_episode == int(frame.episode_id) and previous_frame >= 0 and int(frame.frame_id) != previous_frame + 1:
                arrays["stats"][STAT_FRAME_GAP] += 1
            arrays["last_completed_episode_id"][env_id, view_id] = int(frame.episode_id)
            arrays["last_completed_frame_id"][env_id, view_id] = int(frame.frame_id)
            wait_ms = max(
                0.0,
                (int(inference_end_ns) - int(arrays["request_timestamp_ns"][env_id, view_id])) / 1e6,
            )
            wait_ms_i = int(round(wait_ms))
            arrays["stats"][STAT_REQUEST_WAIT_MS_TOTAL] += wait_ms_i
            arrays["stats"][STAT_REQUEST_WAIT_MS_MAX] = max(int(arrays["stats"][STAT_REQUEST_WAIT_MS_MAX]), wait_ms_i)
            arrays["request_pending"][env_id, view_id] = 0
        else:
            arrays["stats"][STAT_DROPPED_INTERMEDIATE_FRAMES] += 1
            if strict:
                raise RuntimeError(
                    f"Strict VLM result was not exact for env_id={env_id}, view_id={view_id}."
                )
        last_processed_generation[env_id, view_id] = int(frame.generation_id)
        last_processed_episode[env_id, view_id] = int(frame.episode_id)
        if generation <= 3 or generation % 100 == 0:
            q_np = q_sem[i].detach().float().numpy()
            print(
                "[VLM_PUBLISH] "
                f"env_id={env_id} view_id={view_id} episode_id={frame.episode_id} "
                f"source_frame_id={frame.frame_id} source_frame_generation={frame.generation_id} "
                f"semantic_generation={generation} q_mean={float(q_np.mean()):.6f} "
                f"q_max={float(q_np.max()):.6f} q_p90={float(np.percentile(q_np, 90)):.6f} "
                f"inference_ms={(int(inference_end_ns) - int(inference_start_ns)) / 1e6:.3f}",
                flush=True,
            )


def _ensure_local_teacher_path() -> None:
    return None


def _adapt_patch_features_to_spec(patch_features: torch.Tensor, target_dim: int) -> tuple[torch.Tensor, dict[str, object] | None]:
    """Adapt frozen VLM patch features using the migrated VCFA compression rule."""

    features = torch.as_tensor(patch_features).float()
    if features.ndim != 3:
        raise RuntimeError(f"expected dense patch features [B,N,D], got {tuple(features.shape)}")
    target_dim = int(target_dim)
    native_dim = int(features.shape[-1])
    report = {
        "native_shape": list(features.shape),
        "native_dim": native_dim,
        "target_dim": target_dim,
        "learned_parameters": 0,
    }
    if native_dim == target_dim:
        report["method"] = "identity"
        return features.contiguous(), report
    adapted = compress_feature_dim(features, target_dim)
    report["method"] = "vcfa.compress_feature_dim"
    report["source"] = "v4-owned VCFA grid compression with verified V2 numeric parity"
    return adapted.contiguous(), report


def _teacher_parameters(teacher):
    parameters = list(teacher.parameters())
    inner = getattr(teacher, "model", None)
    if not parameters and inner is not None:
        parameters = list(inner.parameters())
    return parameters


if __name__ == "__main__":
    raise SystemExit(main())
