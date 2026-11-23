from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigNode(dict):
    """Small dict-with-attributes wrapper for production configuration."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self) -> "ConfigNode":
        return to_config_node(dict(self))


FORBIDDEN_CONFIG_KEYS = {
    "legacy_qpred",
    "legacy_canonical_reconstruction",
    "teacher_target",
    "qpred",
    "reconstruction",
    "sid",
    "rssm_backend",
}


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ConfigNode:
    path = Path(path)
    data = _load_yaml_with_base(path, seen=set())
    if not isinstance(data, Mapping):
        raise ValueError(f"V5 config must be a mapping: {path}")
    merged = deepcopy(dict(data))
    for key, value in (overrides or {}).items():
        set_by_dotted_key(merged, key, value)
    _resolve_ablation_config(merged)
    _resolve_k_view_config(merged)
    config = to_config_node(merged)
    validate_config(config)
    return config


def _resolve_k_view_config(config: dict[str, Any]) -> None:
    """Resolve every K-view consumer from the single ``model.num_views`` source.

    Legacy configs do not define this field and are intentionally left alone.
    New K-view configs may change K through one dotted override without editing
    the replay, VLM, semantic weighting, VGC, or MSIR sections separately.
    """

    model = config.get("model")
    if not isinstance(model, dict) or "num_views" not in model:
        return
    count = int(model["num_views"])
    if count not in {2, 4, 6}:
        raise ValueError(f"model.num_views must be 2, 4, or 6, got {count}")
    model.setdefault("semantic_weighting", {})["num_views"] = count
    model.setdefault("vlm_gaussian_consensus", {})["num_views"] = count
    model.setdefault("msir", {})["num_views"] = count
    model["num_views_source"] = "model.num_views"

    external_vlm = config.setdefault("external_vlm", {})
    external_vlm["view_num"] = count
    external_vlm["view_num_source"] = "model.num_views"

    counterfactual = config.setdefault("counterfactual", {})
    counterfactual["views"] = count
    counterfactual["view_ids"] = [f"view_{index}" for index in range(count)]
    counterfactual["views_source"] = "model.num_views"

    replay = config.setdefault("replay", {})
    replay["schema_version"] = "v4_kview_v2"
    replay["require_b1_b2_raw_pixels"] = False
    replay["require_counterfactual_raw_pixels"] = True
    replay["num_views_source"] = "model.num_views"


def _resolve_ablation_config(config: dict[str, Any]) -> None:
    """Map the six public ablation switches onto unchanged model/loss fields.

    The model continues to construct every production head.  These switches only
    control whether the corresponding already-computed loss is aggregated.
    """

    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping):
        return
    model = config.setdefault("model", {})
    vgc = model.setdefault("vlm_gaussian_consensus", {})
    msir = model.setdefault("msir", {})
    capc = config.setdefault("capc", {})
    if "use_vgc" in ablation:
        vgc["enabled"] = bool(ablation["use_vgc"])
    if "use_msir" in ablation:
        msir["enabled"] = bool(ablation["use_msir"])
        msir["training_enabled"] = bool(ablation["use_msir"])
    if "use_capc" in ablation:
        capc["enabled"] = bool(ablation["use_capc"])
    if "lambda_vgc" in ablation:
        vgc["weight"] = float(ablation["lambda_vgc"])
    if "lambda_msir" in ablation:
        msir["weight"] = float(ablation["lambda_msir"])
    if "lambda_capc" in ablation:
        capc["weight"] = float(ablation["lambda_capc"])


def _load_yaml_with_base(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ValueError(f"cyclic V4 config inheritance detected at {resolved}")
    seen.add(resolved)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"V5 config must be a mapping: {resolved}")
    data = deepcopy(dict(data))
    base = data.pop("_base_", None)
    if base is None:
        seen.remove(resolved)
        return data
    base_path = Path(str(base))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    merged = _deep_merge(_load_yaml_with_base(base_path, seen=seen), data)
    seen.remove(resolved)
    return merged


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def to_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, Mapping):
        return ConfigNode({str(k): to_config_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    return value


def to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def set_by_dotted_key(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = str(dotted).split(".")
    current = mapping
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"cannot override non-mapping config path {dotted!r}")
    current[parts[-1]] = value


def validate_config(config: ConfigNode) -> None:
    violations = []
    _find_forbidden_keys(config, path="", violations=violations)
    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"V4 config contains legacy/forbidden keys: {joined}")
    horizons = list(config.model.vlm_gaussian_consensus.prediction_horizons)
    if horizons != [0]:
        raise ValueError(f"V4 only supports K=0 prediction_horizons, got {horizons}")
    calibration = config.model.semantic_score_calibration
    if calibration.mode != "identity_probability":
        raise ValueError("V4 semantic score calibration must be identity_probability")
    if bool(calibration.apply_extra_sigmoid) or bool(calibration.per_frame_minmax):
        raise ValueError("V4 forbids extra sigmoid and per-frame min-max score calibration")
    if not bool(config.model.world_model_loss.enabled):
        raise ValueError("V4 formal training requires model.world_model_loss.enabled=true")
    if not bool(config.behavior.enabled):
        raise ValueError("V4 formal training requires behavior.enabled=true")
    if not bool(config.external_vlm.enabled):
        raise ValueError("V4 formal training requires external_vlm.enabled=true")
    if not bool(config.external_vlm.q_sem_only):
        raise ValueError("V4 external_vlm.q_sem_only must be true")
    if bool(config.external_vlm.reference_feature_enabled):
        raise ValueError("V4 forbids reference feature shared-memory path")
    if not bool(config.external_vlm.dense_ordered_patch_features):
        raise ValueError("V4 requires dense ordered VLM patch features")
    if int(config.external_vlm.grid_size) ** 2 != int(config.external_vlm.patch_count):
        raise ValueError("external_vlm.patch_count must equal grid_size^2")
    if int(config.model.vlm_gaussian_consensus.patch_count) != int(config.external_vlm.patch_count):
        raise ValueError("VGC patch_count must match external_vlm.patch_count")
    if int(config.model.vlm_gaussian_consensus.patch_feature_dim) != int(config.external_vlm.feature_dim):
        raise ValueError("VGC patch_feature_dim must match external_vlm.feature_dim")
    counterfactual = config.get("counterfactual")
    counterfactual_k = int(counterfactual.get("views", 2)) if counterfactual else 2
    if counterfactual_k not in {2, 4, 6}:
        raise ValueError(f"counterfactual.views must be 2, 4, or 6, got {counterfactual_k}")
    view_ids = list(counterfactual.get("view_ids", [f"B{i + 1}" for i in range(counterfactual_k)])) if counterfactual else ["B1", "B2"]
    if len(view_ids) != counterfactual_k or len(set(map(str, view_ids))) != counterfactual_k:
        raise ValueError("counterfactual.view_ids must contain one unique label per counterfactual view")
    if int(config.external_vlm.get("view_num", 2)) != counterfactual_k:
        raise ValueError("external_vlm.view_num must equal counterfactual.views")
    formal_k6 = bool(counterfactual and counterfactual.get("enabled", False))
    if formal_k6:
        if counterfactual_k != 6:
            raise ValueError("formal counterfactual E4/E5 requires counterfactual.views=6")
        if str(config.replay.schema_version) != "v4_counterfactual_k6_v2":
            raise ValueError("formal K=6 requires replay.schema_version=v4_counterfactual_k6_v2")
        if bool(config.model.vlm_gaussian_consensus.enabled):
            raise ValueError("formal K=6 must disable old Gaussian/tau VGC")
        if str(counterfactual.center.mode) != "geometric_median":
            raise ValueError("formal E4/E5 center.mode must be geometric_median")
        if str(counterfactual.nuisance.operator) != "soft_attenuation":
            raise ValueError("formal E4/E5 nuisance.operator must be soft_attenuation")
        linalg_chunk_size = int(counterfactual.nuisance.get("linalg_chunk_size", 0))
        if not 1 <= linalg_chunk_size <= 32768:
            raise ValueError("formal E4/E5 nuisance.linalg_chunk_size must be in [1, 32768]")
        if str(counterfactual.task_relevance.mode) not in {"exact", "periodic"}:
            raise ValueError("task_relevance.mode must be exact or periodic")
        if not bool(counterfactual.psab.enabled):
            raise ValueError("formal E4/E5 requires PSAB")
        if not bool(config.replay.get("require_counterfactual_raw_pixels", False)):
            raise ValueError("formal K=6 Replay must require raw_b_views")
        if bool(config.model.msir.enabled):
            if str(config.model.msir.target_mode) != "counterfactual_medoid":
                raise ValueError("E4 MSIR target_mode must be counterfactual_medoid")
            if not bool(config.model.msir.training_enabled):
                raise ValueError("enabled E4 MSIR must have training_enabled=true")
        else:
            if str(config.model.msir.target_mode) != "disabled" or bool(config.model.msir.training_enabled):
                raise ValueError("E5 must set MSIR target_mode=disabled and training_enabled=false")
        if list(map(float, config.model.msir.level_weights)) != [1.0, 0.5, 0.25]:
            raise ValueError("E4/E5 keep fixed MSIR level weights [1.0, 0.5, 0.25]")
        if abs(float(config.training.lr) - 4.0e-5) > 1.0e-12:
            raise ValueError("formal E4/E5 world LR is fixed at the corrected V4 value 4e-5")
        if abs(float(config.behavior.lr) - 4.0e-5) > 1.0e-12:
            raise ValueError("formal E4/E5 behavior LR is fixed at the corrected V4 value 4e-5")
    semantic_weighting = config.model.get("semantic_weighting", {"enabled": False, "mode": "legacy_precision"})
    robust_weighting = bool(semantic_weighting.get("enabled", False))
    if robust_weighting:
        semantic_mode = str(semantic_weighting.get("mode"))
        if semantic_mode not in {"robust_mad_sigmoid", "standard_softmax"}:
            raise ValueError("enabled semantic weighting requires mode=robust_mad_sigmoid or standard_softmax")
        if "num_views" not in config.model:
            raise ValueError("K-view semantic weighting requires model.num_views as its single source of truth")
        canonical_k = int(config.model.num_views)
        if canonical_k < 2 or canonical_k not in {2, 4, 6}:
            raise ValueError("K-view semantic weighting requires model.num_views in {2,4,6}")
        derived_k = {
            "counterfactual.views": counterfactual_k,
            "external_vlm.view_num": int(config.external_vlm.view_num),
            "semantic_weighting.num_views": int(semantic_weighting.get("num_views", 0)),
            "vlm_gaussian_consensus.num_views": int(config.model.vlm_gaussian_consensus.get("num_views", 0)),
            "msir.num_views": int(config.model.msir.get("num_views", 0)),
        }
        mismatched = {key: value for key, value in derived_k.items() if value != canonical_k}
        if mismatched:
            raise ValueError(f"all K-view consumers must resolve from model.num_views={canonical_k}: {mismatched}")
        if formal_k6:
            raise ValueError("K-view robust weighting is separate from the legacy K=6 E4/E5 path")
        if str(config.replay.schema_version) != "v4_kview_v2":
            raise ValueError("K-view semantic weighting requires replay.schema_version=v4_kview_v2")
        if str(semantic_weighting.get("aggregate")) != "mean":
            raise ValueError("K-view semantic weighting requires aggregate=mean")
        if not bool(semantic_weighting.get("mass_normalize", False)):
            raise ValueError("K-view semantic weighting requires mass_normalize=true")
        if semantic_mode == "robust_mad_sigmoid":
            if float(semantic_weighting.get("beta", 0.0)) <= 0.0:
                raise ValueError("semantic weighting beta must be positive")
            if float(semantic_weighting.get("mad_scale", 0.0)) <= 0.0:
                raise ValueError("semantic weighting mad_scale must be positive")
        if semantic_mode == "standard_softmax" and float(semantic_weighting.get("temperature", 0.0)) <= 0.0:
            raise ValueError("standard softmax semantic weighting temperature must be positive")
        if float(semantic_weighting.get("eps", 0.0)) <= 0.0:
            raise ValueError("semantic weighting eps must be positive")
        if not bool(config.replay.get("require_counterfactual_raw_pixels", False)):
            raise ValueError("K-view semantic weighting Replay must require raw_b_views")
        for loss_name in ("vlm_gaussian_consensus", "msir"):
            loss_cfg = config.model[loss_name]
            if float(loss_cfg.get("base_variance", 0.0)) <= 0.0:
                raise ValueError(f"K-view robust {loss_name} requires positive base_variance")
    if str(config.replay.temporal_contract) != "obs_t_with_incoming_action":
        raise ValueError("V4 replay.temporal_contract must be obs_t_with_incoming_action")
    if (
        counterfactual_k == 2
        and not formal_k6
        and not robust_weighting
        and not bool(config.replay.require_b1_b2_raw_pixels)
    ):
        raise ValueError("V4 MSIR requires Replay raw_b1/raw_b2 pixels")
    if str(config.external_vlm.backend) not in {"mock", "resident"}:
        raise ValueError("external_vlm.backend must be mock or resident")
    if str(config.external_vlm.get("internvideo2_backend", "naive")) not in {"naive", "native_flash"}:
        raise ValueError("external_vlm.internvideo2_backend must be naive or native_flash")
    if str(config.env.backend) not in {"synthetic", "dmc"}:
        raise ValueError("env.backend must be synthetic or dmc")
    if str(config.behavior.policy_backend) not in {"dreamer", "random", "mock"}:
        raise ValueError("behavior.policy_backend must be dreamer, random, or mock")
    if int(config.behavior.actor_layers) != 3 or abs(float(config.behavior.actor_outscale) - 0.01) > 1.0e-12:
        raise ValueError("V4 formal actor must use three hidden layers and outscale=0.01")
    if int(config.behavior.value_layers) != 3 or int(config.behavior.value_bins) != 255:
        raise ValueError("V4 formal value must use three hidden layers and 255-bin symexp two-hot output")
    if abs(float(config.behavior.value_outscale)) > 1.0e-12:
        raise ValueError("V4 formal value output weight scale must match reference outscale=0.0")
    if abs(float(config.behavior.repval_weight) - 0.3) > 1.0e-12:
        raise ValueError("V4 formal behavior.repval_weight must equal reference value 0.3")
    if str(config.optimizer.type) != "laprop":
        raise ValueError("V4 formal optimizer.type must be laprop")
    if str(config.optimizer.gradient_processing) != "agc_only":
        raise ValueError("V4 formal gradient processing must be agc_only; global norm clipping is unsupported")
    if abs(float(config.optimizer.agc) - 0.3) > 1.0e-12 or abs(float(config.optimizer.pmin) - 1.0e-3) > 1.0e-12:
        raise ValueError("V4 formal AGC must match the reference: agc=0.3, pmin=1e-3")
    if int(config.optimizer.warmup_updates) != 1000:
        raise ValueError("V4 formal optimizer warmup must match the reference: 1000 updates")
    segmentation_policy = str(config.env.dmc.segmentation_failure_policy)
    if segmentation_policy not in {"error", "invalidate_pair"}:
        raise ValueError("env.dmc.segmentation_failure_policy must be error or invalidate_pair")
    if int(config.env.davis.temporal_stride) < 1:
        raise ValueError("env.davis.temporal_stride must be at least 1")
    if str(config.env.davis.end_policy) not in {"loop", "restart"}:
        raise ValueError("env.davis.end_policy must be loop or restart")
    allowed_davis_ids = config.env.davis.get("allowed_video_ids")
    if allowed_davis_ids is not None:
        if not isinstance(allowed_davis_ids, list) or not allowed_davis_ids:
            raise ValueError("env.davis.allowed_video_ids must be null or a non-empty list")
        normalised_davis_ids = [str(value).strip() for value in allowed_davis_ids]
        if any(not value for value in normalised_davis_ids) or len(set(normalised_davis_ids)) != len(
            normalised_davis_ids
        ):
            raise ValueError("env.davis.allowed_video_ids must contain unique non-empty video ids")
    if str(config.project.implementation_revision) != str(config.checkpoint.implementation_revision):
        raise ValueError("project/checkpoint implementation_revision must match")
    if str(config.pipeline.mode) != "synchronous_shard":
        raise ValueError("V5 only supports pipeline.mode=synchronous_shard")
    if bool(config.replay.persist_replay) and not bool(config.pipeline.checkpoint.persist_shards):
        raise ValueError("replay.persist_replay=true requires pipeline.checkpoint.persist_shards=true")
    if int(config.pipeline.shard.max_collected_shards) < 1:
        raise ValueError("pipeline.shard.max_collected_shards must be at least 1")
    if int(config.pipeline.shard.max_ready_shards) < 0 or int(config.pipeline.shard.retain_ready_shards) < 0:
        raise ValueError("pipeline shard ready retention limits must be non-negative")
    _validate_capc_config(config)
    if int(config.training.sequence_length) < 2:
        raise ValueError("training.sequence_length must be at least 2 for lambda-return and replay value loss")
    expected = int(config.env.env_num) * int(config.pipeline.shard.collect_steps_per_env)
    actual = int(config.pipeline.shard.transitions_per_shard)
    if expected != actual:
        raise ValueError(
            "pipeline.shard.transitions_per_shard must equal "
            f"env.env_num * collect_steps_per_env ({actual} != {expected})"
        )


def _validate_capc_config(config: ConfigNode) -> None:
    capc = config.get("capc", {})
    if not bool(capc.get("enabled", False)):
        return
    if int(config.model.num_views) != 4 or int(capc.get("views", 0)) != 4:
        raise ValueError("CAPC requires K=4 synchronized counterfactual views")
    if bool(config.get("counterfactual", {}).get("enabled", False)):
        raise ValueError("CAPC K=4 must not activate the separate formal K=6 counterfactual branch")
    temporal_delta = config.model.get("temporal_delta", {})
    if bool(temporal_delta.get("enabled", False)) or float(temporal_delta.get("weight", 0.0)) != 0.0:
        raise ValueError("CAPC forbids L_delta")
    if "native_rec" in config.model or "native_reconstruction" in config.model:
        raise ValueError("CAPC forbids Native-Rec")
    if str(capc.get("formula")) != "kl_stopgrad_mean_posterior_to_native_prior":
        raise ValueError("CAPC formula contract changed")
    if str(capc.get("probability_semantics")) != "rssm_get_dist_base_dist_probs_with_unimix":
        raise ValueError("CAPC must use native categorical probabilities including RSSM unimix")
    if not bool(capc.get("target_stopgrad", False)):
        raise ValueError("CAPC consensus posterior target must be stop-gradient")
    if float(capc.get("weight", -1.0)) < 0.0:
        raise ValueError("CAPC weight must be non-negative")
    if str(config.optimizer.type) != "laprop":
        raise ValueError("CAPC optimizer must remain LaProp")


def _find_forbidden_keys(value: Any, *, path: str, violations: list[str]) -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else str(key)
        if str(key) in FORBIDDEN_CONFIG_KEYS:
            violations.append(child_path)
        _find_forbidden_keys(child, path=child_path, violations=violations)
