from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..dreamer.behavior import DreamerBehavior, OnlinePolicyState
from ..envs.factory import create_env, create_paired_view_generator
from ..vlm.client import create_vlm_client, infer_b1_b2, infer_k_views
from .schema import ReplaySchema, validate_record
from .shard import CollectedShard, ShardStore
from .temporal_record import TemporalRecordBuilder, summarize_records


@dataclass
class CollectorReport:
    transitions: int
    vector_steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    completed_episodes: list[dict[str, float | int]] = field(default_factory=list)


class OnlineCollector:
    def __init__(
        self,
        config,
        world_model,
        behavior: DreamerBehavior,
        *,
        env_backend: str | None = None,
        paired_backend: str | None = None,
        vlm_backend: str | None = None,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.world_model = world_model
        self.behavior = behavior
        self.device = torch.device(device)
        self.envs = [create_env(config, env_backend, env_id=i) for i in range(int(config.env.env_num))]
        self.paired = [create_paired_view_generator(config, paired_backend, env_id=i) for i in range(int(config.env.env_num))]
        self.vlm_client = create_vlm_client(config, vlm_backend, env_num=int(config.env.env_num))
        counterfactual = config.get("counterfactual", {})
        self.counterfactual_k = int(counterfactual.get("views", 2))
        self.replay_schema = ReplaySchema.from_config(config)
        self.generic_k_view = self.replay_schema.schema_version == "v4_kview_v2"
        self.temporal = TemporalRecordBuilder(
            env_num=int(config.env.env_num),
            action_dim=int(config.model.action_dim),
            counterfactual_views=self.counterfactual_k,
            view_ids=self.replay_schema.view_ids,
            schema_version=self.replay_schema.schema_version,
            temporal_protocol_version=self.replay_schema.temporal_protocol_version,
            pairing_contract=self.replay_schema.pairing_contract,
            background_stream_policy=self.replay_schema.background_stream_policy,
        )
        self.current_steps = [env.reset() for env in self.envs]
        self.policy_state: OnlinePolicyState | None = None
        self.policy_version = 0
        self._episode_return = np.zeros((len(self.envs),), dtype=np.float64)
        self._episode_length = np.zeros((len(self.envs),), dtype=np.int64)
        self._episode_index = np.zeros((len(self.envs),), dtype=np.int64)

    def collect_shard(self, store: ShardStore, *, collect_steps_per_env: int | None = None) -> tuple[CollectedShard, CollectorReport]:
        collect_steps = int(collect_steps_per_env or self.config.pipeline.shard.collect_steps_per_env)
        shard = store.create_collected(policy_version=self.policy_version)
        metrics: dict[str, float] = {}
        completed_episodes: list[dict[str, float | int]] = []
        for _ in range(collect_steps):
            clean_batch = [step.observation for step in self.current_steps]
            mask_results = [self.envs[i].foreground_mask_with_status(clean_batch[i]) for i in range(len(self.envs))]
            masks = [result.mask for result in mask_results]
            pairs = [self.paired[i].render(clean_batch[i], masks[i], info=self.current_steps[i].info) for i in range(len(self.envs))]
            incoming = np.stack([self.temporal.states[i].incoming_action for i in range(len(self.envs))]).astype(np.float32)
            is_first = np.asarray([self.temporal.states[i].is_first for i in range(len(self.envs))], dtype=bool)
            is_terminal_observation = np.asarray([step.is_last for step in self.current_steps], dtype=bool)
            if str(self.config.behavior.policy_backend) == "dreamer":
                with torch.no_grad():
                    action_t, self.policy_state, policy_metrics = self.behavior.act(
                        self.world_model,
                        torch.as_tensor(np.stack([pair.raw_a for pair in pairs]), device=self.device),
                        torch.as_tensor(incoming, device=self.device),
                        torch.as_tensor(is_first, device=self.device),
                        self.policy_state,
                    )
                actions = action_t.detach().cpu().numpy().astype(np.float32)
                metrics.update({key: float(value.detach().float().mean().cpu()) for key, value in policy_metrics.items()})
            else:
                actions = np.stack([env.action_space.sample() for env in self.envs]).astype(np.float32)
            actions[is_terminal_observation] = 0.0
            records = []
            next_steps = []
            for env_id, (env, pair, mask_result, action) in enumerate(
                zip(self.envs, pairs, mask_results, actions, strict=True)
            ):
                record = self.temporal.build_current_record(
                    env_id,
                    raw_a=pair.raw_a,
                    raw_b1=pair.raw_b1 if self.counterfactual_k == 2 and not self.generic_k_view else None,
                    raw_b2=pair.raw_b2 if self.counterfactual_k == 2 and not self.generic_k_view else None,
                    raw_b_views=pair.raw_b_views if self.generic_k_view or self.counterfactual_k in {4, 6} else None,
                    outgoing_action=action,
                    paired_view_valid=bool(pair.paired_view_valid and mask_result.valid),
                    metadata={
                        **pair.metadata,
                        "clean_rgb": pair.clean,
                        "task_foreground_mask": pair.foreground_mask,
                        "background_id_a": pair.background_id_a,
                        "background_id_b1": pair.background_id_b1,
                        "background_id_b2": pair.background_id_b2,
                        "view_id_b1": pair.view_id_b1,
                        "view_id_b2": pair.view_id_b2,
                        "background_ids_b": pair.background_ids_b,
                        "view_ids_b": pair.view_ids_b,
                        "policy_version": int(self.policy_version),
                        "foreground_mask_valid": bool(mask_result.valid),
                        "foreground_mask_area": mask_result.area,
                        "foreground_mask_failure_reason": mask_result.failure_reason,
                    },
                )
                if bool(self.current_steps[env_id].is_last):
                    next_step = env.reset()
                    self.temporal.reset_env(env_id)
                else:
                    next_step = env.step(action)
                    self._episode_return[env_id] += float(next_step.reward)
                    self._episode_length[env_id] += 1
                    if bool(next_step.is_last):
                        completed_episodes.append(
                            {
                                "episode": int(self._episode_index[env_id]),
                                "env_id": int(env_id),
                                "episode_return": float(self._episode_return[env_id]),
                                "episode_length": int(self._episode_length[env_id]),
                            }
                        )
                        self._episode_index[env_id] += 1
                        self._episode_return[env_id] = 0.0
                        self._episode_length[env_id] = 0
                    self.temporal.advance_after_env_step(
                        env_id,
                        outgoing_action=action,
                        reward=next_step.reward,
                        is_last=next_step.is_last,
                        is_terminal=next_step.is_terminal,
                    )
                records.append(record)
                next_steps.append(next_step)
            if self.generic_k_view or self.counterfactual_k in {4, 6}:
                enriched = infer_k_views(
                    self.vlm_client,
                    records,
                    num_views=self.counterfactual_k,
                )
            else:
                enriched = infer_b1_b2(self.vlm_client, records)
            for record in enriched:
                validate_record(record, require_vlm=True, expected_schema=self.replay_schema)
                shard.records.append(record)
            self.current_steps = next_steps
            metrics["segmentation/invalid_pair_count"] = metrics.get("segmentation/invalid_pair_count", 0.0) + float(
                sum(not result.valid for result in mask_results)
            )
        report_metrics = summarize_records(shard.records)
        record_count = max(1, len(shard.records))
        report_metrics.update(
            {
                "counterfactual/pair_valid_fraction": float(
                    sum(bool(row.get("paired_view_valid", False)) for row in shard.records) / record_count
                ),
                "counterfactual/six_view_valid_fraction": float(
                    sum(bool(row.get("six_view_valid", self.counterfactual_k != 6)) for row in shard.records)
                    / record_count
                ),
                "counterfactual/multi_view_valid_fraction": float(
                    sum(bool(row.get("multi_view_valid", self.counterfactual_k == 2)) for row in shard.records)
                    / record_count
                ),
                "counterfactual/vlm_complete_fraction": float(
                    sum(bool(row.get("vlm_views_valid", row.get("vlm_pair_valid", False))) for row in shard.records)
                    / record_count
                ),
                "counterfactual/K": float(self.counterfactual_k),
                "k_view/configured_num_views": float(self.counterfactual_k),
                "k_view/actual_score_views": float(self.counterfactual_k),
                "k_view/actual_feature_views": float(self.counterfactual_k),
                "k_view/actual_history_streams": float(self.counterfactual_k),
            }
        )
        report_metrics.update(metrics)
        stats_snapshot = getattr(self.vlm_client, "stats_snapshot", None)
        if callable(stats_snapshot):
            report_metrics.update(
                {f"vlm/{key}": float(value) for key, value in stats_snapshot().items()}
            )
        return shard, CollectorReport(
            len(shard.records), collect_steps, report_metrics, completed_episodes
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.vlm_client.close()
