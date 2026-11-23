from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from .networks import ActionHead, ReturnEMA, ValueHead
from .returns import lambda_return
from .tools import tensorstats


ACTOR_VARIANT = "E1_PRECLAMP_SCORE"


@dataclass
class BehaviorOutput:
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    repval_loss: torch.Tensor
    total_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class OnlinePolicyState:
    stoch: torch.Tensor
    deter: torch.Tensor


class DreamerBehavior(nn.Module):
    """Production E1 Dreamer actor/value behavior learning."""

    def __init__(self, config, *, belief_dim: int, action_dim: int, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        bcfg = config.behavior
        self.imag_horizon = int(bcfg.imag_horizon)
        self.horizon = int(bcfg.horizon)
        self.lamb = float(bcfg["lambda"])
        self.actor_entropy = float(bcfg.actor_entropy)
        self.repval_weight = float(bcfg.repval_weight)
        self.slow_target_update = int(bcfg.slow_target_update)
        self.slow_target_fraction = float(bcfg.slow_target_fraction)
        self.actor = ActionHead(
            int(belief_dim),
            int(action_dim),
            hidden_dim=int(bcfg.actor_hidden),
            hidden_layers=int(bcfg.actor_layers),
            outscale=float(bcfg.actor_outscale),
            min_std=float(bcfg.actor_min_std),
            max_std=float(bcfg.actor_max_std),
        )
        self.value = ValueHead(
            int(belief_dim),
            hidden_dim=int(bcfg.value_hidden),
            hidden_layers=int(bcfg.value_layers),
            bin_num=int(bcfg.value_bins),
            outscale=float(bcfg.value_outscale),
        )
        self.slow_value = copy.deepcopy(self.value)
        for param in self.slow_value.parameters():
            param.requires_grad_(False)
        self.return_ema = ReturnEMA(self.device, alpha=float(bcfg.return_ema_alpha))
        self.slow_value_updates = 0

    def trainable_parameters(self):
        return [param for param in list(self.actor.parameters()) + list(self.value.parameters()) if param.requires_grad]

    @torch.no_grad()
    def initial_policy_state(self, world_model, batch_size: int) -> OnlinePolicyState:
        stoch, deter = world_model.initial_state(batch_size)
        return OnlinePolicyState(stoch=stoch.detach(), deter=deter.detach())

    @torch.no_grad()
    def act(
        self,
        world_model,
        raw_a: torch.Tensor,
        incoming_action: torch.Tensor,
        is_first: torch.Tensor,
        state: OnlinePolicyState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, OnlinePolicyState, dict[str, torch.Tensor]]:
        batch = int(torch.as_tensor(incoming_action).shape[0])
        if state is None:
            state = self.initial_policy_state(world_model, batch)
        stoch, deter, _, feat = world_model.posterior_step(
            state.stoch.to(self.device),
            state.deter.to(self.device),
            torch.as_tensor(raw_a, device=self.device),
            torch.as_tensor(incoming_action, device=self.device),
            torch.as_tensor(is_first, device=self.device),
        )
        dist = self.actor(feat)
        action = dist.mean if deterministic else dist.rsample()
        action = action.clamp(-1.0, 1.0)
        metrics = {"behavior/action_entropy": dist.entropy().detach().mean()}
        return action.detach(), OnlinePolicyState(stoch=stoch.detach(), deter=deter.detach()), metrics

    def imagine(self, start_state: tuple[torch.Tensor, torch.Tensor], world_model, horizon: int | None = None):
        horizon = self.imag_horizon + 1 if horizon is None else int(horizon)
        stoch, deter = start_state
        feats, pre_actions, executed_actions = [], [], []
        for _ in range(horizon):
            with torch.no_grad():
                feat = world_model.rssm.get_feat(stoch, deter).detach()
            dist = self.actor(feat.detach())
            pre_action = dist.rsample()
            executed_action = pre_action.clamp(-1.0, 1.0)
            feats.append(feat)
            pre_actions.append(pre_action)
            executed_actions.append(executed_action)
            with torch.no_grad():
                stoch, deter = world_model.rssm.img_step(
                    stoch.detach(), deter.detach(), executed_action.detach()
                )
        return (
            torch.stack(feats, dim=1),
            torch.stack(pre_actions, dim=1),
            torch.stack(executed_actions, dim=1),
        )

    def compute_losses(
        self,
        world_model,
        post_stoch: torch.Tensor,
        post_deter: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> BehaviorOutput:
        bsz, time = post_deter.shape[:2]
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, post_deter.shape[-1]).detach(),
        )
        imag_feat, imag_pre_action, imag_executed_action = self.imagine(
            start, world_model, self.imag_horizon + 1
        )
        with torch.no_grad():
            imag_reward = world_model.reward_from_feat(imag_feat).detach()
            imag_continue = world_model.continue_from_feat(imag_feat).detach()
            imag_value = self.value(imag_feat).mode().detach()
            imag_slow_value = self.slow_value(imag_feat).mode().detach()
            disc = 1.0 - 1.0 / float(self.horizon)
            weight = torch.cumprod(imag_continue * disc, dim=1)
            last = torch.zeros_like(imag_continue)
            term = 1.0 - imag_continue
            ret = lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
            ret_offset, ret_scale = self.return_ema(ret)
            advantage = (ret - imag_value[:, :-1]) / ret_scale
        policy = self.actor(imag_feat.detach())
        # E1 is the exact behavior used by the final production checkpoint:
        # score the retained pre-clamp sample, while the world model receives
        # only the clamped executed action from imagine().
        logpi = policy.log_prob(imag_pre_action.detach())[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        actor_loss = torch.mean(weight[:, :-1].detach() * -(logpi * advantage.detach() + self.actor_entropy * entropy))
        actor_pg_term = torch.mean(weight[:, :-1].detach() * -(logpi * advantage.detach()))
        actor_entropy_term = torch.mean(weight[:, :-1].detach() * -(self.actor_entropy * entropy))
        value_dist = self.value(imag_feat.detach())
        target_padded = torch.cat([ret, 0.0 * ret[:, -1:]], dim=1)
        value_loss = torch.mean(
            weight[:, :-1].detach()
            * (-value_dist.log_prob(target_padded.detach()) - value_dist.log_prob(imag_slow_value.detach()))[:, :-1].unsqueeze(-1)
        )

        with torch.no_grad():
            feat_replay = world_model.rssm.get_feat(post_stoch.detach(), post_deter.detach())
            last_replay = torch.as_tensor(batch["is_last"], device=post_deter.device).float().reshape(bsz, time, 1)
            term_replay = torch.as_tensor(batch["is_terminal"], device=post_deter.device).float().reshape(bsz, time, 1)
            reward_replay = torch.as_tensor(batch["reward"], device=post_deter.device).float().reshape(bsz, time, 1)
            boot = ret[:, 0].reshape(bsz, time, 1)
            replay_value = self.value(feat_replay).mode().detach()
            replay_slow_value = self.slow_value(feat_replay).mode().detach()
            replay_ret = lambda_return(last_replay, term_replay, reward_replay, replay_value, boot, disc, self.lamb)
            replay_target = torch.cat([replay_ret, 0.0 * replay_ret[:, -1:]], dim=1)
            replay_weight = 1.0 - last_replay
        replay_value_dist = self.value(feat_replay.detach())
        repval_loss = torch.mean(
            replay_weight[:, :-1].detach()
            * (-replay_value_dist.log_prob(replay_target.detach()) - replay_value_dist.log_prob(replay_slow_value.detach()))[:, :-1].unsqueeze(-1)
        )
        weighted_repval = self.repval_weight * repval_loss
        total = actor_loss + value_loss + weighted_repval
        actor_params = self.actor.raw_parameters(imag_feat.detach()).detach()
        raw_mean, raw_std = torch.chunk(actor_params, 2, dim=-1)
        tanh_mean = torch.tanh(raw_mean.float())
        transformed_std = (
            (float(self.actor.max_std) - float(self.actor.min_std))
            * torch.sigmoid(raw_std.float() + 2.0)
            + float(self.actor.min_std)
        )
        deterministic_action = tanh_mean
        theoretical_entropy_max = raw_mean.new_tensor(
            0.5 * raw_mean.shape[-1] * (1.0 + math.log(2.0 * math.pi))
        )
        value_logits = value_dist.logits.detach()
        decoded_value = value_dist.mode().detach()
        metrics = {
            "behavior/actor_loss": actor_loss.detach(),
            "behavior/actor_pg_term": actor_pg_term.detach(),
            "behavior/actor_entropy_bonus_term": actor_entropy_term.detach(),
            "behavior/value_loss": value_loss.detach(),
            "behavior/repval_loss": repval_loss.detach(),
            "behavior/repval_weight": torch.tensor(self.repval_weight, device=repval_loss.device),
            "behavior/repval_weighted_loss": weighted_repval.detach(),
            "behavior/total_loss": total.detach(),
            "behavior/policy_entropy": entropy.detach().mean(),
            "behavior/imag_reward": imag_reward.detach().mean(),
            "behavior/imag_continue": imag_continue.detach().mean(),
            "behavior/imag_value": imag_value.detach().mean(),
            "behavior/imag_slow_value": imag_slow_value.detach().mean(),
            "behavior/imag_return": ret.detach().mean(),
            "behavior/imag_return_std": ret.detach().std(),
            "behavior/imag_return_median": torch.quantile(ret.detach().float(), 0.50),
            "behavior/imag_return_p10": torch.quantile(ret.detach().float(), 0.10),
            "behavior/imag_return_p90": torch.quantile(ret.detach().float(), 0.90),
            "behavior/imag_advantage": advantage.detach().mean(),
            "behavior/imag_advantage_std": advantage.detach().std(),
            "behavior/imag_advantage_p05": torch.quantile(advantage.detach().float(), 0.05),
            "behavior/imag_advantage_p25": torch.quantile(advantage.detach().float(), 0.25),
            "behavior/imag_advantage_p50": torch.quantile(advantage.detach().float(), 0.50),
            "behavior/imag_advantage_p75": torch.quantile(advantage.detach().float(), 0.75),
            "behavior/imag_advantage_p95": torch.quantile(advantage.detach().float(), 0.95),
            "behavior/trajectory_weight": weight.detach().mean(),
            "behavior/trajectory_weight_std": weight.detach().std(),
            "behavior/trajectory_weight_min": weight.detach().amin(),
            "behavior/trajectory_weight_max": weight.detach().amax(),
            "behavior/ret_005": self.return_ema.ema_vals[0].detach(),
            "behavior/ret_095": self.return_ema.ema_vals[1].detach(),
            "behavior/return_ema_range": (
                self.return_ema.ema_vals[1] - self.return_ema.ema_vals[0]
            ).detach(),
            "behavior/actor_param_logits_mean": actor_params.mean(),
            "behavior/actor_param_logits_std": actor_params.std(),
            "behavior/value_logits_min": value_logits.amin(),
            "behavior/value_logits_max": value_logits.amax(),
            "behavior/decoded_value_mean": decoded_value.mean(),
            "behavior/decoded_value_std": decoded_value.std(),
            "behavior/pre_action_mean": imag_pre_action.detach().mean(),
            "behavior/pre_action_std": imag_pre_action.detach().std(),
            "behavior/pre_action_rms": imag_pre_action.detach().float().square().mean().sqrt(),
            "behavior/executed_action_mean": imag_executed_action.detach().mean(),
            "behavior/executed_action_std": imag_executed_action.detach().std(),
            "behavior/executed_action_rms": imag_executed_action.detach().float().square().mean().sqrt(),
            "behavior/pre_action_exceed_fraction": (imag_pre_action.detach().abs() > 1.0).float().mean(),
            "behavior/action_positive_boundary_fraction": (imag_executed_action.detach() >= 1.0).float().mean(),
            "behavior/action_negative_boundary_fraction": (imag_executed_action.detach() <= -1.0).float().mean(),
            "behavior/action_saturation_gt_095": (imag_executed_action.detach().abs() > 0.95).float().mean(),
            "behavior/action_saturation_gt_099": (imag_executed_action.detach().abs() > 0.99).float().mean(),
            "behavior/deterministic_action_rms": deterministic_action.square().mean().sqrt(),
            "behavior/deterministic_saturation_gt_099": (
                deterministic_action.abs() > 0.99
            ).float().mean(),
            "behavior/transformed_std_mean": transformed_std.mean(),
            "behavior/transformed_std_max": transformed_std.amax(),
            "behavior/transformed_std_fraction_gt_05": (transformed_std > 0.50).float().mean(),
            "behavior/transformed_std_fraction_gt_08": (transformed_std > 0.80).float().mean(),
            "behavior/transformed_std_fraction_gt_095": (transformed_std > 0.95).float().mean(),
            "behavior/theoretical_entropy_max": theoretical_entropy_max,
            "behavior/entropy_gap_to_theoretical_max": (
                theoretical_entropy_max - entropy.detach().mean()
            ),
            "behavior/logpi_mean": logpi.detach().float().mean(),
            "behavior/logpi_std": logpi.detach().float().std(),
            "likelihood/advantage_positive_fraction": (advantage.detach() > 0).float().mean(),
            "likelihood/advantage_negative_fraction": (advantage.detach() < 0).float().mean(),
        }
        metrics.update(
            {f"behavior/{key}": value for key, value in tensorstats(imag_executed_action, "action").items()}
        )
        return BehaviorOutput(actor_loss, value_loss, repval_loss, total, metrics)

    @torch.no_grad()
    def update_slow_value(self) -> None:
        if self.slow_value_updates % max(1, self.slow_target_update) == 0:
            mix = float(self.slow_target_fraction)
            for src, tgt in zip(self.value.parameters(), self.slow_value.parameters()):
                tgt.data.copy_(mix * src.data + (1.0 - mix) * tgt.data)
        self.slow_value_updates += 1
