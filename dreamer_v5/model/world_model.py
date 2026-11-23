from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..counterfactual.relevance import TaskRelevanceEstimator
from .encoder import CanonicalStem
from .encoder_tail import DreamerEncoderTail
from .patch_predictor import BeliefPatchFeatureHead
from .rgb_decoder import TaskRGBDecoder
from .visual_preprocessing import observation_storage_range, preprocess_observation
from ..dreamer.rssm import RSSM


@dataclass
class V4WorldModelOutput:
    raw_a: torch.Tensor
    raw_b1: torch.Tensor | None
    raw_b2: torch.Tensor | None
    raw_b_views: torch.Tensor | None
    rgb_reconstruction: torch.Tensor
    vgc_patch_prediction: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor
    rssm_post_logit: torch.Tensor
    rssm_prior_logit: torch.Tensor
    online_belief: torch.Tensor
    online_spatial_embedding: torch.Tensor
    reward_pred: torch.Tensor
    continue_logit: torch.Tensor
    metrics: dict[str, torch.Tensor]


class V4WorldModel(nn.Module):
    """Clean V4 A-branch world model with K=0 VLM feature and MSIR heads."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        self.action_dim = int(model_cfg.action_dim)
        self.stem = CanonicalStem(3, int(model_cfg.stem_channels))
        self.encoder = DreamerEncoderTail(int(model_cfg.stem_channels), int(model_cfg.encoder_embed_dim))
        self.rssm = RSSM(config.r2_rssm, embed_size=int(model_cfg.encoder_embed_dim), act_dim=self.action_dim)
        belief_dim = int(self.rssm.feat_size)
        vgc = model_cfg.vlm_gaussian_consensus
        self.patch_head = BeliefPatchFeatureHead(
            belief_dim,
            patch_count=int(vgc.patch_count),
            feature_dim=int(vgc.patch_feature_dim),
            hidden_dim=int(vgc.hidden_dim),
        )
        self.rgb_decoder = TaskRGBDecoder(belief_dim, image_size=int(model_cfg.image_size))
        self.reward_head = _ScalarHead(belief_dim)
        self.continue_head = _ScalarHead(belief_dim)
        self.task_relevance = TaskRelevanceEstimator(int(vgc.patch_count))
        self.belief_dim = belief_dim

    def forward_training_batch(self, batch: dict[str, torch.Tensor]) -> V4WorldModelOutput:
        raw_a_storage = _require(batch, "raw_a")
        raw_storage_min, raw_storage_max = observation_storage_range(raw_a_storage)
        raw_a = preprocess_observation(raw_a_storage, name="raw_a")
        action = _bta(_require(batch, "action"), action_dim=self.action_dim, device=raw_a.device)
        is_first = _bt_bool(batch.get("is_first"), shape=action.shape[:2], device=raw_a.device)
        stem = self.stem(raw_a)
        embed = self.encoder(stem)
        initial = self.rssm.initial(raw_a.shape[0])
        stoch, deter, post_logit = self.rssm.observe(embed, action, initial, is_first)
        _, prior_logit = self.rssm.prior(deter)
        belief = self.rssm.get_feat(stoch, deter)
        msir_cfg = self.config.model.msir
        decoder_training = bool(msir_cfg.get("training_enabled", msir_cfg.enabled))
        return V4WorldModelOutput(
            raw_a=raw_a,
            raw_b1=_optional_btchw(batch.get("raw_b1"), raw_a.device),
            raw_b2=_optional_btchw(batch.get("raw_b2"), raw_a.device),
            raw_b_views=_optional_btkchw(batch.get("raw_b_views"), raw_a.device),
            rgb_reconstruction=self.rgb_decoder(belief if decoder_training else belief.detach()),
            vgc_patch_prediction=self.patch_head(belief),
            stoch=stoch,
            deter=deter,
            rssm_post_logit=post_logit,
            rssm_prior_logit=prior_logit,
            online_belief=belief,
            online_spatial_embedding=stem,
            reward_pred=self.reward_head(belief),
            continue_logit=self.continue_head(belief),
            metrics={
                "temporal/first_action_zero_fraction": _first_action_zero_fraction(action, is_first).detach(),
                "model/belief_dim": torch.tensor(float(belief.shape[-1]), device=belief.device),
                "visual/raw_a_storage_min": raw_storage_min.to(belief.device),
                "visual/raw_a_storage_max": raw_storage_max.to(belief.device),
                "visual/preprocessed_a_min": raw_a.detach().amin(),
                "visual/preprocessed_a_max": raw_a.detach().amax(),
                "visual/encoder_input_min": (raw_a.detach() - 0.5).amin(),
                "visual/encoder_input_max": (raw_a.detach() - 0.5).amax(),
            },
        )

    forward = forward_training_batch

    def encode_observation(self, raw_a: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.stem(preprocess_observation(raw_a, name="raw_a")))

    def posterior_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        raw_a: torch.Tensor,
        incoming_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode_observation(raw_a)
        if embed.ndim == 3:
            embed = embed[:, 0]
        action = torch.as_tensor(incoming_action, device=embed.device).float()
        reset = torch.as_tensor(is_first, device=embed.device).bool().reshape(action.shape[0])
        stoch, deter, logit = self.rssm.obs_step(stoch, deter, action, embed, reset)
        feat = self.rssm.get_feat(stoch, deter)
        return stoch, deter, logit, feat

    def initial_state(self, batch_size: int):
        return self.rssm.initial(int(batch_size))

    def reward_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reward_head(feat).unsqueeze(-1)

    def continue_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(feat)).unsqueeze(-1)

    def world_parameters(self):
        decoder_training = bool(
            self.config.model.msir.get("training_enabled", self.config.model.msir.enabled)
        )
        excluded = set() if decoder_training else {id(param) for param in self.rgb_decoder.parameters()}
        return [param for param in self.parameters() if param.requires_grad and id(param) not in excluded]

    def visual_encoder_parameters(self):
        return [*self.stem.parameters(), *self.encoder.parameters()]

    def rssm_observation_parameters(self):
        return list(self.rssm._obs_net.parameters())

    def rssm_recurrent_prior_parameters(self):
        return [*self.rssm._deter_net.parameters(), *self.rssm._img_net.parameters()]

    def patch_predictor_parameters(self):
        return list(self.patch_head.parameters())

    def decoder_parameters(self):
        return list(self.rgb_decoder.parameters())


class _ScalarHead(nn.Module):
    def __init__(self, belief_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(belief_dim)),
            nn.Linear(int(belief_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.net(belief).squeeze(-1)


def _require(batch: dict, key: str) -> torch.Tensor:
    if key not in batch or batch[key] is None:
        raise KeyError(f"V4WorldModel requires batch['{key}']")
    return batch[key]


def _optional_btchw(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 5 and tensor.shape[2] == 3:
        return tensor.float()
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3).contiguous().float()
    if tensor.ndim == 4 and tensor.shape[1] == 3:
        return tensor[:, None].float()
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        return tensor.permute(0, 3, 1, 2).contiguous()[:, None].float()
    raise ValueError(f"expected optional RGB tensor, got {tuple(tensor.shape)}")


def _optional_btkchw(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 6 and tensor.shape[3] == 3:
        return tensor.float()
    if tensor.ndim == 6 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 2, 5, 3, 4).contiguous().float()
    if tensor.ndim == 5 and tensor.shape[2] == 3:
        return tensor[:, None].float()
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3).contiguous()[:, None].float()
    raise ValueError(f"expected K-view RGB tensor, got {tuple(tensor.shape)}")


def _bta(value: torch.Tensor, *, action_dim: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device).float()
    if tensor.ndim == 2 and tensor.shape[-1] == action_dim:
        return tensor[:, None]
    if tensor.ndim == 3 and tensor.shape[-1] == action_dim:
        return tensor
    raise ValueError(f"expected action [B,T,{action_dim}], got {tuple(tensor.shape)}")


def _bt_bool(value: torch.Tensor | None, *, shape: torch.Size | tuple[int, int], device: torch.device) -> torch.Tensor:
    if value is None:
        out = torch.zeros(tuple(shape), dtype=torch.bool, device=device)
        out[:, 0] = True
        return out
    tensor = torch.as_tensor(value, device=device).bool()
    if tensor.ndim == 1:
        tensor = tensor[:, None].expand(tuple(shape))
    if tuple(tensor.shape) != tuple(shape):
        tensor = tensor.reshape(tuple(shape))
    return tensor


def _first_action_zero_fraction(action: torch.Tensor, is_first: torch.Tensor) -> torch.Tensor:
    if not bool(is_first.any()):
        return torch.tensor(1.0, device=action.device)
    zero = action.detach().abs().sum(dim=-1) <= 1.0e-6
    return zero[is_first].float().mean()
