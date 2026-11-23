"""Strictly restore the trained Walker modules and verify two inference steps."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unswayed.config import to_config_node
from unswayed.dreamer.behavior import DreamerBehavior
from unswayed.model.world_model import WorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = to_config_node(copy.deepcopy(payload["config"]))
    cfg.training.device = str(device)
    cfg.r2_rssm.device = str(device)
    world = WorldModel(cfg).to(device).eval().requires_grad_(False)
    behavior = DreamerBehavior(cfg, belief_dim=world.belief_dim, action_dim=world.action_dim, device=device)
    behavior.to(device).eval().requires_grad_(False)
    world.load_state_dict(payload["model"], strict=True)
    behavior.load_state_dict(payload["behavior"], strict=True)
    with torch.no_grad():
        rgb = torch.zeros((1, 3, 64, 64), dtype=torch.uint8, device=device)
        action = torch.zeros((1, 6), device=device)
        state = behavior.initial_policy_state(world, 1)
        for step in range(2):
            first = torch.tensor([step == 0], device=device)
            action, state, _ = behavior.act(world, rgb, action, first, state, deterministic=True)
            assert action.shape == (1, 6) and torch.isfinite(action).all()
            feature = world.rssm.get_feat(state.stoch, state.deter)
            prediction = world.patch_head(feature[:, None])
            reconstruction = world.rgb_decoder(feature[:, None])
            assert prediction.shape == (1, 1, 64, 64) and torch.isfinite(prediction).all()
            assert torch.isfinite(reconstruction).all()
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({
        "status": "PASS", "strict_world_model_load": True, "strict_actor_critic_load": True,
        "world_state_tensors": len(payload["model"]), "behavior_state_tensors": len(payload["behavior"]),
        "gradient_updates": payload.get("gradient_updates"), "global_env_steps": payload.get("global_env_steps"),
        "checkpoint_sha256": digest, "device": str(device), "inference_steps": 2,
        "torch_version": torch.__version__, "optimizer_constructed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
