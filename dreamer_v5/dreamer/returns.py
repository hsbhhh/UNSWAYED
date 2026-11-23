from __future__ import annotations

import torch


def lambda_return(
    last: torch.Tensor,
    term: torch.Tensor,
    reward: torch.Tensor,
    value: torch.Tensor,
    boot: torch.Tensor,
    disc: float,
    lamb: float,
) -> torch.Tensor:
    """Reference Dreamer lambda-return.

    Shapes are `[B, T, 1]`. The returned sequence has length `T - 1`.
    """

    if not (last.shape == term.shape == reward.shape == value.shape == boot.shape):
        raise ValueError(
            "lambda_return expects matching shapes: "
            f"last={tuple(last.shape)} term={tuple(term.shape)} reward={tuple(reward.shape)} "
            f"value={tuple(value.shape)} boot={tuple(boot.shape)}"
        )
    live = (1.0 - term.float())[:, 1:] * float(disc)
    cont = (1.0 - last.float())[:, 1:] * float(lamb)
    interm = reward.float()[:, 1:] + (1.0 - cont) * live * boot.float()[:, 1:]
    out = [boot.float()[:, -1]]
    for index in reversed(range(live.shape[1])):
        out.append(interm[:, index] + live[:, index] * cont[:, index] * out[-1])
    return torch.stack(list(reversed(out))[:-1], 1)
