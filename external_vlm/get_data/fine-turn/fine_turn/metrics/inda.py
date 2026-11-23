from __future__ import annotations


def inda(background_distances, interaction_distances):
    return (interaction_distances > background_distances).float().mean()

