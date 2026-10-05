"""Level A: fully aggregated, team-level state of a battle."""

import numpy as np
import pandas as pd


def level_a(snapshots: pd.DataFrame, include_shields: bool = True, normalize: bool = False) -> pd.DataFrame:
    """Level A state of one episode: one row per step, indexed by step.

    Columns:
        health_blue, health_red: total health per team (dead units count as 0)
        alive_blue, alive_red:   number of alive units per team
        distance:                distance between the centroids of the alive
                                 units; carried forward once a team is wiped out

    Args:
        snapshots: Per-unit snapshot table of a single episode.
        include_shields: Count shields as part of a team's health.
        normalize: Divide health and alive counts by their values at step 0,
            so they run from 1 down to 0.
    """
    if snapshots.duplicated(["step", "team", "unit_id"]).any():
        raise ValueError("Each unit appears more than once per step: pass a single episode.")

    health = snapshots["hp"] + snapshots["shield"] if include_shields else snapshots["hp"]
    df = snapshots.assign(health=health)

    # Sums over all units (dead units add 0), centroids over alive units only
    totals = df.groupby(["step", "team"])[["health", "alive"]].sum().unstack("team")
    centroids = (
        df[df["alive"]].groupby(["step", "team"])[["x", "y"]].mean().unstack("team").reindex(totals.index)
    )

    states = pd.DataFrame(
        {
            "health_blue": totals["health", "blue"],
            "health_red": totals["health", "red"],
            "alive_blue": totals["alive", "blue"].astype(float),
            "alive_red": totals["alive", "red"].astype(float),
            "distance": np.hypot(
                centroids["x", "blue"] - centroids["x", "red"],
                centroids["y", "blue"] - centroids["y", "red"],
            ).ffill(),
        }
    )

    if normalize:
        cols = ["health_blue", "health_red", "alive_blue", "alive_red"]
        states[cols] = states[cols] / states[cols].iloc[0]
    return states
