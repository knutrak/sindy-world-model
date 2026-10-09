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


def engagement(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Team-level engagement of one episode: one row per step, indexed by step.

    Needs snapshots recorded with target columns (target_team, target_id,
    target_in_range);

    Columns, per team (suffix _blue / _red):
        engaged:  number of alive units with a target in attack range, i.e. the
                  firepower actually being applied (compare alive_*)
        targets:  number of distinct enemy units being shot at by engaged units.
                  1 = all fire focused on one unit; equal to engaged = fully spread
        speed:    mean speed of alive units (map units per second)
    """
    required = {"target_team", "target_id", "target_in_range", "vx", "vy"}
    if missing := required - set(snapshots.columns):
        raise ValueError(f"Snapshots lack {sorted(missing)}: recorded before target/velocity logging?")

    alive = snapshots[snapshots["alive"]]
    engaged = alive[alive["target_in_range"]]
    by = ["step", "team"]
    steps = pd.Index(sorted(snapshots["step"].unique()), name="step")

    def per_team(series: pd.Series) -> pd.DataFrame:
        return series.unstack("team").reindex(index=steps, columns=["blue", "red"]).fillna(0)

    n_engaged = per_team(engaged.groupby(by).size())
    n_targets = per_team(engaged.groupby(by)["target_id"].nunique())
    speed = per_team(alive.assign(speed=np.hypot(alive["vx"], alive["vy"])).groupby(by)["speed"].mean())

    return pd.DataFrame(
        {
            "engaged_blue": n_engaged["blue"],
            "engaged_red": n_engaged["red"],
            "targets_blue": n_targets["blue"],
            "targets_red": n_targets["red"],
            "speed_blue": speed["blue"],
            "speed_red": speed["red"],
        }
    )
