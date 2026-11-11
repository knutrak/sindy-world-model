"""Team-level geometry from unit positions: who is within reach of whom, and how deep the formations are.

Moved here from `notebooks/engagement_from_distance.ipynb` (sections 9 and 11), so
notebooks, scripts and a live planner share one definition.

On the common scale used throughout: health 1 = ten full-health marines, alive and
reach in tens of units, distances in map units.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_wm.io import storage
from sindy_wm.states.aggregate import level_a

REACH = 5 + 2 * 0.375  # marine attack range plus two unit radii: SMAClite's in-range test, centre to centre
REF_UNITS, MARINE_HP = 10, 45.0  # common scale: 1 = ten full-health marines
GEOMETRY = ["front_distance", "reach_blue", "reach_red", "depth_blue", "depth_red", "width_blue", "width_red"]


def geometry_from_positions(blue: np.ndarray, red: np.ndarray, reach: float = REACH) -> dict:
    """Geometry of one moment from the alive units' positions (arrays of shape (n, 2)).

    Returns distance (between centroids), front_distance (closest pair), reach_blue /
    reach_red (number of units with an enemy within `reach`) and depth_blue / depth_red
    (spread of each formation along the line between the centroids) and width_blue /
    width_red (spread across that line). With a side wiped out: distances NaN, the rest 0.
    """
    if len(blue) == 0 or len(red) == 0:
        return {
            "distance": np.nan,
            "front_distance": np.nan,
            "reach_blue": 0.0,
            "reach_red": 0.0,
            "depth_blue": 0.0,
            "depth_red": 0.0,
            "width_blue": 0.0,
            "width_red": 0.0,
        }
    gap = np.linalg.norm(blue[:, None, :] - red[None, :, :], axis=-1)
    centre_blue, centre_red = blue.mean(axis=0), red.mean(axis=0)
    axis = centre_red - centre_blue
    distance = float(np.linalg.norm(axis))
    axis = axis / distance if distance > 0 else np.array([1.0, 0.0])
    perp = np.array([-axis[1], axis[0]])
    return {
        "distance": distance,
        "front_distance": float(gap.min()),
        "reach_blue": float((gap <= reach).any(axis=1).sum()),
        "reach_red": float((gap <= reach).any(axis=0).sum()),
        "depth_blue": float(np.std((blue - centre_blue) @ axis)),
        "depth_red": float(np.std((red - centre_red) @ axis)),
        "width_blue": float(np.std((blue - centre_blue) @ perp)),
        "width_red": float(np.std((red - centre_red) @ perp)),
    }


def geometry(snapshots: pd.DataFrame, reach: float = REACH) -> pd.DataFrame:
    """Per step of one episode: front distance, units within reach, and formation depth.

    Vectorised over steps; same quantities as `geometry_from_positions` (reach as a
    count of units, not scaled). Depth and front distance are carried forward once a
    side is wiped out.
    """
    alive = snapshots[snapshots["alive"]]
    steps = pd.Index(sorted(snapshots["step"].unique()), name="step")
    pos = {}
    for team in ["blue", "red"]:
        units = alive[alive["team"] == team]
        pos[team] = np.stack(
            [units.pivot(index="step", columns="unit_id", values=c).reindex(steps).to_numpy() for c in "xy"],
            axis=-1,
        )  # step, unit, (x, y); NaN once a unit is dead
    gap = np.linalg.norm(pos["blue"][:, :, None, :] - pos["red"][:, None, :, :], axis=-1)
    within = gap <= reach  # NaN compares False: dead units never count
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN steps once a side is gone
        front = np.nanmin(gap, axis=(1, 2))
        centre = {team: np.nanmean(p, axis=1) for team, p in pos.items()}
        axis = centre["red"] - centre["blue"]
        axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
        perp = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
        depth, width = (
            {
                team: np.nanstd(np.einsum("sud,sd->su", p - centre[team][:, None, :], along), axis=1)
                for team, p in pos.items()
            }
            for along in (axis, perp)
        )
    return pd.DataFrame(
        {
            "front_distance": pd.Series(front, index=steps).ffill(),
            "reach_blue": within.any(axis=2).sum(axis=1).astype(float),
            "reach_red": within.any(axis=1).sum(axis=1).astype(float),
            "depth_blue": pd.Series(depth["blue"], index=steps).ffill(),
            "depth_red": pd.Series(depth["red"], index=steps).ffill(),
            "width_blue": pd.Series(width["blue"], index=steps).ffill(),
            "width_red": pd.Series(width["red"], index=steps).ffill(),
        },
        index=steps,
    )


def load_team_states(
    dataset_dir: str | Path, episode_ids: list[int] | None = None, n_chunks: int = 10
) -> pd.DataFrame:
    """Team-level states of a dataset on the common scale, indexed by (episode_idx, step).

    Columns: health_*, alive_*, distance (`level_a`), the `GEOMETRY` columns, and t in
    seconds. Built chunk by chunk to keep memory down.
    """
    dt = storage.load_manifest(dataset_dir)["config"]["env"]["seconds_per_step"]
    if episode_ids is None:
        episode_ids = storage.load_episode_index(dataset_dir)["episode_idx"].tolist()
    parts = []
    for chunk in np.array_split(np.asarray(episode_ids), max(1, min(n_chunks, len(episode_ids)))):
        by_episode = storage.load_snapshots(dataset_dir, chunk.tolist()).groupby("episode_idx")
        parts.append(
            by_episode.apply(level_a, include_groups=False).join(
                by_episode.apply(geometry, include_groups=False)
            )
        )
    states = pd.concat(parts)
    states[["health_blue", "health_red"]] /= REF_UNITS * MARINE_HP
    states[["alive_blue", "alive_red", "reach_blue", "reach_red"]] /= REF_UNITS
    states["t"] = states.index.get_level_values("step") * dt
    return states


def state_from_game(game) -> dict:
    """The same team-level state, read from a live SMAClite game (`env.unwrapped`), on the common scale."""
    blue, red = list(game.agents.values()), list(game.enemies.values())
    geom = geometry_from_positions(
        np.array([u.pos for u in blue]).reshape(-1, 2), np.array([u.pos for u in red]).reshape(-1, 2)
    )
    scale = REF_UNITS * MARINE_HP
    return {
        "health_blue": sum(u.hp + u.shield for u in blue) / scale,
        "health_red": sum(u.hp + u.shield for u in red) / scale,
        "alive_blue": len(blue) / REF_UNITS,
        "alive_red": len(red) / REF_UNITS,
        "distance": geom["distance"],
        "front_distance": geom["front_distance"],
        "reach_blue": geom["reach_blue"] / REF_UNITS,
        "reach_red": geom["reach_red"] / REF_UNITS,
        "depth_blue": geom["depth_blue"],
        "depth_red": geom["depth_red"],
        "width_blue": geom["width_blue"],
        "width_red": geom["width_red"],
    }
