"""Extra team-level features from the per-unit snapshots, for the final exploration (axes A and B).

Everything is computed from one step's snapshot only (no future), per side, on the common scale
(health 1 = ten full marines, counts in tens of units). Indexed by (episode_idx, step) like the
state tables, so the columns can be joined onto them.

Spatial
    rg_<side>        radius of gyration of the alive units (map units)
    hull_<side>      convex-hull area of the alive units (0 with fewer than 3)
    nnmean_<side>    mean distance from an alive unit to its nearest enemy
    vto_<side>       mean velocity towards the enemy centroid (map units / s; > 0 closing in)
    cluster_<side>   share of alive units in the largest cluster (single linkage, gap 2.0)
Engagement
    firing_<side>    units alive with their target in range (as scripts/build_firing_counts.py)
    volley_<side>    units whose weapon cooldown went up since the last step (a shot this step)
    overkill_<side>  shooters in range beyond those needed to kill their target
                     (sum over targets of max(0, n - ceil(hp / damage)))
Health distribution
    hmin_<side>      lowest hp share (hp + shield over the maximum) among alive units
    gini_<side>      Gini coefficient of hp among alive units
    dying_<side>     units with hp + shield at most one shot
Groups
    heng_<side>      health of the units within reach of an enemy (rest: health - heng)
    hfront_<side>    health of the front half (the ceil(n / 2) units closest to the enemy centroid)
    afront_<side>    units in that front half
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull, QhullError

from sindy_wm.states.geometry import MARINE_HP, REACH, REF_UNITS

SIDES = ("blue", "red")
CLUSTER_GAP = 2.0
MARINE_DAMAGE = 6.0


def _hull_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    try:
        return float(ConvexHull(p).volume)  # in 2-D, "volume" is the area
    except QhullError:  # collinear
        return 0.0


def _largest_cluster(p: np.ndarray) -> float:
    if len(p) < 2:
        return 1.0 if len(p) else 0.0
    labels = fcluster(linkage(p, "single"), CLUSTER_GAP, criterion="distance")
    return float(np.bincount(labels).max() / len(p))


def _gini(x: np.ndarray) -> float:
    if len(x) < 2 or x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def unit_features(
    snapshots: pd.DataFrame,
    reach: float = REACH,
    unit_health: float = MARINE_HP,
    damage: float = MARINE_DAMAGE,
) -> pd.DataFrame:
    """The features above for every step of one episode's snapshots."""
    snap = snapshots.sort_values(["step", "team", "unit_id"])
    snap = snap.assign(life=snap["hp"] + snap["shield"], life_max=snap["max_hp"] + snap["max_shield"])
    prev_cd = snap.groupby(["team", "unit_id"])["cooldown"].shift(1)
    snap = snap.assign(volley=(snap["cooldown"] > prev_cd + 1e-9) & snap["alive"])
    rows = []
    for step, g in snap.groupby("step", sort=True):
        alive = g[g["alive"]]
        pos = {s: alive.loc[alive["team"] == s, ["x", "y"]].to_numpy() for s in SIDES}
        row = {"step": step}
        for side, opp in (("blue", "red"), ("red", "blue")):
            a = alive[alive["team"] == side]
            p, q = pos[side], pos[opp]
            n = len(p)
            life = a["life"].to_numpy()
            if n == 0:
                row.update(
                    {
                        f"{k}_{side}": 0.0
                        for k in ("rg", "hull", "nnmean", "vto", "cluster", "firing", "volley", "overkill")
                    }
                )
                row.update(
                    {f"{k}_{side}": 0.0 for k in ("hmin", "gini", "dying", "heng", "hfront", "afront")}
                )
                continue
            c = p.mean(0)
            row[f"rg_{side}"] = float(np.sqrt(((p - c) ** 2).sum(1).mean()))
            row[f"hull_{side}"] = _hull_area(p)
            row[f"cluster_{side}"] = _largest_cluster(p)
            if len(q):
                gap = np.linalg.norm(p[:, None] - q[None], axis=-1)
                nearest = gap.min(1)
                row[f"nnmean_{side}"] = float(nearest.mean())
                axis = q.mean(0) - c
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm > 0 else np.array([1.0, 0.0])
                v = a[["vx", "vy"]].to_numpy() if "vx" in a else np.zeros_like(p)
                row[f"vto_{side}"] = float(np.nan_to_num(v @ axis).mean())
                to_centre = np.linalg.norm(p - q.mean(0), axis=1)
                front = np.argsort(to_centre)[: int(np.ceil(n / 2))]
                row[f"heng_{side}"] = float(life[nearest <= reach].sum())
                row[f"hfront_{side}"] = float(life[front].sum())
                row[f"afront_{side}"] = float(len(front))
            else:
                row.update({f"{k}_{side}": 0.0 for k in ("nnmean", "vto", "heng", "hfront", "afront")})
            in_range = (
                a["target_in_range"].astype(bool) if "target_in_range" in a else pd.Series(False, a.index)
            )
            row[f"firing_{side}"] = float(in_range.sum())
            row[f"volley_{side}"] = float(a["volley"].sum())
            # overkill: shooters in range per target, against the shots its remaining life needs
            excess = 0.0
            if "target_id" in a and in_range.any():
                targets = alive[alive["team"] == opp].set_index("unit_id")["life"]
                shooters = a.loc[in_range, "target_id"].value_counts()
                need = np.ceil(targets.reindex(shooters.index).fillna(0) / damage)
                excess = float(np.maximum(shooters - need, 0).sum())
            row[f"overkill_{side}"] = excess
            share = life / a["life_max"].to_numpy()
            row[f"hmin_{side}"] = float(share.min())
            row[f"gini_{side}"] = _gini(life)
            row[f"dying_{side}"] = float((life <= damage + 1e-9).sum())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("step")
    for side in SIDES:
        for k in ("heng", "hfront"):
            out[f"{k}_{side}"] /= REF_UNITS * unit_health
        for k in ("firing", "volley", "overkill", "dying", "afront"):
            out[f"{k}_{side}"] /= REF_UNITS
    return out


def features_for_dataset(dataset_dir, episode_ids=None, offset: int = 0, **kw) -> pd.DataFrame:
    """`unit_features` for every episode of a raw dataset, indexed (episode_idx + offset, step)."""
    from sindy_wm.io import storage

    if episode_ids is None:
        episode_ids = storage.load_episode_index(dataset_dir)["episode_idx"].tolist()
    parts = []
    for chunk in np.array_split(np.asarray(episode_ids), max(1, len(episode_ids) // 100)):
        snap = storage.load_snapshots(dataset_dir, chunk.tolist())
        for ep, g in snap.groupby("episode_idx"):
            f = unit_features(g, **kw)
            f.index = pd.MultiIndex.from_product([[ep + offset], f.index], names=["episode_idx", "step"])
            parts.append(f)
    return pd.concat(parts)
