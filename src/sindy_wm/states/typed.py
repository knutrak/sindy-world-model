"""Per-type team states for battles with more than one unit type (branch mixed-marauders).

For each side and unit type: health, units alive, units in reach, and the measured fire allocation
(how many units of attacker type a have a target of type t in attack range). Plus the side-level
geometry of `states.geometry` (centroid distance, front distance, depth, width).

Scales, per type (so the coefficients read as unit rates):
    health_<side>_<type>   hp + shield summed over the type's units, in units of ten full-health
                           units of that type (1 = ten full units)
    alive_<side>_<type>    alive units / 10
    reach_<side>_<type>    units of the type with an enemy inside its own attack reach, / 10.
                           Reach is per pair: attack range + own radius + target radius (SMAClite's
                           in-range test), so a marine reaches a marine at 5.75 and a marauder at 5.94.
    fire_<side>_<a>_<t>    alive units of type a whose current target is a unit of type t and in
                           range, / 10 (from the snapshot's target columns)

With one type present this reduces to `geometry.load_team_states` (tested).
"""

import warnings
from functools import cache

import numpy as np
import pandas as pd
from smaclite.env.units.unit_type import StandardUnit

from sindy_wm.io import storage
from sindy_wm.states.geometry import REF_UNITS

SIDES = ("blue", "red")


@cache
def unit_stats(name: str) -> dict:
    s = StandardUnit[name].stats
    return {
        "hp": float(s.hp),
        "shield": float(getattr(s, "shield", 0) or 0),
        "range": float(s.attack_range),
        "radius": float(s.size) / 2,
        "damage": float(s.damage),
        "cooldown": float(s.cooldown),
        "armor": float(s.armor),
        "speed": float(s.speed),
    }


def pair_reach(attacker: str, target: str) -> float:
    """Centre-to-centre distance at which `attacker` can shoot `target` (SMAClite's in-range test)."""
    a, t = unit_stats(attacker), unit_stats(target)
    return a["range"] + a["radius"] + t["radius"]


def typed_states(
    snapshots: pd.DataFrame, types: tuple[str, ...], max_reach: float | None = None
) -> pd.DataFrame:
    """Per-type state of one episode, one row per step (see the module docstring).

    max_reach caps every pair reach (centre to centre), e.g. 6.0: SMAClite only offers agents attack
    actions within AGENT_TARGET_RANGE = 6, whatever the unit's attack range (docs/mixed/PROGRESS.md).
    The fire columns are unaffected (they come from the recorded targets).
    """
    unknown = sorted(set(snapshots["unit_type"].unique()) - set(types))
    if unknown:  # their positions would count in the geometry and reach, but not their health or fire
        raise ValueError(f"the battle has unit types {unknown} that are not in types={types}")
    steps = pd.Index(sorted(snapshots["step"].unique()), name="step")
    alive = snapshots[snapshots["alive"]]
    out: dict[str, np.ndarray] = {}
    pos, kind = {}, {}
    for side in SIDES:
        units = snapshots[snapshots["team"] == side]
        ids = np.sort(units["unit_id"].unique())
        kind[side] = units.groupby("unit_id")["unit_type"].first().reindex(ids).to_numpy()
        a = alive[alive["team"] == side]
        pos[side] = np.stack(
            [
                a.pivot(index="step", columns="unit_id", values=c)
                .reindex(index=steps, columns=ids)
                .to_numpy()
                for c in "xy"
            ],
            axis=-1,
        )  # step, unit, (x, y); NaN once dead
        hp = (units["hp"] + units["shield"]).to_numpy()
        h = (
            units.assign(h=hp)
            .pivot(index="step", columns="unit_id", values="h")
            .reindex(index=steps, columns=ids)
        )
        al = units.pivot(index="step", columns="unit_id", values="alive").reindex(index=steps, columns=ids)
        for t in types:
            cols = kind[side] == t
            st = unit_stats(t)
            out[f"health_{side}_{t.lower()}"] = h.loc[:, cols].sum(axis=1).to_numpy() / (
                REF_UNITS * (st["hp"] + st["shield"])
            )
            out[f"alive_{side}_{t.lower()}"] = (
                al.loc[:, cols].sum(axis=1).to_numpy().astype(float) / REF_UNITS
            )
    gap = np.linalg.norm(pos["blue"][:, :, None, :] - pos["red"][:, None, :, :], axis=-1)  # step, blue, red
    for side, other, g in (("blue", "red", gap), ("red", "blue", gap.transpose(0, 2, 1))):
        r = np.array(
            [[min(pair_reach(a, t), max_reach or np.inf) for t in kind[other]] for a in kind[side]]
        ).reshape(len(kind[side]), len(kind[other]))
        within = (g <= r[None]).any(axis=2)  # NaN compares False: dead units never count
        for t in types:
            out[f"reach_{side}_{t.lower()}"] = within[:, kind[side] == t].sum(axis=1) / REF_UNITS
    # measured fire allocation: attacker type -> target type, target in range
    tgt = alive[(alive["target_in_range"]) & alive["target_team"].notna()]
    type_of = {
        side: dict(
            zip(
                np.sort(snapshots.loc[snapshots["team"] == side, "unit_id"].unique()), kind[side], strict=True
            )
        )
        for side in SIDES
    }
    for side, other in (("blue", "red"), ("red", "blue")):
        s = tgt[tgt["team"] == side]
        ttype = s["target_id"].map(type_of[other])
        for a in types:
            for t in types:
                n = s[(s["unit_type"] == a) & (ttype == t)].groupby("step").size()
                out[f"fire_{side}_{a.lower()}_{t.lower()}"] = (
                    n.reindex(steps, fill_value=0).to_numpy() / REF_UNITS
                )
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = {side: np.nanmean(p, axis=1) for side, p in pos.items()}
        axis = centre["red"] - centre["blue"]
        dist = np.linalg.norm(axis, axis=1)
        norm = dist[:, None]
        axis = np.where(norm == 0, np.array([1.0, 0.0]), axis / norm)
        perp = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
        front = np.nanmin(gap, axis=(1, 2))
        for side, p in pos.items():
            rel = p - centre[side][:, None, :]
            out[f"depth_{side}"] = np.nanstd(np.einsum("sud,sd->su", rel, axis), axis=1)
            out[f"width_{side}"] = np.nanstd(np.einsum("sud,sd->su", rel, perp), axis=1)
            for t in types:  # how far forward the type sits in its formation (+ = towards the enemy)
                sign = 1.0 if side == "blue" else -1.0
                along = sign * np.einsum("sud,sd->su", rel[:, kind[side] == t], axis)
                out[f"front_{side}_{t.lower()}"] = (
                    np.nanmean(along, axis=1) if along.shape[1] else np.zeros(len(steps))
                )
    out["distance"], out["front_distance"] = dist, front
    df = pd.DataFrame(out, index=steps)
    carry = ["distance", *[c for c in df if c.startswith(("depth_", "width_", "front_"))]]
    df[carry] = df[carry].ffill()
    return df


def load_typed_states(
    dataset_dir, types: tuple[str, ...], episode_ids: list[int] | None = None, max_reach: float | None = None
) -> pd.DataFrame:
    """Per-type states of a dataset, indexed by (episode_idx, step), with t in seconds (`typed_states`)."""
    dt = storage.load_manifest(dataset_dir)["config"]["env"]["seconds_per_step"]
    if episode_ids is None:
        episode_ids = storage.load_episode_index(dataset_dir)["episode_idx"].tolist()
    parts = []
    for chunk in np.array_split(np.asarray(episode_ids), max(1, min(10, len(episode_ids)))):
        snaps = storage.load_snapshots(dataset_dir, chunk.tolist())
        parts.append(
            snaps.groupby("episode_idx").apply(
                typed_states, types=types, max_reach=max_reach, include_groups=False
            )
        )
    states = pd.concat(parts)
    states["t"] = states.index.get_level_values("step") * dt
    return states
