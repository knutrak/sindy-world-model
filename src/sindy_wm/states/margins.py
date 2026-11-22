"""Reach margins: each alive unit's distance to its nearest enemy minus their pair reach (capped at
6), and their mean and spread per side and unit type (docs/generalisation/09, 10). A unit is in reach
exactly when its margin is <= 0, so `in_<side>_<type>` / 10 equals the typed states' reach column."""

import warnings

import numpy as np
import pandas as pd

from sindy_wm.states.typed import pair_reach

SIDES = ("blue", "red")
MAX_REACH = 6.0


def margins(snapshots: pd.DataFrame, types) -> pd.DataFrame:
    """Per step and (side, type): alive count, mean and std of the reach margin, units with margin <= 0."""
    steps = pd.Index(sorted(snapshots["step"].unique()), name="step")
    alive = snapshots[snapshots["alive"]]
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
        )
    gap = np.linalg.norm(pos["blue"][:, :, None, :] - pos["red"][:, None, :, :], axis=-1)
    out = {}
    for side, other, g in (("blue", "red", gap), ("red", "blue", gap.transpose(0, 2, 1))):
        r = np.array([[min(pair_reach(a, t), MAX_REACH) for t in kind[other]] for a in kind[side]])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmin(g - r[None], axis=2)  # step, unit; NaN if the unit or every enemy is dead
        for t in types:
            mt = m[:, kind[side] == t]
            ok = ~np.isnan(mt)
            n = ok.sum(1)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"n_{side}_{t.lower()}"] = n
                out[f"mu_{side}_{t.lower()}"] = np.nanmean(mt, axis=1)
                out[f"sd_{side}_{t.lower()}"] = np.nanstd(mt, axis=1)
                out[f"in_{side}_{t.lower()}"] = (mt <= 0).sum(1)
    return pd.DataFrame(out, index=steps)
