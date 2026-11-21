"""Health compartments per side and unit type (docs/planning/13_compartment_plan.md).

A unit is wounded while alive and below half of its own start health (its health at step 0 of the battle). Per
side and type, on the per-type tables' scales:
    W_<side>_<type>    wounded units / 10
    HW_<side>_<type>   health of the wounded units, in ten full units of the type
    h0_<side>_<type>   mean start health of the type's units, a fraction of full (constant per battle)
"""

import numpy as np
import pandas as pd

CUT = 0.5


def unit_health(snaps: pd.DataFrame) -> pd.Series:
    """h = (hp + shield) / full per snapshot row (0 for dead units)."""
    full = snaps["max_hp"] + snaps["max_shield"]
    return pd.Series(np.where(snaps["alive"], (snaps["hp"] + snaps["shield"]) / full, 0.0), index=snaps.index)


def compartments(snaps: pd.DataFrame, types: tuple[str, ...], start: pd.Series | None = None) -> pd.DataFrame:
    """Rows indexed (episode_idx, step). `start`: each unit's start health, indexed (episode_idx, team,
    unit_id);
    by default the unit's health at the first step of `snaps` in its episode."""
    s = snaps.assign(h=unit_health(snaps))
    if start is None:
        first = s[s["step"] == s.groupby("episode_idx")["step"].transform("min")]
        start = first.set_index(["episode_idx", "team", "unit_id"])["h"]
    s = s.join(start.rename("h_start"), on=["episode_idx", "team", "unit_id"])
    s["wounded"] = s["alive"] & (s["h"] < CUT * s["h_start"])
    idx = pd.MultiIndex.from_frame(s[["episode_idx", "step"]].drop_duplicates())
    out = {}
    for side in ("blue", "red"):
        for t in types:
            sub = s[(s["team"] == side) & (s["unit_type"] == t)]
            g = sub.groupby(["episode_idx", "step"])
            w = g["wounded"].sum().reindex(idx, fill_value=0) / 10
            hw = sub.assign(hw=sub["h"] * sub["wounded"]).groupby(["episode_idx", "step"])["hw"].sum()
            h0 = sub.groupby("episode_idx")["h_start"].mean()
            out[f"W_{side}_{t.lower()}"] = w.to_numpy(float)
            out[f"HW_{side}_{t.lower()}"] = hw.reindex(idx, fill_value=0).to_numpy(float) / 10
            out[f"h0_{side}_{t.lower()}"] = h0.reindex(idx.get_level_values(0)).fillna(1.0).to_numpy(float)
    return pd.DataFrame(out, index=idx)


def compartment_events(snaps: pd.DataFrame, types: tuple[str, ...]) -> pd.DataFrame:
    """Per-step events between step k and k + 1, per side and type, indexed (episode_idx, k)
    (docs/planning/15): I_ (alive healthy -> alive wounded), KW_ (wounded -> dead), KF_ (healthy -> dead),
    counts
    in units; D_ (health lost by the type's alive units) and DW_ (health lost by those wounded at k), full
    units."""
    s = snaps.assign(h=unit_health(snaps))
    first = s[s["step"] == s.groupby("episode_idx")["step"].transform("min")]
    s = s.join(
        first.set_index(["episode_idx", "team", "unit_id"])["h"].rename("h_start"),
        on=["episode_idx", "team", "unit_id"],
    )
    s["wounded"] = s["alive"] & (s["h"] < CUT * s["h_start"])
    s = s.sort_values(["episode_idx", "team", "unit_id", "step"])
    nxt = s.groupby(["episode_idx", "team", "unit_id"])[["alive", "wounded", "h"]].shift(-1)
    s = s.assign(
        alive1=nxt["alive"].astype("boolean"), wounded1=nxt["wounded"].astype("boolean"), h1=nxt["h"]
    )
    s = s[s["alive1"].notna()]
    a0, w0 = s["alive"], s["wounded"]
    a1, w1 = s["alive1"].astype(bool), s["wounded1"].astype(bool)
    s = s.assign(
        I=(a0 & ~w0 & a1 & w1).astype(float),
        KW=(w0 & ~a1).astype(float),
        KF=(a0 & ~w0 & ~a1).astype(float),
        D=np.where(a0, np.clip(s["h"] - s["h1"], 0, None), 0.0),
    )
    s["DW"] = np.where(w0, s["D"], 0.0)
    idx = pd.MultiIndex.from_frame(
        s[["episode_idx", "step"]].drop_duplicates().sort_values(["episode_idx", "step"])
    )
    out = {}
    for side in ("blue", "red"):
        for t in types:
            g = (
                s[(s["team"] == side) & (s["unit_type"] == t)]
                .groupby(["episode_idx", "step"])[["I", "KW", "KF", "D", "DW"]]
                .sum()
            )
            g = g.reindex(idx, fill_value=0.0)
            for c in g:
                out[f"{c}_{side}_{t.lower()}"] = g[c].to_numpy(float)
    return pd.DataFrame(out, index=idx)
