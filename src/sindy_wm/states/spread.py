"""Health-spread states per side and unit type (docs/planning/09_spread_state_plan.md).

Over the alive units of a side and type, with h_i = (hp + shield) / full:
    even_<side>_<type>      evenness c = (Σ h_i)² / (N Σ h_i²); 1 with one unit or none
    fragile_<side>_<type>   share of alive units with h_i < 0.5; 0 with none alive
"""

import numpy as np
import pandas as pd


def spread(snaps: pd.DataFrame, types: tuple[str, ...]) -> pd.DataFrame:
    s = snaps[["episode_idx", "step", "team", "unit_type", "alive", "hp", "shield", "max_hp", "max_shield"]]
    idx = pd.MultiIndex.from_frame(s[["episode_idx", "step"]].drop_duplicates())
    a = s[s["alive"]].copy()
    a["h"] = (a["hp"] + a["shield"]) / (a["max_hp"] + a["max_shield"])
    a["h2"] = a["h"] ** 2
    a["low"] = (a["h"] < 0.5).astype(float)
    g = a.groupby(["episode_idx", "step", "team", "unit_type"])[["h", "h2", "low"]].agg(["sum", "count"])
    n = g[("h", "count")]
    even = np.where(n >= 2, g[("h", "sum")] ** 2 / (n * g[("h2", "sum")]), 1.0)
    frag = g[("low", "sum")] / n
    stats = pd.DataFrame({"even": even, "fragile": frag.to_numpy()}, index=g.index)
    out = {}
    present = set(zip(stats.index.get_level_values(2), stats.index.get_level_values(3), strict=True))
    for side in ("blue", "red"):
        for t in types:
            e = pd.Series(1.0, index=idx)
            f = pd.Series(0.0, index=idx)
            if (side, t) in present:
                sub = stats.xs((side, t), level=(2, 3))
                e.loc[sub.index] = sub["even"].to_numpy()
                f.loc[sub.index] = sub["fragile"].to_numpy()
            out[f"even_{side}_{t.lower()}"] = e
            out[f"fragile_{side}_{t.lower()}"] = f
    return pd.DataFrame(out, index=idx)
