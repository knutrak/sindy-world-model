"""Why do marauders in reach fire less? Firing share by distance to the nearest enemy (val, 200 battles;
Blue excluding withdraw). See docs/mixed/PROGRESS.md, 20 Nov.

    uv run experiments/mixed/firing_share.py
"""

import numpy as np
import pandas as pd

from sindy_wm.io import storage
from sindy_wm.paths import RAW_DIR

rows = []
for exp, reach in (("exp06_intents", 5.75), ("exp09_marauders", 7.125)):
    d = RAW_DIR / exp / "val"
    snaps = storage.load_snapshots(d, list(range(200)))
    ctrl = storage.load_controls(d)[["episode_idx", "step", "movement"]]
    snaps = snaps.merge(ctrl, on=["episode_idx", "step"], how="left")
    snaps["movement"] = snaps.groupby("episode_idx")["movement"].ffill()
    for _key, g in snaps[snaps.alive].groupby(["episode_idx", "step"]):
        b, r = g[g.team == "blue"], g[g.team == "red"]
        if b.empty or r.empty:
            continue
        D = np.linalg.norm(b[["x", "y"]].to_numpy()[:, None] - r[["x", "y"]].to_numpy()[None], axis=-1)
        for side, units, dmin in (("blue", b, D.min(1)), ("red", r, D.min(0))):
            for i, (_, u) in enumerate(units.iterrows()):
                if dmin[i] <= reach and (side == "red" or u.movement != "withdraw"):
                    rows.append(
                        {
                            "units": exp.split("_")[0],
                            "side": side,
                            "nearest enemy": pd.cut([dmin[i]], [0, 5.375, 5.5625, 6.0, 7.125])[0],
                            "firing": bool(u.target_in_range),
                        }
                    )
df = pd.DataFrame(rows)
print(
    df.pivot_table(
        index=["units", "nearest enemy"],
        columns="side",
        values="firing",
        aggfunc=["mean", "size"],
        observed=True,
    )
    .round(3)
    .to_string()
)
