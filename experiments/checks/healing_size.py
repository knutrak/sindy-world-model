# ruff: noqa: B023, E402, E501, UP031
"""Effect size of a medivac-like 9 hp/s of healing next to the health rates the threshold is scaled by (train only)."""

import sys

sys.path.insert(0, "experiments/generalisation")
import common
import numpy as np
import pandas as pd

from sindy_wm.models.pipeline import Layout, prepare
from sindy_wm.paths import DATA_DIR

HP = {"marine": 45, "marauder": 125, "stalker": 160}
T3 = ("MARINE", "MARAUDER", "STALKER")
for s in ("marines", "mixed", "triple"):
    if s == "triple":
        f = DATA_DIR / "processed" / "exp12_triple_typed"
        c = f / "rates_train_w9.parquet"
        r = (
            pd.read_parquet(c)
            if c.exists()
            else prepare(pd.read_parquet(f / "train.parquet"), Layout(tuple(t.lower() for t in T3)))
        )
        tt = [t.lower() for t in T3]
    else:
        r = common.rows(s)
        tt = [t.lower() for t in common.SCENARIOS[s][1]]
    for side in ("blue", "red"):
        o = "red" if side == "blue" else "blue"
        eng = sum(r[f"reach_{o}_{a}"] for a in tt).to_numpy() > 0.05
        hurt = 0
        tot = sum(r[f"d_health_{side}_{t}"].to_numpy() * 10 * HP[t] for t in tt)
        print(
            s,
            side,
            "rows",
            len(r),
            f"engaged share {eng.mean():.2f}",
            f"| team loss hp/s when engaged: mean {-tot[eng].mean():.0f}, median {-np.median(tot[eng]):.0f}",
            "| last-quarter ->",
            "late engaged (<5 own alive): %.0f"
            % (-tot[eng & (sum(10 * r[f"alive_{side}_{t}"] for t in tt).to_numpy() < 5)].mean()),
        )
        for t in tt:
            y = r[f"d_health_{side}_{t}"].to_numpy()
            sd = y.std()
            h = 9 / (10 * HP[t])
            p = eng.mean()
            print(
                "   ",
                t,
                "sd(y) hp/s %.1f" % (sd * 10 * HP[t]),
                "| heal on/off with engagement, all 9 hp/s on this type: effect size %.2f"
                % (h * np.sqrt(p * (1 - p)) / sd),
                "| half of it: %.2f" % (0.5 * h * np.sqrt(p * (1 - p)) / sd),
                "| share 9/(mean loss engaged): %.2f" % (9 / max(-(tot[eng].mean()), 1e-9)),
            )
