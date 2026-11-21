"""Does formation limit the frozen model? (exploratory diagnostic, docs/planning/24_formation_and_s3.md)

On val (S1 exp06, S2 exp08): the frozen model's error in the margin change at 2 and 4 s, by Red's formation
depth
at the start; then the same rollouts given the *true* formation shape (depth, width per side) at every step
instead
of holding it fixed. The share of the error that disappears is an upper bound on what modelling the formation
could gain for these predictions.

    uv run --extra explore experiments/planning/formation_check.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_regret import MODELS  # noqa: E402

from sindy_wm.models.intent_model import DT, SHAPE, add_controls  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402

SETS = {"S1": ("exp06_intents_typed", "marines"), "S2": ("exp08_mixed_typed", "mixed")}


def main() -> list[str]:
    models = pickle.load(open(MODELS, "rb"))
    lines = []
    for scen, (folder, name) in SETS.items():
        m = models[name]
        L = m.L
        st = add_controls(pd.read_parquet(DATA_DIR / "processed" / folder / "val.parquet"))
        with np.errstate(invalid="ignore", divide="ignore"):
            B = L.batch(st, horizon=32)
            Y0 = L.team_view(m.rollout(B))
            shp = {ep: g[SHAPE].to_numpy(float) for ep, g in st.groupby(level=0, sort=False)}
            traj = np.full((len(B["ep"]), 33, 4), np.nan)
            for i, (e, k) in enumerate(zip(B["ep"], B["k0"], strict=True)):
                a = shp[e][k : k + 33]
                traj[i, : len(a)] = a
                traj[i, len(a) :] = a[-1]
            Y1 = L.team_view(m.rollout(B | {"shape_traj": traj}))
        T = B["team"]["Y"]
        depth = B["shape0"][:, SHAPE.index("depth_red")]
        q = pd.qcut(depth, 4, labels=["Red depth Q1 (compact)", "Q2", "Q3", "Q4 (strung out)"])
        rows = []
        for h in (2.0, 4.0):
            n = round(h / DT)
            ok = ~np.isnan(T[:, n, 0])
            true = (T[ok, n, 0] - T[ok, n, 1]) - (T[ok, 0, 0] - T[ok, 0, 1])
            p0 = (Y0[ok, n, 0] - Y0[ok, n, 1]) - (Y0[ok, 0, 0] - Y0[ok, 0, 1])
            p1 = (Y1[ok, n, 0] - Y1[ok, n, 1]) - (Y1[ok, 0, 0] - Y1[ok, 0, 1])
            f = pd.DataFrame(
                {"q": np.asarray(q)[ok], "e0": np.abs(p0 - true), "e1": np.abs(p1 - true), "bias0": p0 - true}
            )
            for lab, g in [("all", f), *f.groupby("q", observed=True)]:
                rows.append(
                    {
                        "h": h,
                        "start states": lab,
                        "n": len(g),
                        "margin error, shape held (frozen)": g["e0"].mean(),
                        "with the true shape": g["e1"].mean(),
                        "error removed": 1 - g["e1"].mean() / g["e0"].mean(),
                        "bias, frozen (pred − true)": g["bias0"].mean(),
                    }
                )
        t = pd.DataFrame(rows).set_index(["h", "start states"])
        lines += [f"## {scen} ({name} model, val)", "", "```", t.round(4).to_string(), "```", ""]
    print("\n".join(lines))
    return lines


if __name__ == "__main__":
    main()
