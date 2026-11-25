# ruff: noqa: B023, E402, E501, UP031
"""Intent ranking of A, B, S and the frozen model on the counterfactual sets (as experiments/planning/model_regret.py)."""

import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path[:0] = ["experiments/planning", "notebooks/lib"]
import model_regret as MR
import numpy as np
import pandas as pd
import sindyc_07_lab as lab

from sindy_wm.models.intent_model import DT

P = pickle.load(open("data/processed/exp06_intents/sindyc_07_models.pkl", "rb"))
F = pickle.load(open(MR.MODELS, "rb"))["marines"]
L = F.L
states, gap = MR.from_branch(MR.load("exp06_intents_typed"), L)
with np.errstate(all="ignore"):
    B = L.batch(states, horizon=round(8 / DT), every=10**9, history=0, min_future=1)
    YF = L.team_view(F.rollout(B))
t = B["team"]
TY = t["Y"]
b = {k: t[k] for k in ("X0", "U", "shape0", "reach0")}
b["shape0"] = np.nan_to_num(b["shape0"])
b["Y"] = TY
Y = {
    "frozen": YF,
    **{
        n: lab.simulate(p, b)
        for n, p in (
            ("A", P["A model_derived"]),
            ("B", P["B refit, same terms"]),
            ("S", P["S share as a state"]),
        )
    },
}
g = states.groupby(level=0, sort=False)
seed = g["seed"].first().loc[B["ep"]].to_numpy()
br = g["branch"].first().loc[B["ep"]].to_numpy()
rng = np.random.default_rng(0)
out = []
reg = {}
for h in (1.0, 2.0, 4.0):

    def piv(v):
        return (
            pd.DataFrame(dict(seed=seed, b=br, v=v)).pivot(index="seed", columns="b", values="v").to_numpy()
        )

    T = piv(MR.margin_change(TY, TY, h))
    for n, y in Y.items():
        M = piv(MR.margin_change(y, TY, h))
        tab, info = MR.rank_table(T, M, rng)
        r = T.max(1) - T[np.arange(len(T)), M.argmax(1)]
        reg[(n, h)] = r
        out.append(
            dict(
                h=h,
                model=n,
                regret=r.mean(),
                rank=tab.loc["model", "rank correlation"],
                best=tab.loc["model", "picks the true best"],
                always_advnear=(T.max(1) - T[:, 0]).mean(),
                **{k: round(v, 2) for k, v in info["model's picks"].items()},
            )
        )
    for n in ("A", "B", "S"):
        x = reg[(n, h)] - reg[("frozen", h)]
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        print(
            f"{n} - frozen regret at {h}: {x.mean():+.4f} [{np.quantile(m, 0.025):+.4f}, {np.quantile(m, 0.975):+.4f}]"
        )
    x = reg[("B", h)] - reg[("A", h)]
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    print(
        f"B - A regret at {h}: {x.mean():+.4f} [{np.quantile(m, 0.025):+.4f}, {np.quantile(m, 0.975):+.4f}]"
    )
print(len(T), "states")
print(pd.DataFrame(out).round(4).to_string(index=False))
