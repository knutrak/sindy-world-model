# ruff: noqa: B023, E402, E501, UP031
"""Paired read-outs of `b_planner.py` against the recorded planners of docs/planning/35 and 37."""

import numpy as np
import pandas as pd

d = pd.read_csv("data/processed/planning/b_planner_S1.csv")
o = pd.read_csv("data/processed/planning/closed34_S1.csv")


def w(x, p):
    return x[x.policy == p].set_index("seed").won.astype(float)


print(
    "frozen rerun identical to recorded:",
    (w(d, "model:h3") == w(o, "model:h3").reindex(w(d, "model:h3").index)).mean(),
)
rng = np.random.default_rng(0)
W = {
    **{p: w(d, p) for p in d.policy.unique()},
    **{p: w(o, p) for p in ("fixed:hold/focus", "fixed:kite/nearest", "musigma:h3", "mlp:h3", "oracle5-h3")},
}


def pair(a, b):
    x = (W[a] - W[b]).dropna().to_numpy()
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return f"{x.mean():+.3f} [{np.quantile(m, 0.025):+.3f}, {np.quantile(m, 0.975):+.3f}]"


for a in ("modellabB:h3", "modellabA:h3", "modellabS:h3"):
    for b in ("model:h3", "fixed:hold/focus", "musigma:h3", "fixed:kite/nearest", "oracle5-h3", "mlp:h3"):
        print(a, "-", b, pair(a, b))
print("B - A", pair("modellabB:h3", "modellabA:h3"), "S - A", pair("modellabS:h3", "modellabA:h3"))
c = [x for x in d if x.startswith("pick_")] + ["switches", "seconds", "margin"]
print(d.groupby("policy")[[x for x in c if x in d]].mean().round(2).T)
