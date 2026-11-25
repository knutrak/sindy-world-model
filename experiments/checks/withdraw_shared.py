# ruff: noqa: B023, E402, E501, UP031
"""One shared withdraw term per equation against the 0.15 threshold (train only)."""

import sys

sys.path.insert(0, "experiments/generalisation")
import common
import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import stlsq
from sindy_wm.models.pipeline import Layout, Pipeline, prepare
from sindy_wm.paths import DATA_DIR

T3 = ("MARINE", "MARAUDER", "STALKER")
for s in ("marines", "marauders", "mixed", "triple"):
    if s in common.SCENARIOS:
        rows, types = common.rows(s), common.SCENARIOS[s][1]
    else:
        f = DATA_DIR / "processed" / "exp12_triple_typed"
        types = T3
        rows = (
            pd.read_parquet(f / "rates_train_w9.parquet")
            if (f / "rates_train_w9.parquet").exists()
            else prepare(pd.read_parquet(f / "train.parquet"), Layout(tuple(t.lower() for t in T3)))
        )
    P = Pipeline(types, withdraw="fitted", alloc="rule")
    tt = P.L.types
    wd = rows["withdraw"].to_numpy()
    R = {a: rows[f"reach_blue_{a}"].to_numpy() for a in tt}
    A = {a: rows[f"alive_blue_{a}"].to_numpy() for a in tt}
    sig = P._sigma_measured(rows, "blue")
    for t in tt:
        y = rows[f"d_health_red_{t}"].to_numpy()
        L = P._terms(R, sig, A, wd, "blue", t)
        own = [f"R[{a}]*s[{t}]" for a in tt]
        X = np.column_stack([L[n] for n in own])
        m = wd == 0
        k, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)  # k from non-withdraw steps only
        F = X @ k  # predicted loss rate at full fire
        G = {"Fire": F, "Fire*wd": F * wd}
        Z = np.column_stack(list(G.values()))
        c, *_ = np.linalg.lstsq(Z, y, rcond=None)
        st = stlsq(Z, y, 0.15)
        print(
            s,
            t,
            f"fire kept under wd {1 + c[1] / c[0]:.2f} | effect size of Fire*wd {abs(c[1]) * (F * wd).std() / y.std():.3f} | STLSQ 0.15 picks",
            [n for n, v in zip(G, st, strict=False) if v],
        )
    # pooled over target types (team health loss, weighted): one shared withdraw factor
