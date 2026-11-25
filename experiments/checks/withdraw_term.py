# ruff: noqa: B023, E402, E501, UP031
"""Does SINDy find the withdraw effect, per scenario and threshold (train only)?"""

import sys

sys.path.insert(0, "experiments/generalisation")
sys.path.insert(0, "experiments/triple")
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common

from sindy_wm.models.intent_model import r2, stlsq
from sindy_wm.models.pipeline import Layout, Pipeline, prepare
from sindy_wm.paths import DATA_DIR

T3 = ("MARINE", "MARAUDER", "STALKER")


def get(s):
    if s in common.SCENARIOS:
        return common.rows(s), common.SCENARIOS[s][1]
    f = DATA_DIR / "processed" / "exp12_triple_typed" / "rates_train_w9.parquet"
    if not f.exists():
        st = pd.read_parquet(f.parent / "train.parquet")
        return prepare(st, Layout(tuple(t.lower() for t in T3))), T3
    return pd.read_parquet(f), T3


for s in ("marines", "marauders", "mixed", "triple"):
    rows, types = get(s)
    P = Pipeline(types, withdraw="fitted", alloc="rule")
    P0 = Pipeline(types, withdraw="rule", alloc="rule")
    tt = P.L.types
    wd = rows["withdraw"].to_numpy()
    R = {a: rows[f"reach_blue_{a}"].to_numpy() for a in tt}
    A = {a: rows[f"alive_blue_{a}"].to_numpy() for a in tt}
    sig = P._sigma_measured(rows, "blue")
    print(
        f"\n==== {s}: {len(rows)} rows, wd share {wd.mean():.3f}; R under wd {np.mean([R[a][wd > 0].mean() for a in tt]) * 10:.2f} units vs otherwise {np.mean([R[a][wd == 0].mean() for a in tt]) * 10:.2f}"
    )
    # direct: fire per unit in reach
    for a in tt:
        f = sum(rows[f"fire_blue_{a}_{t}"].to_numpy(float) for t in tt)
        for lab, m in (("not withdrawing", wd == 0), ("withdrawing", wd > 0)):
            print(
                f"  {a} firing per unit in reach, {lab}: {f[m].sum() / max(R[a][m].sum(), 1e-9):.3f}", end=";"
            )
    print()
    for t in tt:
        y = rows[f"d_health_red_{t}"].to_numpy()
        L = P._terms(R, sig, A, wd, "blue", t)
        names = list(L)
        th = np.column_stack(list(L.values()))
        for thr in (0.15, 0.10, 0.05, 0.02):
            c = dict(zip(names, stlsq(th, y, thr), strict=False))
            print(f"  H_R^{t}' thr {thr}: " + "  ".join(f"{v:+.3f} {k}" for k, v in c.items() if v))
        # forced physical terms only
        own = [n for n in names if n in [f"R[{a}]*s[{t}]" for a in tt] + [f"R[{a}]*s[{t}]*wd" for a in tt]]
        X = np.column_stack([L[n] for n in own])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        d = dict(zip(own, c, strict=False))
        es = {n: abs(v) * L[n].std() / y.std() for n, v in d.items()}
        for a in tt:
            k = d[f"R[{a}]*s[{t}]"]
            q = d[f"R[{a}]*s[{t}]*wd"]
            print(
                f"     forced LS {a}->{t}: k {k:+.3f}, kappa {q:+.3f}, kappa/k {-q / k:.2f}, effect size k {es[f'R[{a}]*s[{t}]']:.3f} kappa {es[f'R[{a}]*s[{t}]*wd']:.3f}"
            )
        Lr = P0._terms(R, sig, A, wd, "blue", t)
        ow = [f"R[{a}]*s[{t}]" for a in tt]
        Xr = np.column_stack([Lr[n] for n in ow])
        cr, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        Xn = np.column_stack([L[n] for n in ow])
        cn, *_ = np.linalg.lstsq(Xn, y, rcond=None)
        print(
            f"     R2: no wd {r2(y, Xn @ cn):.4f}  rule {r2(y, Xr @ cr):.4f}  fitted {r2(y, X @ c):.4f}; k rule {cr.round(3)}"
        )
