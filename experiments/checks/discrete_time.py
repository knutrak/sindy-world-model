# ruff: noqa: E501, E702, I001
"""Continuous or discrete time? (exploratory, 25 Nov 2026; fit on train, score on val, marines and mixed)

C    the frozen pipeline models: SG-smoothed derivatives, Euler steps of 0.125 s
Dm   the same candidates and closures fitted on raw differences (x[k+m] - x[k]) / (m dt) with the regressors at k, only
     windows with one intent; stepped with m dt (m = 1, 4, 8: 0.125, 0.5, 1 s). Health and distance cutoffs equal C's
     in absolute size (STLSQ's threshold is relative to the target's spread, and raw differences have more spread).
D1t  D1 with the original relative thresholds
Cm   C stepped with m dt
Rule, fixed before the run: discrete time is better if Dm's health error at 2 s is below C's (paired 95% interval below 0,
bootstrap over battles) in both scenarios.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in ("experiments/generalisation", "experiments/mixed", "scripts"):
    sys.path.insert(0, str(ROOT / p))
from common import layout, rows, states  # noqa: E402
from frozen_models import equations  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models import pipeline as PL  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "docs" / "checks" / "discrete_time.md"
DT = 0.125
CFG = yaml.safe_load((ROOT / "configs/generalisation_pipeline.yaml").read_text())
KW = {
    k: CFG["pipeline"][k]
    for k in ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd")
}


def difference_rows(st: pd.DataFrame, smooth: pd.DataFrame, L, m: int) -> pd.DataFrame:
    """The smoothed table's rows with its d_ columns replaced by forward differences over m steps."""
    s = add_controls(st)
    cols = [*L.x_cols]
    g = s.groupby(level=0, sort=False)
    diff = (g[cols].shift(-m) - s[cols]) / (m * DT)
    same = np.ones(len(s), bool)
    key = s["movement"].astype(str) + s["target"].astype(str)
    for j in range(
        1, m
    ):  # one intent through the window (the control of step k + m - 1 is the last one used)
        same &= (key.groupby(level=0, sort=False).shift(-j) == key).to_numpy()
    r = smooth.copy()
    assert r.index.isin(s.index).all()
    d = diff.reindex(r.index)
    for c in cols:
        r[f"d_{c}"] = d[c].to_numpy()
    ok = pd.Series(same, index=s.index).reindex(r.index).to_numpy() & d.notna().all(axis=1).to_numpy()
    return r[ok]


class AbsoluteCut(Pipeline):
    """STLSQ with the same absolute cutoff as the smoothed fit: threshold x (smoothed spread / raw spread)."""

    smooth_sd: dict = {}

    def fit(self, r):
        tt = self.L.types
        sds = [
            self.smooth_sd[f"d_health_{s}_{t}"] / r[f"d_health_{s}_{t}"].std() for s in PL.SIDES for t in tt
        ]
        self._q = list(sds)
        self.thr_distance = self.thr_distance * self.smooth_sd["d_distance"] / r["d_distance"].std()
        return super().fit(r)

    def _health_stlsq(self, theta, y, names):
        return PL.stlsq(theta, y, self.thr_health * self._q.pop(0))


def coarse(B: dict, m: int) -> dict:
    if m == 1:
        return B
    U = B["U"][:, : (B["U"].shape[1] // m) * m]
    return {**B, "U": U.reshape(len(U), -1, m, U.shape[2]).mean(2)}


def roll(model, B, m):
    old = PL.DT
    PL.DT = m * DT
    try:
        return model.L.team_view(model.rollout(coarse(B, m)))
    finally:
        PL.DT = old


def errs(Y, T, m, h):
    """Mean absolute health error per start point (both sides), and margin change real / predicted."""
    n = round(h / DT)
    ok = ~np.isnan(T["Y"][:, n, 0])
    y, t, x0 = Y[ok, n // m], T["Y"][ok, n], T["X0"][ok]
    e = np.abs(y[:, :2] - t[:, :2]).mean(1)
    real = (t[:, 0] - t[:, 1]) - (x0[:, 0] - x0[:, 1])
    pred = (y[:, 0] - y[:, 1]) - (x0[:, 0] - x0[:, 1])
    r2 = 1 - ((real - pred) ** 2).sum() / ((real - real.mean()) ** 2).sum()
    return e, r2, T["ep"][ok], np.abs(y[:, 4] - t[:, 4]).mean(), 10 * np.abs(y[:, 2:4] - t[:, 2:4]).mean()


def ci(d, ep, n=1000):
    g = pd.DataFrame({"d": d, "ep": ep}).groupby("ep")["d"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(g), (n, len(g)))
    return np.percentile(g[i, 0].sum(1) / g[i, 1].sum(1), [2.5, 97.5])


def main():
    frozen = pickle.load(open(H.CACHE / "generalisation_frozen.pkl", "rb"))
    lines, verdict = (
        ["# Continuous or discrete time", "", "Exploratory (val).", "", "```", __doc__.strip(), "```", ""],
        {},
    )
    for sc in ("marines", "mixed"):
        t0 = time.time()
        L, s = layout(sc), CFG["scenarios"][sc]
        smooth, st = rows(sc), states(sc, "train")
        sd = {c: float(smooth[c].std()) for c in smooth if c.startswith("d_")}
        models = {"C": (frozen[sc], 1), "C4": (frozen[sc], 4), "C8": (frozen[sc], 8)}
        for m in (1, 4, 8):
            r = difference_rows(st, smooth, L, m)
            a = AbsoluteCut(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **KW)
            a.smooth_sd = sd
            models[f"D{m}"] = (a.fit(r), m)
            if m == 1:
                models["D1t"] = (
                    Pipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **KW).fit(r),
                    1,
                )
            print(sc, "fitted", m, len(r), f"{time.time() - t0:.0f} s", flush=True)
        B = L.batch(add_controls(states(sc, "val")))
        T = B["team"]
        Ys = {k: roll(mm, B, m) for k, (mm, m) in models.items()}
        base = {h: errs(Ys["C"], T, 1, h) for h in (1, 2, 4)}
        tab = []
        for k, (_, m) in models.items():
            row = {"model": k, "step s": m * DT}
            for h in (1, 2, 4):
                e, r2, ep, dist, al = errs(Ys[k], T, m, h)
                row |= {f"health {h}s": e.mean(), f"margin R2 {h}s": r2}
                if h == 2:
                    row |= {"alive 2s": al, "distance 2s": dist}
                lo, hi = ci(e - base[h][0], ep)
                row[f"- C, {h}s 95%"] = f"[{lo:+.4f}, {hi:+.4f}]"
                if h == 2:
                    verdict[(sc, k)] = bool(hi < 0)
            tab.append(row)
        lines += [
            f"## {sc}",
            "",
            "```",
            pd.DataFrame(tab).set_index("model").round(4).T.to_string(),
            "```",
            "",
        ]
        for k in ("C", "D1", "D1t", "D4", "D8"):
            lines += [f"{k}:", "", "```", equations(models[k][0]), "```", ""]
    lines += ["## Rule (health error at 2 s below C's in both scenarios)", ""]
    for k in ("D1", "D1t", "D4", "D8", "C4", "C8"):
        lines.append(f"- {k}: marines {verdict[('marines', k)]}, mixed {verdict[('mixed', k)]}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
