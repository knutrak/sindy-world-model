"""POST-HOC (19 Nov): the 17 Nov gates applied to "A*, separate k" (= G/alive-reach): A's separate
firepower coefficients k_B, k_R (STLSQ) with A*'s alive law (A' = c R_enemy A / H, p = c / k).

Same procedure and sets as scripts/gates_post_hoc.py (val, cf_*, exp07 val; G3a 30 battle-resampled
refits with closures fixed; G3b SG 5 / 9 / 17 and weak 1 s / 2 s). Differences:
- the health equations are A's STLSQ fit (SG) / A's weak STLSQ fit (weak form), not one shared k;
- G7 is not evaluated: the model was not run on test2 (spent), so no held-out / val ratio exists.

    uv run --extra explore scripts/gates_separate_k.py
Writes docs/final_exploration/05_gates_separate_k.md and .json.
"""

import copy
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "notebooks" / "lib"))

import evaluate_final as E  # noqa: E402
import gates_post_hoc as G  # noqa: E402
from weak_sindy import bump, fit_weak, weak_rows  # noqa: E402

from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.evaluation.harness import FitData  # noqa: E402
from sindy_wm.models.intent_model import DT  # noqa: E402

DOCS = REPO / "docs" / "final_exploration"
N_REFITS = 30
NAME = "A*, separate k"


def fit_sep(rows: pd.DataFrame, P: dict):
    return AliveVariantA(law="reach", P=copy.deepcopy(P)).fit(FitData(states=None, rates=rows)).m


def weak_sep(rows: pd.DataFrame, A, radius: int):
    """A's weak-form fit (health and distance by weak STLSQ), alive law replaced by the weak reach law."""
    W = weak_rows(rows, A.P, A.names, radius=radius)
    m = copy.deepcopy(fit_weak(A, W))
    names = m.names
    k = {"blue": m.coef["health_blue"][names.index("Rr")], "red": m.coef["health_red"][names.index("Rb")]}
    phi, dphi, w = bump(radius)
    lhs, reg = {"blue": [], "red": []}, {"blue": [], "red": []}
    for _, g in rows.groupby(level=0, sort=False):
        X = g[["health_blue", "health_red", "alive_blue", "alive_red"]].to_numpy(float)
        R = g[["reach_blue", "reach_red"]].to_numpy(float)
        foc = g["focus"].to_numpy(float)
        both = (X[:, 2] > 0) & (X[:, 3] > 0) & (X[:, 0] > 0.03) & (X[:, 1] > 0.03)
        for c in range(radius, len(X) - radius, 2):
            sl = slice(c - radius, c + radius + 1)
            if not both[sl].all():
                continue
            for j, (side, enemy_j) in enumerate((("blue", 1), ("red", 0))):
                x = R[sl, enemy_j] * X[sl, 2 + j] / X[sl, j]
                lhs[side].append(-(w * dphi) @ X[sl, 2 + j])
                reg[side].append([(w * phi) @ x, (w * phi) @ (x * foc[sl])])
    for side in ("blue", "red"):
        Rg = np.array(reg[side])
        T = Rg[:, :1] if side == "blue" else Rg
        c, *_ = np.linalg.lstsq(T, np.array(lhs[side]), rcond=None)
        p = c / k[side]
        m.p[side] = {"1": float(p[0]), "hold": 0.0, "wd": 0.0, "focus": float(p[1]) if side == "red" else 0.0}
    return m


def gate3b_sep(A, raw_rows: pd.DataFrame) -> tuple[bool, dict]:
    fits = {}
    for wdw in (5, 9, 17):
        fits[f"SG {wdw}"] = E.terms(fit_sep(E.rows_with_rates("train", window=wdw), A.P))
    for radius in (8, 16):
        fits[f"weak {radius * DT:g} s"] = E.terms(weak_sep(raw_rows, A, radius))
    sets = {k: {eq: sorted(v) for eq, v in T.items()} for k, T in fits.items()}
    same = all(s == sets["SG 9"] for s in sets.values())
    within, spread = True, {}
    for eq in ("health_blue", "health_red"):
        for term in fits["SG 9"][eq]:
            vals = np.array([T[eq].get(term, 0.0) for T in fits.values()])
            med = np.median(vals)
            dev = float(np.max(np.abs(vals - med) / abs(med)))
            spread[f"{eq}: {term}"] = {"values": vals.round(4).tolist(), "max deviation from median": dev}
            within &= dev <= 0.10
    alive = {
        k: {eq: {t: round(float(v), 4) for t, v in T[eq].items()} for eq in ("alive_blue", "alive_red")}
        for k, T in fits.items()
    }
    return same and within, {
        "same term set": same,
        "term sets that differ": {k: s for k, s in sets.items() if s != sets["SG 9"]},
        "health coefficients": spread,
        "alive exponents": alive,
    }


_CTX: dict = {}


def _init(P):
    _CTX["rows"] = E.rows_with_rates("train")
    _CTX["P"] = P


def _refit(seed):
    return E.terms(fit_sep(E._resample_rows(_CTX["rows"], seed), _CTX["P"]))


def gate3a_sep(T: dict, P: dict, workers: int) -> tuple[bool, dict]:
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(P,)) as pool:
        refits = pool.map(_refit, [1000 + i for i in range(N_REFITS)])
    detail, ok = {}, True
    for eq, ts in T.items():
        for term in ts:
            kept = [v for v in (r.get(eq, {}).get(term) for r in refits) if v]
            share = len(kept) / len(refits)
            lo, hi = np.quantile(kept, [0.025, 0.975]) if kept else (np.nan, np.nan)
            passed = share >= 0.9 and bool(kept) and (lo > 0 or hi < 0)
            detail[f"{eq}: {term}"] = {"kept": share, "range": [float(lo), float(hi)], "pass": bool(passed)}
            ok &= passed
    return ok, detail


def main() -> None:
    t0 = time.time()
    rows = E.rows_with_rates("train")
    A = E.load_models(rows)["A"]
    M = fit_sep(rows, A.P)
    Astar = G.fit_astar(rows, A.P)
    base = E.baselines(rows)
    Bf, Bcf, Bs = E.batch("val"), E.batch("cf"), E.batch("val", "exp07_stalkers")
    Y, Ycf, Ys = E.rollout(M, Bf), E.rollout(M, Bcf), E.rollout(M, Bs, stalkers=True)
    Yb = {k: E.baseline_rollout(b, Bf) for k, b in base.items()}
    Ybs = {
        k: E.baseline_rollout(b, Bs, stalkers=True)
        for k, b in base.items()
        if k in ("persistence", "no control")
    }
    T = E.terms(M)
    r = {"terms": {eq: {k: float(v) for k, v in ts.items()} for eq, ts in T.items()}}
    r["G1"], r["G1 detail"] = E.gate1(T)
    r["G2"], r["G2 detail"] = E.gate2(T)
    r["G3a"], r["G3a detail"] = gate3a_sep(T, A.P, 9)
    print(f"G3a done ({time.time() - t0:.0f} s)", flush=True)
    r["G3b"], r["G3b detail"] = gate3b_sep(A, rows)
    print(f"G3b done ({time.time() - t0:.0f} s)", flush=True)
    r["G3"] = r["G3a"] and r["G3b"]
    ok_f, d_f = E.gate4(Y, Bf)
    ok_c, d_c = E.gate4(Ycf, Bcf)
    r["G4"], r["G4 detail"] = ok_f and ok_c, {"val": d_f, "cf": d_c}
    cis = {b: E.ci_health_diff(Yb[b], Y, Bf) for b in Yb}
    r["G5"], r["G5 detail"] = all(ci[1] < 0 for ci in cis.values()), cis
    hf, rf = E.horizon(Y, Bf)
    hc, rc = E.horizon(Ycf, Bcf)
    r["G6"], r["G6 detail"] = {"val": hf, "cf": hc}, {"val": rf, "cf": rc}
    r["G7"], r["G7 detail"] = None, "not evaluated: no test2 run for this model (test2 is spent)"
    cis_s = {b: E.ci_health_diff(Ybs[b], Ys, Bs) for b in Ybs}
    r["G8"], r["G8 detail"] = all(ci[1] < 0 for ci in cis_s.values()), cis_s
    # paired against A* on val (health 4 s and margin 4 s)
    Ya = E.rollout(Astar, Bf)
    pb = E.lab.paired_bootstrap(Ya, Y, Bf, h=4, n_boot=E.N_BOOT_CI)
    r["vs A* (val, paired 95%, this − A*)"] = {k: [float(v[0]), float(v[1])] for k, v in pb.items()}
    interp = all(bool(r[g]) for g in ("G1", "G2", "G3", "G4", "G5"))
    r["verdict"] = "valid interpretable model (G7 not evaluated)" if interp else "not valid"
    r["transfers"] = bool(r["G8"]) and interp
    (DOCS / "05_gates_separate_k.json").write_text(json.dumps({NAME: r}, indent=1, default=E._plain))
    table = {g: r[g] for g in ("G1", "G2", "G3a", "G3b", "G4", "G5", "G7", "G8")} | {
        "G6 horizon val / cf": f"{hf} / {hc}",
        "verdict": r["verdict"],
        "transfers": r["transfers"],
    }
    lines = [
        f'# The 17 Nov gates applied to "{NAME}" (POST-HOC, 19 Nov)',
        "",
        "Same procedure and sets as `04_gates_post_hoc.md` (val, cf_*, exp07 val). G7 not evaluated (no",
        "test2 run). Script: `scripts/gates_separate_k.py`; every number: `05_gates_separate_k.json`.",
        "",
        "```",
        pd.Series(table).to_string(),
        "```",
        "",
        f"- terms: {r['terms']}",
        f"- G2 sign failures: {r['G2 detail'] or 'none'}",
        f"- G3a terms failing: {({k: v for k, v in r['G3a detail'].items() if not v['pass']}) or 'none'}",
        f"- G3b: {r['G3b detail']}",
        f"- G5: {r['G5 detail']}",
        f"- G6: {r['G6']}",
        f"- G8: {r['G8 detail']}",
        f"- vs A* on val: {r['vs A* (val, paired 95%, this − A*)']}",
        "",
    ]
    (DOCS / "05_gates_separate_k.md").write_text("\n".join(lines))
    print(pd.Series(table).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
