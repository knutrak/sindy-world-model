"""POST-HOC: apply the 17 Nov acceptance gates (docs/2026-11-17_acceptance-gates.md) to A* and A.

Not a pre-registered test. The `final` sets were spent on 17 Nov and `test2` on 18 Nov, so this runs
the gates' definitions on the non-final sets, as `evaluate_final.py --dry-run` does:
exp06 `val` in place of `final`, `cf_*` in place of `final_cf`, exp07 `val` in place of exp07 `final`.
A goes through the identical procedure as a control, so differences come from the model.

What differs from evaluate_final.py, for A* only (its fit has no STLSQ step for k and p):
- G3a: 30 refits of the A* recipe on battle-resampled `train` (closures fixed, as for A).
- G3b: A*'s own fit on SG 5 / 9 / 17 rates and in weak form (1 s, 2 s windows, the same bump test
  functions as notebooks/lib/weak_sindy.py): shared k by least squares on the stacked weak rows; the
  alive law A' = c R_enemy A / H integrated against the test functions (p = c / k); the distance
  equation by STLSQ as in `fit_weak`.
- G7 compares held-out with val, which is meaningless when held-out *is* val. It uses the test2 / val
  ratio of health error at 4 s already recorded by the Phase 3 run (03_test_results.json; no new
  reads of test2).

    uv run --extra explore scripts/gates_post_hoc.py
Writes docs/final_exploration/04_gates_post_hoc.md and .json.
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
from weak_sindy import bump, fit_weak, weak_rows  # noqa: E402

from sindy_wm.evaluation.explore_models import CombinedA  # noqa: E402
from sindy_wm.evaluation.harness import FitData  # noqa: E402
from sindy_wm.models.intent_model import DT, MOVE_FAMILY, _family_cols, stlsq  # noqa: E402

DOCS = REPO / "docs" / "final_exploration"
N_REFITS = 30


def fit_astar(rows: pd.DataFrame, P: dict):
    m = CombinedA(P=copy.deepcopy(P)).fit(FitData(states=None, rates=rows))
    return m.m


# ------------------------------------------------------------------------------------- weak A*
def weak_astar(rows: pd.DataFrame, A, radius: int):
    """A* fitted in weak form (see module docstring)."""
    W = weak_rows(rows, A.P, A.names, radius=radius)
    m = copy.deepcopy(fit_weak(A, W))  # distance (and A's alive) by weak STLSQ; health / alive replaced below
    names, G = A.names, W["G"]
    iRr, iRb, iRbwd = names.index("Rr"), names.index("Rb"), names.index("Rb*wd")
    y = np.concatenate([W["b_hb"], W["b_hr"]])
    T = np.column_stack(
        [np.concatenate([G[:, iRr], G[:, iRb]]), np.concatenate([0 * G[:, iRr], G[:, iRbwd]])]
    )
    (k, kwd), *_ = np.linalg.lstsq(T, y, rcond=None)
    for eq in ("health_blue", "health_red"):
        m.coef[eq] = np.zeros(len(names))
    m.coef["health_blue"][iRr] = k
    m.coef["health_red"][iRb] = k
    m.coef["health_red"][iRbwd] = kwd
    cols = [j for j in _family_cols(names, MOVE_FAMILY) if names[j] not in A.exclude]
    m.coef["distance"] = np.zeros(len(names))
    m.coef["distance"][cols] = stlsq(G[:, cols], W["b_d"], 0.05)
    # alive: -∫φ' A = c ∫φ R_enemy A / H (+ c_f ∫φ R_enemy A / H focus for Red)
    phi, dphi, w = bump(radius)
    lhs = {"blue": [], "red": []}
    reg = {"blue": [], "red": []}
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
    for side, kk in (("blue", k), ("red", k)):
        Rg = np.array(reg[side])
        T = Rg[:, :1] if side == "blue" else Rg
        c, *_ = np.linalg.lstsq(T, np.array(lhs[side]), rcond=None)
        p = c / kk
        m.p[side] = {"1": float(p[0]), "hold": 0.0, "wd": 0.0, "focus": float(p[1]) if side == "red" else 0.0}
    return m


def gate3b_astar(A, raw_rows: pd.DataFrame) -> tuple[bool, dict]:
    fits = {}
    for wdw in (5, 9, 17):
        fits[f"SG {wdw}"] = E.terms(fit_astar(E.rows_with_rates("train", window=wdw), A.P))
    for radius in (8, 16):
        fits[f"weak {radius * DT:g} s"] = E.terms(weak_astar(raw_rows, A, radius))
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


# ---------------------------------------------------------------------------------------- G3a
_CTX: dict = {}


def _init(P):
    _CTX["rows"] = E.rows_with_rates("train")
    _CTX["P"] = P


def _refit(seed):
    return E.terms(fit_astar(E._resample_rows(_CTX["rows"], seed), _CTX["P"]))


def gate3a_astar(T: dict, P: dict, workers: int) -> tuple[bool, dict]:
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(P,)) as pool:
        refits = pool.map(_refit, [1000 + i for i in range(N_REFITS)])
    detail, ok = {}, True
    for eq, ts in T.items():
        for term in ts:
            kept = [r.get(eq, {}).get(term) for r in refits]
            kept = [v for v in kept if v]
            share = len(kept) / len(refits)
            lo, hi = np.quantile(kept, [0.025, 0.975]) if kept else (np.nan, np.nan)
            passed = share >= 0.9 and bool(kept) and (lo > 0 or hi < 0)
            detail[f"{eq}: {term}"] = {"kept": share, "range": [float(lo), float(hi)], "pass": bool(passed)}
            ok &= passed
    return ok, detail


# --------------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    rows = E.rows_with_rates("train")
    A = E.load_models(rows)["A"]  # model_derived; load_models also fits W, unused here
    Astar = fit_astar(rows, A.P)
    models = {"A": A, "A*": Astar}
    base = E.baselines(rows)
    Bf, Bcf, Bs = E.batch("val"), E.batch("cf"), E.batch("val", "exp07_stalkers")
    Y = {k: E.rollout(m, Bf) for k, m in models.items()}
    Ycf = {k: E.rollout(m, Bcf) for k, m in models.items()}
    Ys = {k: E.rollout(m, Bs, stalkers=True) for k, m in models.items()}
    Yb = {k: E.baseline_rollout(b, Bf) for k, b in base.items()}
    Ybs = {
        k: E.baseline_rollout(b, Bs, stalkers=True)
        for k, b in base.items()
        if k in ("persistence", "no control")
    }
    test = json.loads((DOCS / "03_test_results.json").read_text())["splits"]["test2"]["metrics"]
    val_rows = pd.read_csv(DOCS / "results.csv")
    val_rows = val_rows[(val_rows["split"] == "val") & val_rows["experiment"].isin(["ref/A", "deep/A*"])]
    val_h = val_rows.groupby("experiment")["health_4s"].last()

    E._CTX["rows"] = rows  # evaluate_final.gate3b reads its weak-form rows from here (as on 17 Nov)
    g3b_A = E.gate3b({"A": A})
    g3b_S = gate3b_astar(A, rows)
    result = {}
    for name, m in models.items():
        print(f"-- {name} ({time.time() - t0:.0f} s)", flush=True)
        T = E.terms(m)
        r = {"terms": {eq: {k: float(v) for k, v in ts.items()} for eq, ts in T.items()}}
        r["G1"], r["G1 detail"] = E.gate1(T)
        r["G2"], r["G2 detail"] = E.gate2(T)
        if name == "A":
            r["G3a"], r["G3a detail"] = E.gate3a("A", T, 9, N_REFITS)
            r["G3b"], r["G3b detail"] = g3b_A
        else:
            r["G3a"], r["G3a detail"] = gate3a_astar(T, A.P, 9)
            r["G3b"], r["G3b detail"] = g3b_S
        r["G3"] = r["G3a"] and r["G3b"]
        ok_f, d_f = E.gate4(Y[name], Bf)
        ok_c, d_c = E.gate4(Ycf[name], Bcf)
        r["G4"], r["G4 detail"] = ok_f and ok_c, {"val": d_f, "cf": d_c}
        cis = {b: E.ci_health_diff(Yb[b], Y[name], Bf) for b in Yb}
        r["G5"], r["G5 detail"] = all(ci[1] < 0 for ci in cis.values()), cis
        hf, rf = E.horizon(Y[name], Bf)
        hc, rc = E.horizon(Ycf[name], Bcf)
        r["G6"], r["G6 detail"] = {"val": hf, "cf": hc}, {"val": rf, "cf": rc}
        key = {"A": "ref/A", "A*": "deep/A*"}[name]
        e_t, e_v = float(test[key]["health_4s"]), float(val_h[key])
        r["G7"], r["G7 detail"] = e_t <= 1.1 * e_v, {"health 4 s test2": e_t, "val": e_v, "ratio": e_t / e_v}
        cis_s = {b: E.ci_health_diff(Ybs[b], Ys[name], Bs) for b in Ybs}
        r["G8"], r["G8 detail"] = all(ci[1] < 0 for ci in cis_s.values()), cis_s
        result[name] = r
    ci = E.lab.paired_bootstrap(Y["A"], Y["A*"], Bf, h=4, n_boot=E.N_BOOT_CI)["margin b-a"]
    result["A*"]["margin 4 s, A* − A"] = [float(ci[0]), float(ci[1])]
    result["A*"]["beats A on margin 4 s"] = bool(ci[0] > 0)
    for _n, r in result.items():
        interp = all(bool(r[g]) for g in ("G1", "G2", "G3", "G4", "G5", "G7"))
        ext = (
            (not interp)
            and all(bool(r[g]) for g in ("G4", "G5", "G7"))
            and bool(r.get("beats A on margin 4 s"))
        )
        r["verdict"] = (
            "valid interpretable model" if interp else "valid predictive extension" if ext else "not valid"
        )
        r["transfers"] = bool(r["G8"]) and r["verdict"] != "not valid"
    table = pd.DataFrame(
        {
            n: {g: r[g] for g in ("G1", "G2", "G3a", "G3b", "G4", "G5", "G7", "G8")}
            | {
                "G6 horizon val": r["G6"]["val"],
                "G6 horizon cf": r["G6"]["cf"],
                "verdict": r["verdict"],
                "transfers": r["transfers"],
            }
            for n, r in result.items()
        }
    )
    print(table.to_string())
    (DOCS / "04_gates_post_hoc.json").write_text(json.dumps(result, indent=1, default=E._plain))
    lines = [
        "# The 17 Nov gates applied to A* and A (POST-HOC, 18 Nov)",
        "",
        "**Not a pre-registered test.** `final` (17 Nov) and `test2` (18 Nov) are spent; this applies the",
        "gates' definitions to `val`, `cf_*` and exp07 `val`, as `evaluate_final.py --dry-run` does. A* was",
        "designed after looking at cv and val results. G7 uses the test2 / val ratio from the Phase 3 run.",
        "Script: `scripts/gates_post_hoc.py`; every number: `04_gates_post_hoc.json`.",
        "",
        "```",
        table.to_string(),
        "```",
        "",
    ]
    for n, r in result.items():
        lines += [
            f"## {n}",
            "",
            f"- G1 term counts: {r['G1 detail']}",
            f"- G2 sign failures: {r['G2 detail'] or 'none'}",
        ]
        failed = {k: v for k, v in r["G3a detail"].items() if not v["pass"]}
        lines += [f"- G3a terms failing: {failed or 'none'}", f"- G3b: {r['G3b detail']}"]
        lines += [
            f"- G4: {r['G4 detail']}",
            f"- G5 (paired 95%, model − baseline, health 4 s): {r['G5 detail']}",
        ]
        lines += [f"- G6 horizon: {r['G6']}", f"- G7: {r['G7 detail']}", f"- G8: {r['G8 detail']}"]
        if "margin 4 s, A* − A" in r:
            lines += [f"- margin R² at 4 s, A* − A (paired 95%): {r['margin 4 s, A* − A']}"]
        lines += [""]
    (DOCS / "04_gates_post_hoc.md").write_text("\n".join(lines))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
