"""The pre-registered final evaluation: apply docs/2026-11-17_acceptance-gates.md to models A, W, B, S.

Reads the fitted models and the data; fits nothing on the final sets. Run ONCE on the final data:

    uv run scripts/evaluate_final.py --final        the real evaluation (writes the result files)
    uv run scripts/evaluate_final.py --dry-run      the same code with val / cf_* / exp07 val in
                                                    place of the final sets, to check that it runs

Outputs (--final only): docs/2026-11-17_final-evaluation.md (tables and verdicts) and
data/processed/exp06_intents/final_evaluation.json (every number).

Choices the gates file leaves open, fixed here before running:
- G3 bootstrap refits keep the closures (reach curve and movement switches) fixed and refit the
  SINDy coefficients only, as in `sindyc_02`; B and S are refitted on rollout error from 20,000
  start points of the resampled battles, S's share terms on top of the refitted B.
- G3's "coefficient's 95% range" is the 2.5-97.5% quantile over the refits that keep the term.
- G3(b) compares every equation's term set (health, distance, alive).
- B's offset decay times (tau) are not library terms and have no sign rule in G2.
- G4 is checked on `final` and `final_cf`; G6's horizon is the largest h meeting its rule.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks" / "lib"))

import sindyc_07_lab as lab  # noqa: E402
from weak_sindy import fit_weak, weak_rows  # noqa: E402

from sindy_wm.models.intent_model import (  # noqa: E402
    DT,
    IntentModel,
    add_controls,
    add_rates,
    make_batch,
)
from sindy_wm.models.switch_library import scaled  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402

PROCESSED = DATA_DIR / "processed" / "exp06_intents"
STALKERS = DATA_DIR / "processed" / "exp07_stalkers"
HORIZONS = (1, 2, 3, 4, 6, 8)
N_BOOT_CI, N_REFITS = 1000, 30
ALIVE, HEALTH = ["p_b", "p_r", "p_r_focus"], ["kb", "kr", "kr_wd"]
DIST, TAU = ["d_adv", "d_wd", "d_flee", "d_hold_far", "d_hold_contact"], ["tau_b", "tau_r"]
SHARE = ("sa", "sk", "sf_adv", "sf_hold", "sf_wd", "so_wd", "sd")


# ---------------------------------------------------------------- data
def rows_with_rates(name: str, window: int = 9) -> pd.DataFrame:
    d = add_rates(add_controls(pd.read_parquet(PROCESSED / f"{name}.parquet")), window=window)
    return d[(d["alive_blue"] > 0) & (d["alive_red"] > 0)]


def batch(name: str, experiment: str = "exp06_intents") -> dict:
    """Start points of one set; `<prefix>cf` concatenates the six counterfactual sets (grouped by seed)."""
    folder = DATA_DIR / "processed" / experiment
    if name.endswith("cf"):
        parts = []
        for k, f in enumerate(sorted(folder.glob(f"{name}_*.parquet"))):
            d = pd.read_parquet(f).reset_index()
            d["episode_idx"] += 10_000 * (k + 1)
            parts.append(d.set_index(["episode_idx", "step"]))
        B = make_batch(add_controls(pd.concat(parts)))
        B["group"] = B["ep"] % 10_000
        return B
    B = make_batch(add_controls(pd.read_parquet(folder / f"{name}.parquet")))
    B["group"] = B["ep"]
    return B


# ---------------------------------------------------------------- models
def load_models(rows: pd.DataFrame) -> dict:
    A = pickle.load(open(PROCESSED / "model_derived.pkl", "rb"))
    W = fit_weak(A, weak_rows(rows, A.P, A.names, radius=16))
    M = pickle.load(open(PROCESSED / "sindyc_07_models.pkl", "rb"))
    return {"A": A, "W": W, "B": M["B refit, same terms"], "S": M["S share as a state"]}


def rollout(model, B: dict, stalkers: bool = False) -> np.ndarray:
    """Rollout of an IntentModel or a lab parameter dict; scaled to stalkers without refitting if asked."""
    if isinstance(model, IntentModel):
        m = scaled(model, "marine", "stalker") if stalkers else model
        return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])
    return lab.simulate(lab.scale_params(model) if stalkers else model, B)


def baselines(rows: pd.DataFrame) -> dict:
    Hb, Hr = rows["health_blue"].to_numpy(), rows["health_red"].to_numpy()
    a = -np.dot(Hr, rows["d_health_blue"]) / np.dot(Hr, Hr)
    b = -np.dot(Hb, rows["d_health_red"]) / np.dot(Hb, Hb)
    closures = pickle.load(open(PROCESSED / "closures.pkl", "rb"))
    no_control = IntentModel(closures, ctrl=False).fit(rows, thr_shape=None)
    return {"persistence": "persistence", "Lanchester": (a, b), "no control": no_control}


def baseline_rollout(base, B: dict, stalkers: bool = False) -> np.ndarray:
    steps = B["U"].shape[1]
    if isinstance(base, str):
        return np.repeat(B["X0"][:, None], steps + 1, 1)
    if isinstance(base, tuple):
        a, b = base
        if stalkers:
            from sindy_wm.models.switch_library import UNITS

            f = UNITS["marine"]["tau"] / UNITS["stalker"]["tau"]
            a, b = a * f, b * f
        X, out = B["X0"].astype(float).copy(), [B["X0"].astype(float).copy()]
        for _ in range(steps):
            X = X.copy()
            X[:, :2] = np.maximum(X[:, :2] + DT * np.column_stack([-a * X[:, 1], -b * X[:, 0]]), 0)
            out.append(X)
        return np.stack(out, 1)
    return rollout(base, B, stalkers)


# ---------------------------------------------------------------- G1, G2: terms and signs
def terms(model) -> dict:
    """Kept terms per equation: {equation: {term: coefficient}}."""
    if isinstance(model, IntentModel):
        out = {eq: {n: c for n, c in zip(model.names, model.coef[eq], strict=True) if c} for eq in model.coef}
        for side in ("blue", "red"):
            out[f"alive_{side}"] = {k: v for k, v in model.p[side].items() if v}
        return out
    p = model
    out = {
        "health_blue": {"Rr": p["kb"]},
        "health_red": {k: v for k, v in (("Rb", p["kr"]), ("Rb*wd", p["kr_wd"])) if v},
        "distance": {k: p[k] for k in DIST if p[k]},
        "alive_blue": {"1": p["p_b"]},
        "alive_red": {k: v for k, v in (("1", p["p_r"]), ("focus", p["p_r_focus"])) if v},
    }
    if p.get("share_state"):
        for t, side in (("b", "blue"), ("r", "red")):
            out[f"share_{side}"] = {k: p[f"{k}_{t}"] for k in SHARE if p.get(f"{k}_{t}")}
    return out


def gate1(T: dict) -> tuple[bool, dict]:
    limits = {"health_blue": 3, "health_red": 3, "distance": 6, "alive_blue": 2, "alive_red": 2}
    limits |= {"share_blue": 7, "share_red": 7}
    counts = {eq: len(v) for eq, v in T.items()}
    return all(counts[eq] <= limits[eq] for eq in counts), counts


DIST_NAMES = {"d_adv": "close*adv", "d_wd": "close*wd", "d_flee": "flee*wd", "d_hold_far": "close*hold*far"}
DIST_NAMES["d_hold_contact"] = "close*hold*contact"


def sign_ok(eq: str, term: str, c: float, T: dict) -> bool | None:
    """The G2 rule for one kept term; None if the table has no rule for it."""
    term = DIST_NAMES.get(term, term)
    if eq == "health_blue" and term == "Rr" or eq == "health_red" and term == "Rb":
        return c < 0
    if eq == "health_red" and term == "Rb*wd":
        return 0 < c < abs(T["health_red"].get("Rb", 0))
    if eq == "distance" and term.startswith("close*"):
        return c < 0
    if eq == "distance" and term == "flee*wd":
        return c > 0
    if eq.startswith("alive_"):
        base = T[eq].get("1", 0)
        return 0 < (base if term == "1" else base + c) <= 1
    if eq.startswith("share_"):
        return c <= 0 if term == "sd" else c >= 0
    return None


def gate2(T: dict) -> tuple[bool, list]:
    bad = []
    for eq, ts in T.items():
        for term, c in ts.items():
            ok = sign_ok(eq, term, c, T)
            if not ok:
                bad.append(f"{eq}: {term} = {c:+.3f} ({'no rule' if ok is None else 'wrong sign'})")
    return not bad, bad


# ---------------------------------------------------------------- G3: robustness
def _resample_rows(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Battles drawn with replacement; each draw gets its own episode id."""
    codes, uniq = pd.factorize(rows.index.get_level_values(0))
    positions = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    pick = np.random.default_rng(seed).integers(0, len(uniq), len(uniq))
    idx = np.concatenate([positions[g] for g in pick])
    new_ep = np.repeat(np.arange(len(pick)), [len(positions[g]) for g in pick])
    out = rows.iloc[idx].copy()
    out.index = pd.MultiIndex.from_arrays(
        [new_ep, rows.index.get_level_values(1)[idx]], names=rows.index.names
    )
    return out


def _resample_batch(B: dict, seed: int) -> dict:
    groups = np.unique(B["group"])
    pick = np.random.default_rng(seed).choice(groups, len(groups), replace=True)
    by_group = {g: np.flatnonzero(B["group"] == g) for g in groups}
    idx = np.concatenate([by_group[g] for g in pick])
    Bb = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(B["group"]) else v) for k, v in B.items()}
    return lab.subsample(Bb, 20000, seed=seed)


_CTX = {}


def _refit_one(args):
    name, seed = args
    c = _CTX
    if name == "A":
        A = c["models"]["A"]
        return terms(
            IntentModel(A.P, exclude=A.exclude).fit(
                _resample_rows(c["rows"], seed), reach_true=True, thr_health=0.15, thr_shape=None
            )
        )
    if name == "W":
        A = c["models"]["A"]
        return terms(fit_weak(A, weak_rows(_resample_rows(c["rows"], seed), A.P, A.names, radius=16)))
    Bb = _resample_batch(c["Btrain"], seed)
    pB, _ = lab.refit(ALIVE + HEALTH + DIST + TAU, c["pA"], Bb)
    if name == "B":
        return terms(pB)
    pS0 = c["models"]["S"]
    keep = [k for k in pS0 if k.startswith(tuple(f"{s}_" for s in SHARE)) and pS0[k]]
    start = dict(pS0, **{k: pB[k] for k in ALIVE + HEALTH + DIST + TAU})
    pS, _ = lab.refit(keep, start, Bb, reach_w=1.0)
    return terms(pS)


def _init_worker(name: str) -> None:
    """Load, in a fresh worker, only what `_refit_one` needs for this model."""
    M = pickle.load(open(PROCESSED / "sindyc_07_models.pkl", "rb"))
    models = {"A": pickle.load(open(PROCESSED / "model_derived.pkl", "rb")), "S": M["S share as a state"]}
    _CTX.update(models=models)
    if name in ("A", "W"):
        _CTX.update(rows=rows_with_rates("train"))
    else:
        _CTX.update(pA=lab.base_params(), Btrain=lab.load_batch("train"))


def gate3a(name: str, T: dict, workers: int, n_refits: int) -> tuple[bool, dict]:
    # "spawn", not "fork": Apple's Accelerate (numpy's BLAS here) deadlocks in forked children once
    # the parent has used its threads, which hung the B and S refits.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(name,)) as pool:
        refits = pool.map(_refit_one, [(name, 1000 + i) for i in range(n_refits)])
    detail, ok = {}, True
    for eq, ts in T.items():
        for term in ts:
            vals = [r.get(eq, {}).get(term) for r in refits]
            kept = [v for v in vals if v]
            share = len(kept) / len(refits)
            lo, hi = np.quantile(kept, [0.025, 0.975]) if kept else (np.nan, np.nan)
            one_side = bool(kept) and (lo > 0 or hi < 0)
            detail[f"{eq}: {term}"] = {
                "kept": share,
                "range": [float(lo), float(hi)],
                "pass": share >= 0.9 and one_side,
            }
            ok &= detail[f"{eq}: {term}"]["pass"]
    return ok, detail


def gate3b(models: dict) -> tuple[bool, dict]:
    A = models["A"]
    fits = {}
    for w in (5, 9, 17):
        fits[f"SG {w}"] = terms(
            IntentModel(A.P, exclude=A.exclude).fit(
                rows_with_rates("train", window=w), reach_true=True, thr_health=0.15, thr_shape=None
            )
        )
    raw = _CTX["rows"]
    for radius in (8, 16):
        fits[f"weak {radius * DT:g} s"] = terms(fit_weak(A, weak_rows(raw, A.P, A.names, radius=radius)))
    sets = {k: {eq: sorted(v) for eq, v in T.items()} for k, T in fits.items()}
    same_terms = all(s == sets["SG 9"] for s in sets.values())
    within = True
    spread = {}
    for eq in ("health_blue", "health_red"):
        for term in fits["SG 9"][eq]:
            vals = np.array([T[eq].get(term, 0.0) for T in fits.values()])
            med = np.median(vals)
            dev = float(np.max(np.abs(vals - med) / abs(med)))
            spread[f"{eq}: {term}"] = {"values": vals.round(4).tolist(), "max deviation from median": dev}
            within &= dev <= 0.10
    differ = {k: s for k, s in sets.items() if s != sets["SG 9"]}
    return same_terms and within, {
        "same term set": same_terms,
        "term sets that differ": differ,
        "health coefficients": spread,
    }


# ---------------------------------------------------------------- G4-G8 on held-out data
def gate4(Y: np.ndarray, B: dict) -> tuple[bool, dict]:
    finite = bool(np.isfinite(Y).all())
    nonneg = bool((Y[:, :, [0, 1, 2, 3, 4]] >= -1e-9).all())
    alive_cap = bool((Y[:, :, 2:4] <= B["X0"][:, None, 2:4] + 1e-9).all())
    return finite and nonneg and alive_cap, {
        "finite": finite,
        "non-negative": nonneg,
        "alive never above start": alive_cap,
    }


def ci_health_diff(Y_base, Y_model, B, h=4):
    ci = lab.paired_bootstrap(Y_base, Y_model, B, h=h, n_boot=N_BOOT_CI)["health b-a"]
    return [float(ci[0]), float(ci[1])]


def horizon(Y, B) -> tuple[float | None, dict]:
    rows, best = {}, None
    m = lab.metrics(Y, B, horizons=HORIZONS)
    for h in HORIZONS:
        lo, hi = lab.paired_bootstrap(Y, Y, B, h=h, n_boot=N_BOOT_CI)["margin a"]
        ok = m[f"margin R2 {h}s"] >= 0.3 and lo > 0
        rows[f"{h} s"] = {
            "margin R2": float(m[f"margin R2 {h}s"]),
            "interval": [float(lo), float(hi)],
            "meets rule": bool(ok),
        }
        if ok:
            best = h
    return best, rows


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refits", type=int, default=N_REFITS, help="dry run only; --final always uses 30")
    args = ap.parse_args()
    n_refits = N_REFITS if args.final else args.refits
    t0 = time.time()
    held = ("final", "final_cf", "final") if args.final else ("val", "cf", "val")
    print(f"evaluating on {held[0]}, {held[1]} and exp07 {held[2]}", flush=True)

    rows = rows_with_rates("train")
    models = load_models(rows)
    base = baselines(rows)
    _CTX.update(rows=rows, models=models, pA=lab.base_params(), Btrain=lab.load_batch("train"))
    Bf, Bcf, Bval = batch(held[0]), batch(held[1]), batch("val")
    Bs = batch(held[2], "exp07_stalkers")
    Y = {k: rollout(m, Bf) for k, m in models.items()}
    Ycf = {k: rollout(m, Bcf) for k, m in models.items()}
    Yval = {k: rollout(m, Bval) for k, m in models.items()}
    Ys = {k: rollout(m, Bs, stalkers=True) for k, m in models.items()}
    Yb = {k: baseline_rollout(b, Bf) for k, b in base.items()}
    Ybs = {
        k: baseline_rollout(b, Bs, stalkers=True)
        for k, b in base.items()
        if k in ("persistence", "no control")
    }

    g3b_ok, g3b = gate3b(models)
    result = {}
    for name, m in models.items():
        print(f"-- {name} ({time.time() - t0:.0f} s)", flush=True)
        T = terms(m)
        r = {"terms": {eq: {k: float(v) for k, v in ts.items()} for eq, ts in T.items()}}
        r["G1"], r["G1 detail"] = gate1(T)
        r["G2"], r["G2 detail"] = gate2(T)
        r["G3a"], r["G3a detail"] = gate3a(name, T, args.workers, n_refits)
        r["G3b"], r["G3b detail"] = (g3b_ok, g3b) if name in ("A", "W") else (None, "not applicable")
        r["G3"] = r["G3a"] and (r["G3b"] is not False)
        ok_f, d_f = gate4(Y[name], Bf)
        ok_c, d_c = gate4(Ycf[name], Bcf)
        r["G4"], r["G4 detail"] = ok_f and ok_c, {held[0]: d_f, held[1]: d_c}
        cis = {b: ci_health_diff(Yb[b], Y[name], Bf) for b in Yb}
        r["G5"], r["G5 detail"] = all(ci[1] < 0 for ci in cis.values()), {"model − baseline, health 4 s": cis}
        hf, rf = horizon(Y[name], Bf)
        hc, rc = horizon(Ycf[name], Bcf)
        r["G6"] = {held[0]: hf, held[1]: hc}
        r["G6 detail"] = {held[0]: rf, held[1]: rc}
        e_f = lab.metrics(Y[name], Bf)["health 4s"]
        e_v = lab.metrics(Yval[name], Bval)["health 4s"]
        r["G7"], r["G7 detail"] = (
            bool(e_f <= 1.1 * e_v),
            {"health 4 s held-out": e_f, "health 4 s val": e_v, "ratio": e_f / e_v},
        )
        cis_s = {b: ci_health_diff(Ybs[b], Ys[name], Bs) for b in Ybs}
        r["G8"], r["G8 detail"] = (
            all(ci[1] < 0 for ci in cis_s.values()),
            {"model − baseline, stalkers health 4 s": cis_s},
        )
        r["metrics"] = {
            held[0]: lab.metrics(Y[name], Bf, horizons=HORIZONS).to_dict(),
            held[1]: lab.metrics(Ycf[name], Bcf, horizons=HORIZONS).to_dict(),
            f"stalkers {held[2]}": lab.metrics(Ys[name], Bs, horizons=(1, 2, 4, 8)).to_dict(),
        }
        result[name] = r
    for name, r in result.items():
        if name == "A":
            r["beats A on margin 4 s"] = None
            continue
        ci = lab.paired_bootstrap(Y["A"], Y[name], Bf, h=4, n_boot=N_BOOT_CI)["margin b-a"]
        r["beats A on margin 4 s"] = bool(ci[0] > 0)
        r["margin 4 s, model − A"] = [float(ci[0]), float(ci[1])]
    for r in result.values():
        interp = all(r[g] for g in ("G1", "G2", "G3", "G4", "G5", "G7"))
        ext = (not interp) and all(r[g] for g in ("G4", "G5", "G7")) and bool(r.get("beats A on margin 4 s"))
        r["verdict"] = (
            "valid interpretable model" if interp else "valid predictive extension" if ext else "not valid"
        )
        r["transfers"] = bool(r["G8"]) and r["verdict"] != "not valid"
    print(f"done in {time.time() - t0:.0f} s", flush=True)

    table = pd.DataFrame(
        {
            n: {g: r[g] for g in ("G1", "G2", "G3", "G4", "G5", "G7", "G8")}
            | {
                f"G6 horizon {held[0]}": r["G6"][held[0]],
                f"G6 horizon {held[1]}": r["G6"][held[1]],
                "verdict": r["verdict"],
                "transfers": r["transfers"],
            }
            for n, r in result.items()
        }
    )
    print(table.to_string())
    if not args.final:
        return
    json.dump(result, open(PROCESSED / "final_evaluation.json", "w"), indent=1, default=_plain)
    write_report(result, table)


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def write_report(result: dict, table: pd.DataFrame) -> None:
    lines = [
        "# Final evaluation (pre-registered, run once)",
        "",
        "Gates: `docs/2026-11-17_acceptance-gates.md` (FINAL). Script: `scripts/evaluate_final.py --final`.",
        "Data: exp06 `final`, `final_cf_*`; exp07 `final`. Every number: "
        "`data/processed/exp06_intents/final_evaluation.json`.",
        "",
        "## Gates and verdicts",
        "",
        "```",
        table.to_string(),
        "```",
        "",
    ]
    for name, r in result.items():
        lines += [f"## {name}", ""]
        lines += [f"- G1 term counts: {r['G1 detail']}"]
        lines += [f"- G2 sign failures: {r['G2 detail'] or 'none'}"]
        failed3 = {k: v for k, v in r["G3a detail"].items() if not v["pass"]}
        lines += [f"- G3a terms failing the bootstrap rule: {failed3 or 'none'}"]
        if r["G3b"] is not None:
            lines += [f"- G3b: {r['G3b detail']}"]
        lines += [f"- G4: {r['G4 detail']}", f"- G5 (paired 95% intervals): {r['G5 detail']}"]
        lines += [f"- G6: {r['G6 detail']}", f"- G7: {r['G7 detail']}", f"- G8: {r['G8 detail']}"]
        if "margin 4 s, model − A" in r:
            lines += [f"- margin R² at 4 s, model − A (paired 95%): {r['margin 4 s, model − A']}"]
        lines += [""]
    (REPO / "docs" / "2026-11-17_final-evaluation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
