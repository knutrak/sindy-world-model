"""Reach as a state from a generic library (approved plan: docs/generalisation/07_share_weak_plan.md).

uv run --extra explore experiments/generalisation/reach_state.py dry      200 battles per scenario, timed
uv run --extra explore experiments/generalisation/reach_state.py fit      all fits and bootstrap (R1)
uv run --extra explore experiments/generalisation/reach_state.py val      val rollouts (R2, R3, R4)
uv run --extra explore experiments/generalisation/reach_state.py report   -> docs/generalisation/08
"""

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import DT, add_controls  # noqa: E402
from sindy_wm.models.kite import CTRL_K, add_kite  # noqa: E402
from sindy_wm.models.pipeline import SIDES  # noqa: E402
from sindy_wm.models.reach_state import (  # noqa: E402
    CONTROLS,
    ReachStatePipeline,
    closure_share_fn,
    gram,
    group_index,
    rows_for,
    two_level_gram,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
OUT = DATA_DIR / "processed" / "generalisation" / "reach_state"
DOC = PROJECT_ROOT / "docs" / "generalisation" / "08_reach_state.md"
TYPED = DATA_DIR / "processed"
SCEN = {
    "marines": {"train": [("exp06_intents_typed", "train")], "val": [("exp06_intents_typed", "val")]},
    "mixed": {
        "train": [("exp08_mixed_typed", "train"), ("exp11_kite_typed", "train")],
        "val": [("exp08_mixed_typed", "val"), ("exp11_kite_typed", "val")],
    },
}
DEGREES = (2, 3)
LIBS = ("pure", "clo")
METHODS = (("weak", 8), ("weak", 16), ("step", 0))  # (method, radius in steps)
THRESHOLDS = (0.025, 0.05, 0.1)
MAIN_THR = 0.05
N_BOOT = 30
HORIZONS = (1.0, 2.0, 4.0)


def frozen(scen: str):
    return pickle.load(open(FROZEN, "rb"))[scen]


def load(scen: str, split: str, max_battles: int | None = None) -> pd.DataFrame:
    parts = []
    for k, (folder, name) in enumerate(SCEN[scen][split]):
        d = pd.read_parquet(TYPED / folder / f"{name}.parquet").reset_index()
        if max_battles:
            keep = d["episode_idx"].drop_duplicates().iloc[:max_battles]
            d = d[d["episode_idx"].isin(keep)]
        d["episode_idx"] = d["episode_idx"] + 100_000 * k
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_kite(add_controls(pd.concat(parts)))


def controls_for(scen: str):
    return CONTROLS if scen == "mixed" else tuple(c for c in CONTROLS if c != "kite")


def key(scen, side, a, deg, lib, method, radius) -> str:
    return f"{scen}|{side}|{a}|d{deg}|{lib}|{method}{radius}"


def fit_one(states, m, scen, side, a, deg, lib, method, radius, boot: bool, rng):
    ctrls = controls_for(scen)
    clo_fn = closure_share_fn(m) if lib == "clo" else None
    t0 = time.time()
    Th, b, ids, names = rows_for(
        states, m.L, side, a, deg, ctrls, method, radius=radius or 8, stride=8, clo_fn=clo_fn
    )
    t_rows = time.time() - t0
    groups, base = group_index(names, ctrls)
    G = gram(Th, b)
    fits = {thr: two_level_gram(*G, groups, base, thr) for thr in THRESHOLDS}
    t_fit = time.time() - t0 - t_rows
    out = {
        "names": names,
        "controls": ctrls,
        "degree": deg,
        "fits": fits,
        "n": len(b),
        "t_rows": t_rows,
        "t_fit": t_fit,
        "r2": {},
    }
    for thr, c in fits.items():
        p = Th.astype(np.float64) @ c
        out["r2"][thr] = float(1 - ((b - p) ** 2).sum() / ((b - b.mean()) ** 2).sum())
    if boot:
        u, inv = np.unique(ids, return_inverse=True)
        kept = []
        t1 = time.time()
        for _ in range(N_BOOT):
            wb = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
            c = two_level_gram(*gram(Th, b, wb[inv]), groups, base, MAIN_THR)
            kept.append(np.unique(groups[c != 0]))
        out["boot"] = kept
        out["t_boot"] = time.time() - t1
    return out


def kept_groups(res) -> set:
    groups, _ = group_index(res["names"], res["controls"])
    return set(np.unique(groups[res["fits"][MAIN_THR] != 0]).tolist())


def equation_text(res, thr=MAIN_THR) -> str:
    c = res["fits"][thr]
    terms = []
    for j, v in enumerate(c):
        if v:
            g, ctl = res["names"][j // len(res["controls"])], res["controls"][j % len(res["controls"])]
            terms.append(f"{v:+.4g}·{g}" + ("" if ctl == "1" else f"·{ctl}"))
    return " ".join(terms) if terms else "0"


# ------------------------------------------------------------------ dry run
def dry() -> None:
    rng = np.random.default_rng(0)
    for scen in SCEN:
        m = frozen(scen)
        st = load(scen, "train", max_battles=200)
        a = m.L.types[0]
        for deg in DEGREES:
            r = fit_one(st, m, scen, "blue", a, deg, "clo", "weak", 8, boot=False, rng=rng)
            t1 = time.time()
            groups, base = group_index(r["names"], r["controls"])
            print(
                f"{scen} d{deg}: rows {r['n']}, columns {len(groups)}, rows {r['t_rows']:.1f} s, "
                f"3 fits {r['t_fit']:.1f} s, one Gram {time.time() - t1:.2f} s; eq: {equation_text(r)[:200]}"
            )


# ------------------------------------------------------------------ fit
def fit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {}
    for scen in SCEN:
        m = frozen(scen)
        st = load(scen, "train")
        for side in SIDES:
            for a in m.L.types:
                for deg in DEGREES:
                    for lib in LIBS:
                        for method, radius in METHODS:
                            k = key(scen, side, a, deg, lib, method, radius)
                            r = fit_one(st, m, scen, side, a, deg, lib, method, radius, method == "weak", rng)
                            results[k] = r
                            print(
                                f"{k}: n {r['n']}, {r['t_rows'] + r['t_fit'] + r.get('t_boot', 0):.0f} s, "
                                f"{len(kept_groups(r))} groups",
                                flush=True,
                            )
        pickle.dump(results, open(OUT / "fits.pkl", "wb"))
    pickle.dump(results, open(OUT / "fits.pkl", "wb"))


def r1_table(results) -> pd.DataFrame:
    rows = []
    for k, r in results.items():
        scen, side, a, deg, lib, meth = k.split("|")
        if meth != "weak8":
            continue
        r2 = results[k.replace("weak8", "weak16")]
        g1, g2 = kept_groups(r), kept_groups(r2)
        jac = len(g1 & g2) / max(len(g1 | g2), 1)
        boot_ok = True
        for rr, gg in ((r, g1), (r2, g2)):
            for g in gg:
                share = np.mean([g in set(b.tolist()) for b in rr["boot"]])
                boot_ok &= share >= 0.9
        rows.append(
            {
                "scenario": scen,
                "equation": f"{side} {a}",
                "degree": deg,
                "library": lib,
                "groups 1 s": len(g1),
                "groups 2 s": len(g2),
                "Jaccard": jac,
                "bootstrap >= 90%": boot_ok,
                "R1": bool(jac >= 0.75 and boot_ok and (g1 or g2)),
                "R2 groups <= 8": bool(len(g1) <= 8 and len(g2) <= 8),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ val rollouts
def true_reach(states, L, B, steps):
    cols = {s: [f"reach_{s}_{a}" for a in L.types] for s in SIDES}
    arr = {
        ep: np.column_stack([g[cols[s]].sum(1).to_numpy(float) for s in SIDES])
        for ep, g in states.groupby(level=0, sort=False)
    }
    out = np.full((len(B["ep"]), steps + 1, 2), np.nan)
    for i, (ep, k0) in enumerate(zip(B["ep"], B["k0"], strict=True)):
        x = arr[ep][k0 : k0 + steps + 1]
        out[i, : len(x)] = x
    return out


def boot(x, grp, rng, n=2000):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((w * sums).sum() / (w * cnt).sum())
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def variant(m, results, scen, deg, lib, meth, thr=MAIN_THR):
    eqs = {}
    for side in SIDES:
        for a in m.L.types:
            r = results[key(scen, side, a, deg, lib, *meth)]
            eqs[(side, a)] = (r["names"], r["controls"], deg, r["fits"][thr])
    return ReachStatePipeline.from_frozen(m, eqs, with_clo=lib == "clo")


def val() -> None:
    results = pickle.load(open(OUT / "fits.pkl", "rb"))
    rng = np.random.default_rng(0)
    rows, kite_rows = [], []
    for scen in SCEN:
        m = frozen(scen)
        horizon = round(max(HORIZONS) / DT)
        st = load(scen, "val")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = m.L.batch(st, horizon=horizon, ctrl=CTRL_K)
        TR = true_reach(st, m.L, B, horizon)
        TY = B["team"]["Y"]
        ep = B["ep"]
        models = {"frozen": ReachStatePipeline.from_frozen(m, None)}
        for deg in DEGREES:
            for lib in LIBS:
                for meth in METHODS:
                    models[f"d{deg} {lib} {meth[0]}{meth[1]}"] = variant(m, results, scen, deg, lib, meth)
        out = {}
        for name, mod in models.items():
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                Y = mod.L.team_view(mod.rollout(B))
            out[name] = (Y, mod.R_log, mod.blown.copy())
        kite_win = (B["U"][:, : round(2 / DT), 4] == 1).all(1)
        fY, fR, _ = out["frozen"]
        for name, (Y, R, blown) in out.items():
            for h in HORIZONS:
                n = round(h / DT)
                ok = ~np.isnan(TY[:, n, 0]) & ~np.isnan(TR[:, n - 1, 0])
                real = TY[ok, n, 0] - TY[ok, n, 1] - (TY[ok, 0, 0] - TY[ok, 0, 1])

                def errs(Y, R, ok=ok, n=n, real=real, TR=TR, TY=TY):
                    pred = Y[ok, n, 0] - Y[ok, n, 1] - (Y[ok, 0, 0] - Y[ok, 0, 1])
                    return (
                        10 * np.abs(R[ok, n - 1] - TR[ok, n - 1]).sum(1),
                        np.abs(Y[ok, n, :2] - TY[ok, n, :2]).sum(1),
                        np.abs(pred - real),
                        pred,
                    )

                re, he, me, pred = errs(Y, R)
                rf, hf, mf, _ = errs(fY, fR)
                r2 = 1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum()
                row = {
                    "scenario": scen,
                    "model": name,
                    "h": h,
                    "reach error": re.mean(),
                    "health error": he.mean(),
                    "margin error": me.mean(),
                    "margin R²": r2,
                    "blow-ups": float(blown.mean()),
                }
                if name != "frozen":
                    for lab, a_, b_ in (("reach", re, rf), ("margin", me, mf), ("health", he, hf)):
                        mean, lo, hi = boot(a_ - b_, ep[ok], rng)
                        row[f"{lab} − frozen"] = mean
                        row[f"{lab} 95%"] = f"[{lo:+.4f}, {hi:+.4f}]"
                        row[f"{lab} below 0"] = bool(hi < 0)
                    row["health ratio"] = he.mean() / hf.mean()
                rows.append(row)
            if scen == "mixed" and kite_win.any():
                n = round(2 / DT)
                ok = kite_win & ~np.isnan(TR[:, n - 1, 0])
                kite_rows.append(
                    {
                        "model": name,
                        "kite starts": int(ok.sum()),
                        "Blue reach bias at 2 s (units)": float(
                            10 * (R[ok, n - 1, 0] - TR[ok, n - 1, 0]).mean()
                        ),
                    }
                )
        print(f"{scen} done", flush=True)
    t = pd.DataFrame(rows)
    json.dump({"val": rows, "kite": kite_rows}, open(OUT / "val.json", "w"), indent=1, default=float)
    print(t.round(4).to_string(index=False))
    print(pd.DataFrame(kite_rows).round(4).to_string(index=False))


def r3_table(val_rows) -> pd.DataFrame:
    t = pd.DataFrame(val_rows)
    rows = []
    for deg in DEGREES:
        for lib in LIBS:
            ok_all = True
            detail = {}
            for scen in SCEN:
                for rad in (8, 16):
                    name = f"d{deg} {lib} weak{rad}"
                    s = t[(t.scenario == scen) & (t.model == name)].set_index("h")
                    passed = all(s.loc[h, "reach below 0"] and s.loc[h, "margin below 0"] for h in (2.0, 4.0))
                    passed &= s.loc[1.0, "health ratio"] <= 1.05
                    detail[f"{scen} r{rad // 8} s"] = bool(passed)
                    ok_all &= passed
            rows.append({"degree": deg, "library": lib, **detail, "R3": bool(ok_all)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def report() -> None:
    results = pickle.load(open(OUT / "fits.pkl", "rb"))
    md = lambda t: "```\n" + t.to_string(index=False) + "\n```"  # noqa: E731
    lines = [
        "# Reach as a state from a generic library: results",
        "",
        "Plan: `07_share_weak_plan.md` (approved 22 Nov, amendments 1-8 before any fit). Script:",
        "`experiments/generalisation/reach_state.py`. Fitted on train; evaluated on val.",
        "Threshold 0.05 unless stated.",
        "",
        "## Equations (M-weak, radius 1 s)",
        "",
    ]
    for k, r in results.items():
        if k.endswith("|weak8"):
            lines.append(f"- `{k}` ({len(kept_groups(r))} groups): {equation_text(r)}")
    lines += ["", "## R1, R2 (groups): stability", "", md(r1_table(results).round(3)), ""]
    lines += ["## One-step reference (M-step): groups kept", ""]
    for k, r in results.items():
        if k.endswith("|step0"):
            lines.append(f"- `{k}` ({len(kept_groups(r))} groups): {equation_text(r)[:400]}")
    lines += ["", "## Fit quality (R² of the regression rows, by threshold)", ""]
    fq = [
        {
            "fit": k,
            **{f"thr {t}": r["r2"][t] for t in THRESHOLDS},
            **{
                f"groups {t}": int(
                    len(set(np.unique(group_index(r["names"], r["controls"])[0][r["fits"][t] != 0])))
                )
                for t in THRESHOLDS
            },
        }
        for k, r in results.items()
    ]
    lines += [md(pd.DataFrame(fq).round(3)), ""]
    if (OUT / "val.json").exists():
        v = json.load(open(OUT / "val.json"))
        t = pd.DataFrame(v["val"])
        keep = [
            "scenario",
            "model",
            "h",
            "reach error",
            "health error",
            "margin error",
            "margin R²",
            "blow-ups",
            "reach 95%",
            "margin 95%",
            "health ratio",
        ]
        lines += [
            "## Val rollouts (R2 blow-ups, R3)",
            "",
            md(t[[c for c in keep if c in t]].round(4)),
            "",
            "## R3",
            "",
            md(r3_table(v["val"])),
            "",
            "## R4: kite windows (exp11 val)",
            "",
            md(pd.DataFrame(v["kite"]).round(4)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["dry", "fit", "val", "report"])
    {"dry": dry, "fit": fit, "val": val, "report": report}[p.parse_args().what]()


if __name__ == "__main__":
    main()
