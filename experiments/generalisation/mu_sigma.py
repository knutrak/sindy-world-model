"""μ, σ reach states (approved plan: docs/generalisation/10_mu_sigma_plan.md; amendment 1: L-p and L-nop).

uv run --extra explore experiments/generalisation/mu_sigma.py fit      c, k; all fits; bootstrap (M1)
uv run --extra explore experiments/generalisation/mu_sigma.py val      val rollouts (M2, M3)
uv run --extra explore experiments/generalisation/mu_sigma.py cf       intent ranking (M4)
uv run --extra explore experiments/generalisation/mu_sigma.py report   -> docs/generalisation/11
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_state import boot, true_reach  # noqa: E402

from sindy_wm.models.intent_model import DT, add_controls  # noqa: E402
from sindy_wm.models.kite import CTRL_K, add_kite  # noqa: E402
from sindy_wm.models.mu_sigma import (  # noqa: E402
    MuSigmaPipeline,
    attach_margins,
    rows_for,
    share,
    train_bounds,
)
from sindy_wm.models.pipeline import SIDES  # noqa: E402
from sindy_wm.models.reach_state import (  # noqa: E402
    CONTROLS,
    ReachStatePipeline,
    gram,
    group_index,
    two_level_gram,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
OUT = DATA_DIR / "processed" / "generalisation" / "mu_sigma"
DOC = PROJECT_ROOT / "docs" / "generalisation" / "11_mu_sigma.md"
P = DATA_DIR / "processed"
SCEN = {
    "marines": {
        "train": [("exp06_intents", "train")],
        "val": [("exp06_intents", "val")],
        "cf": [
            ("exp06_intents", f"cf_{b}")
            for b in ("advance_nearest", "advance_focus", "hold_nearest", "hold_focus", "withdraw_nearest")
        ],
    },
    "mixed": {
        "train": [("exp08_mixed", "train"), ("exp11_kite", "train")],
        "val": [("exp08_mixed", "val"), ("exp11_kite", "val")],
        "cf": [
            ("exp08_mixed", f"cf_{b}")
            for b in ("advance_nearest", "advance_focus", "hold_nearest", "hold_focus", "withdraw_nearest")
        ]
        + [("exp11_kite", "cf_kite_nearest"), ("exp11_kite", "cf_kite_focus")],
    },
}
LIBS = {"p": True, "nop": False}
METHODS = (("weak", 8), ("weak", 16), ("step", 0))
THRESHOLDS = (0.025, 0.05, 0.1)
MAIN_THR = 0.05
N_BOOT = 30
HORIZONS = (1.0, 2.0, 4.0)


def frozen(scen):
    return pickle.load(open(FROZEN, "rb"))[scen]


def load_set(exp, name):
    t = pd.read_parquet(P / f"{exp}_typed" / f"{name}.parquet")
    return t.join(pd.read_parquet(P / f"{exp}_margins" / f"{name}.parquet"), how="inner")


def load(scen, split, branch=False):
    parts = []
    for k, (exp, name) in enumerate(SCEN[scen][split]):
        d = load_set(exp, name).reset_index()
        d["episode_idx"] = d["episode_idx"] + 100_000 * k
        if branch:
            d["branch"] = k
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_kite(add_controls(pd.concat(parts)))


def controls_for(scen):
    return CONTROLS if scen == "mixed" else tuple(c for c in CONTROLS if c != "kite")


def fit_ck(st, types):
    s = st.iloc[::4]
    tt = list(types)

    def resid(p):
        return np.concatenate(
            [
                s[f"alive_{sd}_{t}"].to_numpy()
                * share(s[f"mu_{sd}_{t}"].to_numpy(), s[f"sd_{sd}_{t}"].to_numpy(), *p)
                - s[f"reach_{sd}_{t}"].to_numpy()
                for sd in SIDES
                for t in tt
            ]
        )

    return tuple(float(x) for x in least_squares(resid, x0=(0.0, 1.0)).x)


def key(scen, side, a, target, lib, method, radius):
    return f"{scen}|{side}|{a}|{target}|{lib}|{method}{radius}"


def kept(r, thr=MAIN_THR) -> set:
    g, _ = group_index(r["names"], r["controls"])
    return set(np.unique(g[r["fits"][thr] != 0]).tolist())


def equation_text(r, thr=MAIN_THR) -> str:
    c, nc = r["fits"][thr], len(r["controls"])
    t = [
        f"{v:+.3g}·{r['names'][j // nc]}"
        + ("" if r["controls"][j % nc] == "1" else f"·{r['controls'][j % nc]}")
        for j, v in enumerate(c)
        if v
    ]
    return " ".join(t) if t else "0"


# ------------------------------------------------------------------ fit
def fit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results, ck = {}, {}
    for scen in SCEN:
        m = frozen(scen)
        tt = m.L.types
        st = load(scen, "train")
        ck[scen] = fit_ck(st, tt)
        print(scen, "c, k =", ck[scen], flush=True)
        ctrls = controls_for(scen)
        for side in SIDES:
            for a in tt:
                for target in ("mu", "sd"):
                    for lib, with_p in LIBS.items():
                        for method, radius in METHODS:
                            t0 = time.time()
                            Th, b, ids, names = rows_for(
                                st, tt, side, a, target, with_p, ctrls, method, *ck[scen], radius=radius or 8
                            )
                            groups, base = group_index(names, ctrls)
                            G = gram(Th, b)
                            fits = {thr: two_level_gram(*G, groups, base, thr) for thr in THRESHOLDS}
                            r = {"names": names, "controls": ctrls, "fits": fits, "n": len(b)}
                            r["r2"] = {
                                thr: float(
                                    1 - ((b - Th.astype(float) @ c) ** 2).sum() / ((b - b.mean()) ** 2).sum()
                                )
                                for thr, c in fits.items()
                            }
                            if method == "weak":
                                u, inv = np.unique(ids, return_inverse=True)
                                r["boot"] = []
                                for _ in range(N_BOOT):
                                    wb = np.bincount(
                                        rng.integers(0, len(u), len(u)), minlength=len(u)
                                    ).astype(float)
                                    c = two_level_gram(*gram(Th, b, wb[inv]), groups, base, MAIN_THR)
                                    r["boot"].append(np.unique(groups[c != 0]))
                            k = key(scen, side, a, target, lib, method, radius)
                            results[k] = r
                            print(
                                f"{k}: n {len(b)}, {len(kept(r))} groups, R² {r['r2'][MAIN_THR]:.3f}, "
                                f"{time.time() - t0:.0f} s",
                                flush=True,
                            )
    pickle.dump({"fits": results, "ck": ck}, open(OUT / "fits.pkl", "wb"))


def m1_table(res) -> pd.DataFrame:
    rows = []
    for k, r in res.items():
        scen, side, a, target, lib, meth = k.split("|")
        if meth != "weak8":
            continue
        r2 = res[k.replace("weak8", "weak16")]
        g1, g2 = kept(r), kept(r2)
        jac = len(g1 & g2) / max(len(g1 | g2), 1)
        bok = all(
            np.mean([g in set(x.tolist()) for x in rr["boot"]]) >= 0.9
            for rr, gg in ((r, g1), (r2, g2))
            for g in gg
        )
        rows.append(
            {
                "scenario": scen,
                "equation": f"{target}′ {side} {a}",
                "library": lib,
                "groups 1 s": len(g1),
                "groups 2 s": len(g2),
                "Jaccard": jac,
                "bootstrap ≥ 90%": bool(bok),
                "M1": bool(len(g1) <= 8 and len(g2) <= 8 and jac >= 0.75 and bok),
            }
        )
    return pd.DataFrame(rows)


def variant(m, res, ck, scen, lib, meth, thr=MAIN_THR):
    eqs, names, ctrls = {}, None, None
    for side in SIDES:
        for a in m.L.types:
            eqs[(side, a)] = {}
            for target in ("mu", "sd"):
                r = res[key(scen, side, a, target, lib, *meth)]
                eqs[(side, a)][target], names, ctrls = r["fits"][thr], r["names"], r["controls"]
    return MuSigmaPipeline.build(m, eqs, names, ctrls, LIBS[lib], *ck[scen])


def models_for(m, res, ck, scen):
    out = {"frozen": ReachStatePipeline.from_frozen(m, None)}
    for lib in LIBS:
        for meth in METHODS:
            out[f"{lib} {meth[0]}{meth[1]}"] = variant(m, res, ck, scen, lib, meth)
    return out


# ------------------------------------------------------------------ val (M2, M3)
def val() -> None:
    F = pickle.load(open(OUT / "fits.pkl", "rb"))
    res, ck = F["fits"], F["ck"]
    rng = np.random.default_rng(0)
    rows, kite_rows = [], []
    for scen in SCEN:
        m = frozen(scen)
        hz = round(max(HORIZONS) / DT)
        st = load(scen, "val")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach_margins(m.L.batch(st, horizon=hz, ctrl=CTRL_K), st, m.L)
        TR, TY, ep = true_reach(st, m.L, B, hz), B["team"]["Y"], B["ep"]
        out = {}
        for name, mod in models_for(m, res, ck, scen).items():
            with np.errstate(all="ignore"):
                Y = mod.L.team_view(mod.rollout(B))
            out[name] = (Y, mod.R_log, mod.blown.copy())
        fY, fR, _ = out["frozen"]
        kite_win = (B["U"][:, : round(2 / DT), 4] == 1).all(1)
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
                row = {
                    "scenario": scen,
                    "model": name,
                    "h": h,
                    "reach error": re.mean(),
                    "health error": he.mean(),
                    "margin error": me.mean(),
                    "margin R²": 1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum(),
                    "blow-ups": float(blown.mean()),
                }
                if name != "frozen":
                    for lab, a_, b_ in (("reach", re, rf), ("margin", me, mf), ("health", he, hf)):
                        mean, lo, hi = boot(a_ - b_, ep[ok], rng)
                        row[f"{lab} − frozen"], row[f"{lab} 95%"], row[f"{lab} below 0"] = (
                            mean,
                            f"[{lo:+.4f}, {hi:+.4f}]",
                            bool(hi < 0),
                        )
                    row["health ratio"] = he.mean() / hf.mean()
                rows.append(row)
            if scen == "mixed":
                n = round(2 / DT)
                ok = kite_win & ~np.isnan(TR[:, n - 1, 0])
                kite_rows.append(
                    {
                        "model": name,
                        "kite starts": int(ok.sum()),
                        "Blue reach bias at 2 s (units)": float(
                            10 * (R[ok, n - 1, 0] - TR[ok, n - 1, 0]).mean()
                        ),
                        "Blue reach error at 2 s (units)": float(
                            10 * np.abs(R[ok, n - 1, 0] - TR[ok, n - 1, 0]).mean()
                        ),
                    }
                )
        print(scen, "done", flush=True)
    json.dump({"val": rows, "kite": kite_rows}, open(OUT / "val.json", "w"), indent=1, default=float)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(kite_rows).round(4).to_string(index=False))


def m2_table(v) -> pd.DataFrame:
    t = pd.DataFrame(v)
    out = []
    for lib in LIBS:
        detail, allp = {}, True
        for scen in SCEN:
            for rad in (8, 16):
                s = t[(t.scenario == scen) & (t.model == f"{lib} weak{rad}")].set_index("h")
                p = all(s.loc[h, "reach below 0"] and s.loc[h, "margin below 0"] for h in (2.0, 4.0))
                p &= bool(s.loc[1.0, "health ratio"] <= 1.05) and bool(s["blow-ups"].max() < 0.01)
                detail[f"{scen} r{rad // 8} s"] = bool(p)
                allp &= p
        out.append({"library": lib, **detail, "M2": bool(allp)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ cf (M4)
def from_branch(states, L, nb):
    blue = [f"alive_blue_{t}" for t in L.types]
    red = [f"alive_red_{t}" for t in L.types]
    parts = []
    for _, g in states.groupby(level=0, sort=False):
        k = int(round(g["branch_at"].iloc[0] / DT))
        if k < len(g) - 8 and g[blue].iloc[k].sum() > 0 and g[red].iloc[k].sum() > 0:
            parts.append(g.iloc[k:])
    out = pd.concat(parts)
    n = out.groupby("seed")["branch"].nunique()
    return out[out["seed"].isin(n[n == nb].index)]


def margin_change(Y, truth, h):
    last = (~np.isnan(truth[:, :, 0])).sum(1) - 1
    i = np.minimum(round(h / DT), last)
    r = np.arange(len(Y))
    return (Y[r, i, 0] - Y[r, i, 1]) - (Y[:, 0, 0] - Y[:, 0, 1])


def cf() -> None:
    F = pickle.load(open(OUT / "fits.pkl", "rb"))
    res, ck = F["fits"], F["ck"]
    rng = np.random.default_rng(0)
    rows, paired = [], []
    for scen in SCEN:
        m = frozen(scen)
        names = [n for _, n in SCEN[scen]["cf"]]
        st = from_branch(load(scen, "cf", branch=True), m.L, len(names))
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach_margins(
                m.L.batch(st, horizon=64, every=10**9, history=0, min_future=1, ctrl=CTRL_K), st, m.L
            )
        TY = B["team"]["Y"]
        g = st.groupby(level=0, sort=False)
        seed = g["seed"].first().loc[B["ep"]].to_numpy()
        branch = g["branch"].first().loc[B["ep"]].to_numpy()

        def wide(v, seed=seed, branch=branch):
            return (
                pd.DataFrame({"s": seed, "b": branch, "v": v})
                .pivot(index="s", columns="b", values="v")
                .to_numpy()
            )

        Ys = {}
        for name, mod in models_for(m, res, ck, scen).items():
            if "step" in name:
                continue
            with np.errstate(all="ignore"):
                Ys[name] = mod.L.team_view(mod.rollout(B))
        for h in HORIZONS:
            T = wide(margin_change(TY, TY, h))
            idx = np.arange(len(T))
            picks = {n: wide(margin_change(Y, TY, h)).argmax(1) for n, Y in Ys.items()}
            picks |= {
                f"always {names[j].removeprefix('cf_')}": np.full(len(T), j)
                for j in range(len(names))
                if names[j] in ("cf_advance_nearest", "cf_kite_focus")
            }
            reg = {n: T.max(1) - T[idx, p] for n, p in picks.items()}
            kite_cols = [j for j, n in enumerate(names) if "kite" in n]
            for n, p in picks.items():
                rows.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "ranker": n,
                        "regret": reg[n].mean(),
                        "picks kite": float(np.isin(p, kite_cols).mean()) if kite_cols else 0.0,
                        "picks the true best": float((reg[n] <= 1e-9).mean()),
                    }
                )
            refs = ["frozen"] + (["always kite_focus"] if scen == "mixed" else [])
            for n in Ys:
                if n == "frozen":
                    continue
                for ref in refs:
                    d = reg[n] - reg[ref]
                    mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                    lo, hi = np.quantile(mm, [0.025, 0.975])
                    paired.append(
                        {
                            "scenario": scen,
                            "h": h,
                            "pair": f"{n} − {ref}",
                            "difference": d.mean(),
                            "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                            "below 0": bool(hi < 0),
                        }
                    )
        print(scen, "states", len(idx), flush=True)
    json.dump({"rows": rows, "paired": paired}, open(OUT / "cf.json", "w"), indent=1, default=float)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))


def m4_table(c) -> pd.DataFrame:
    p = pd.DataFrame(c["paired"])
    r = pd.DataFrame(c["rows"]).set_index(["scenario", "h", "ranker"])["regret"]
    out = []
    for lib in LIBS:
        ok, detail = True, {}
        for scen in SCEN:
            for rad in (8, 16):
                n = f"{lib} weak{rad}"
                s = p[(p.scenario == scen) & (p.pair == f"{n} − frozen")].set_index("h")
                q = all(s.loc[h, "below 0"] for h in (2.0, 4.0))
                if scen == "mixed":
                    q &= any(r[(scen, h, n)] < r[(scen, h, "always kite_focus")] for h in (2.0, 4.0))
                detail[f"{scen} r{rad // 8} s"] = bool(q)
                ok &= q
        out.append({"library": lib, **detail, "M4": bool(ok)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ G0 (docs/planning/34)
CHOSEN = ("p", ("weak", 16))
MODELS_OUT = DATA_DIR / "processed" / "planning" / "musigma_models.pkl"


def regrets(models: dict, st, m, names) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        B = attach_margins(
            m.L.batch(st, horizon=64, every=10**9, history=0, min_future=1, ctrl=CTRL_K), st, m.L
        )
    TY = B["team"]["Y"]
    g = st.groupby(level=0, sort=False)
    seed = g["seed"].first().loc[B["ep"]].to_numpy()
    branch = g["branch"].first().loc[B["ep"]].to_numpy()

    def wide(v):
        return (
            pd.DataFrame({"s": seed, "b": branch, "v": v})
            .pivot(index="s", columns="b", values="v")
            .to_numpy()
        )

    Ys, clipped = {}, {}
    for n, mod in models.items():
        with np.errstate(all="ignore"):
            Ys[n] = mod.L.team_view(mod.rollout(B))
        clipped[n] = float(getattr(mod, "clipped", np.zeros(1)).mean())
    out = {}
    for h in HORIZONS:
        T = wide(margin_change(TY, TY, h))
        idx = np.arange(len(T))
        picks = {n: wide(margin_change(Y, TY, h)).argmax(1) for n, Y in Ys.items()}
        if "cf_kite_focus" in names:
            picks["always kite_focus"] = np.full(len(T), names.index("cf_kite_focus"))
        out[h] = {n: T.max(1) - T[idx, p] for n, p in picks.items()}
    return out, clipped


def g0() -> None:
    F = pickle.load(open(OUT / "fits.pkl", "rb"))
    res, ck = F["fits"], F["ck"]
    rng = np.random.default_rng(0)
    rows, chosen_models = [], {}
    for scen in SCEN:
        m = frozen(scen)
        bounds = train_bounds(load(scen, "train"), m.L.types)
        models = {"frozen": ReachStatePipeline.from_frozen(m, None)}
        for lib in ("p", "nop"):
            mod = variant(m, res, ck, scen, lib, CHOSEN[1])
            mod.bounds = bounds
            models[f"{lib} weak16 clipped"] = mod
        chosen_models[scen] = models["p weak16 clipped"]
        names = [n for _, n in SCEN[scen]["cf"]]
        st = from_branch(load(scen, "cf", branch=True), m.L, len(names))
        R, clipped = regrets(models, st, m, names)
        for h, reg in R.items():
            for n in models:
                if n == "frozen":
                    continue
                d = reg[n] - reg["frozen"]
                mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.quantile(mm, [0.025, 0.975])
                row = {
                    "scenario": scen,
                    "h": h,
                    "model": n,
                    "regret": reg[n].mean(),
                    "frozen": reg["frozen"].mean(),
                    "− frozen 95%": f"[{lo:+.4f}, {hi:+.4f}]",
                    "below frozen": bool(hi < 0),
                    "clip binds (cf starts)": clipped[n],
                }
                if "always kite_focus" in reg:
                    row["always kite/focus"] = reg["always kite_focus"].mean()
                rows.append(row)
        # clip binding on val
        st_v = load(scen, "val")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach_margins(m.L.batch(st_v, horizon=round(max(HORIZONS) / DT), ctrl=CTRL_K), st_v, m.L)
            mod = chosen_models[scen]
            mod.rollout(B)
        rows.append(
            {
                "scenario": scen,
                "h": np.nan,
                "model": "p weak16 clipped (val)",
                "clip binds (val starts)": float(mod.clipped.mean()),
                "blow-ups (val)": float(mod.blown.mean()),
            }
        )
        print(scen, "done", flush=True)
    t = pd.DataFrame(rows)
    ok = True
    for scen in SCEN:
        s = t[(t.scenario == scen) & (t.model == "p weak16 clipped")].set_index("h")
        ok &= bool(s.loc[2.0, "below frozen"] and s.loc[4.0, "below frozen"])
        if scen == "mixed":
            ok &= bool(any(s.loc[h, "regret"] < s.loc[h, "always kite/focus"] for h in (2.0, 4.0)))
    MODELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(chosen_models, open(MODELS_OUT, "wb"))
    json.dump({"rows": rows, "passes": ok}, open(OUT / "g0.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    print("G0 passes:", ok)


# ------------------------------------------------------------------ report
def report() -> None:
    F = pickle.load(open(OUT / "fits.pkl", "rb"))
    res, ck = F["fits"], F["ck"]
    md = lambda t: "```\n" + t.to_string(index=False) + "\n```"  # noqa: E731
    lines = [
        "# μ, σ reach states: results",
        "",
        "Plan: `10_mu_sigma_plan.md` (approved 22 Nov; amendment 1: libraries with p and without).",
        "Script: `experiments/generalisation/mu_sigma.py`. Fitted on train, evaluated on val and the",
        "counterfactual sets. Threshold 0.05 unless stated.",
        "",
        f"Mapping from μ, σ to reach, fitted on train: marines c = {ck['marines'][0]:.3f}, "
        f"k = {ck['marines'][1]:.3f}; mixed c = {ck['mixed'][0]:.3f}, k = {ck['mixed'][1]:.3f}.",
        "",
        "## Equations (M-weak, radius 1 s)",
        "",
    ]
    for k, r in res.items():
        if k.endswith("|weak8"):
            lines.append(f"- `{k}` ({len(kept(r))} groups, R² {r['r2'][MAIN_THR]:.3f}): {equation_text(r)}")
    lines += ["", "## M1: sparse and stable", "", md(m1_table(res).round(3)), ""]
    fq = [
        {
            "fit": k,
            **{f"groups {t}": len(kept(r, t)) for t in THRESHOLDS},
            **{f"R² {t}": round(r["r2"][t], 3) for t in THRESHOLDS},
        }
        for k, r in res.items()
    ]
    lines += ["## Groups and R² by threshold (all fits)", "", md(pd.DataFrame(fq)), ""]
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
            "## Val rollouts",
            "",
            md(t[[c for c in keep if c in t]].round(4)),
            "",
            "## M2",
            "",
            md(m2_table(v["val"])),
            "",
            "## M3: kite windows (exp11 val, 2 s)",
            "",
            md(pd.DataFrame(v["kite"]).round(4)),
            "",
        ]
    if (OUT / "cf.json").exists():
        c = json.load(open(OUT / "cf.json"))
        lines += [
            "## Intent ranking (counterfactual sets)",
            "",
            md(pd.DataFrame(c["rows"]).round(4)),
            "",
            md(pd.DataFrame(c["paired"]).round(4)),
            "",
            "## M4",
            "",
            md(m4_table(c)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["fit", "val", "cf", "g0", "report"])
    {"fit": fit, "val": val, "cf": cf, "g0": g0, "report": report}[p.parse_args().what]()


if __name__ == "__main__":
    main()
