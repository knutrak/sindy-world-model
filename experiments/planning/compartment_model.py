"""The compartment variant (pre-registered in docs/planning/13_compartment_plan.md).

uv run --extra explore experiments/planning/compartment_model.py fit      train: fit, equations, H1, H2
uv run --extra explore experiments/planning/compartment_model.py val      H3 (val)
uv run --extra explore experiments/planning/compartment_model.py cf       H4 (counterfactual sets)
uv run --extra explore experiments/planning/compartment_model.py smoke    5 seeds per scenario, 19 900 000 +
uv run --extra explore experiments/planning/compartment_model.py closed   H5 (fresh seeds 16 x00 000 +)
uv run --extra explore experiments/planning/compartment_model.py report   -> docs/planning/14_compartments.md
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compartments import power_fit, power_pred  # noqa: E402
from model_regret import BRANCHES, from_branch, margin_change, rank_table  # noqa: E402
from spread_state import health_rate  # noqa: E402

from sindy_wm.models.intent_model import CTRL, DT, add_controls  # noqa: E402
from sindy_wm.models.pipeline import (  # noqa: E402
    COMPART_ALWAYS,
    SIDES,
    compart_features,
    group_stlsq,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

P = DATA_DIR / "processed" / "planning"
MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
VARIANTS = P / "intent_variants.pkl"
OUT = P / "compart_models.pkl"
RES = P / "compart_results.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "14_compartments.md"
SETS = {"S1": ("exp06_intents", "marines"), "S2": ("exp08_mixed", "mixed")}
THR = 0.05
LAWS = ("I", "KW", "KF", "sigma")
N_BOOT = 2000
FRESH = {f"S{i + 1}": 16_000_000 + 100_000 * i for i in range(5)}
SMOKE = 19_900_000
MAIN = ("S1", "S2", "S5")


# ------------------------------------------------------------------ train windows
def windows(scen: str, m) -> pd.DataFrame:
    """The 1 s windows of `12` (relative wounded, train), with d-hat and the reach share at the window start;
    only windows with a steady Blue intent and both sides alive (inner join on the rate rows)."""
    exp = SETS[scen][0]
    t = pd.read_parquet(P / f"compartments_rel_{scen}.parquet")
    sh = pd.read_parquet(P / f"start_health_{exp}.parquet")
    t = t.merge(sh, on="episode_idx", how="left")
    t["h0"] = np.where(t["side"] == "blue", t["start_health_blue"], t["start_health_red"])
    t = t[t["steady"] & (t["A0"] > 0)]
    rows = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "rates_train_w9.parquet")
    parts = []
    for side in SIDES:
        for typ in m.L.types:
            al = rows[f"alive_{side}_{typ}"].to_numpy()
            parts.append(
                pd.DataFrame(
                    {
                        "episode_idx": rows.index.get_level_values(0),
                        "step": rows.index.get_level_values(1),
                        "side": side,
                        "type": typ,
                        "dhat": np.maximum(-health_rate(m, rows, side, typ), 0) * 10,
                        "rho": np.divide(
                            rows[f"reach_{side}_{typ}"].to_numpy(), al, out=np.zeros_like(al), where=al > 0
                        ),
                        **{c: rows[c].to_numpy() for c in CTRL},
                    }
                )
            )
    return t.merge(pd.concat(parts), on=["episode_idx", "step", "side", "type"], how="inner")


def design(g: pd.DataFrame, law: str, use_hw: bool = True):
    A, Wn = g["A0"].to_numpy(float), g["W0"].to_numpy(float)
    H, HW = g["H0"].to_numpy(float), g["HW0"].to_numpy(float)
    thr = 0.5 * g["h0"].to_numpy(float)
    w = Wn / A
    mF = np.divide(H - HW, (A - Wn) * thr, out=np.ones_like(A), where=(A - Wn) > 0)
    mW = np.divide(HW, Wn * thr, out=np.ones_like(A), where=Wn > 0)
    F = compart_features(
        law, g["dhat"].to_numpy(), w, Wn, mF, mW, g["rho"].to_numpy(), g[CTRL].to_numpy(), use_hw
    )
    names = list(F)
    X = np.column_stack([F[k] for k in names])
    if law == "I":
        y = g["new_w"].to_numpy(float)
    elif law == "KW":
        y = g["kills_w"].to_numpy(float)
    elif law == "KF":
        y = (g["kills"] - g["kills_w"]).to_numpy(float)
    else:
        ok = g["dmg"].to_numpy() > 0.02
        X = X[ok]
        y = (g["dmg_w"] / g["dmg"]).to_numpy(float)[ok]
        return X, y, names, g["episode_idx"].to_numpy()[ok]
    return X, y, names, g["episode_idx"].to_numpy()


def fit_law(X, y, names, law):
    return group_stlsq(X, y, [n.split("|")[0] for n in names], THR, always=(COMPART_ALWAYS[law],))


def cv(X, y, ep, fit, pred=lambda c, X: X @ c, folds=5) -> float:
    out = np.zeros_like(y)
    for k in range(folds):
        te = ep % folds == k
        out[te] = pred(fit(X[~te], y[~te]), X[te])
    return float(1 - ((y - out) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def equations(laws: dict) -> str:
    lines = []
    for (s, t), L in laws.items():
        side = {"blue": "B", "red": "R"}[s]
        for law, (names, c) in L.items():
            groups = {}
            for n, v in zip(names, c, strict=True):
                gname, u = n.split("|")
                groups.setdefault(gname, []).append(f"{v:+.4f}" + ("" if u == "1" else f"·{u}"))
            body = " + ".join(f"{g}·({' '.join(v)})" for g, v in groups.items())
            lines.append(f"{law}_{side}^{t} = {body}")
    return "\n".join(lines)


def fit() -> None:
    frozen = pickle.load(open(MODELS, "rb"))
    models, h1, h2 = {}, [], []
    for scen, (_, name) in SETS.items():
        m = frozen[name]
        t = windows(scen, m)
        for variant, use_hw in (("compartments", True), ("without H_W", False)):
            laws = {}
            for (side, typ), g in t.groupby(["side", "type"]):
                L = {}
                for law in LAWS:
                    X, y, names, ep = design(g, law, use_hw)
                    c = fit_law(X, y, names, law)
                    L[law] = ([n for n, v in zip(names, c, strict=True) if v], c[c != 0])
                    h1.append(
                        {
                            "scenario": scen,
                            "variant": variant,
                            "side": side,
                            "type": typ,
                            "law": law,
                            "CV R²": cv(
                                X, y, ep, lambda X_, y_, names=names, law=law: fit_law(X_, y_, names, law)
                            ),
                            "windows": len(y),
                        }
                    )
                laws[(side, typ)] = L
            models.setdefault(scen, {})[variant] = laws
        # H2: kills over 1 s, compartments (KW + KF) vs the frozen law refit, both with d-hat
        for (side, typ), g in t.groupby(["side", "type"]):
            ep = g["episode_idx"].to_numpy()
            Xw, yw, nw, _ = design(g, "KW")
            Xf, yf, nf, _ = design(g, "KF")
            ytot = yw + yf
            Xb = np.column_stack([Xw, Xf])
            split = Xw.shape[1]

            def fit_both(X, y, split=split, nw=nw, nf=nf):
                return (fit_law(X[:, :split], y[0], nw, "KW"), fit_law(X[:, split:], y[1], nf, "KF"))

            pred = np.zeros_like(ytot)
            for k in range(5):
                te = ep % 5 == k
                cw, cf_ = fit_both(Xb[~te], (yw[~te], yf[~te]))
                pred[te] = Xb[te, :split] @ cw + Xb[te, split:] @ cf_
            r2c = float(1 - ((ytot - pred) ** 2).sum() / ((ytot - ytot.mean()) ** 2).sum())
            frac = g["dhat"].to_numpy() / g["H0"].clip(lower=1e-6).to_numpy()
            Xp = np.column_stack([g["A0"], frac, g["focus"] if "focus" in g else g["target"].eq("focus")])
            r2p = cv(Xp.astype(float), ytot, ep, power_fit, power_pred)
            h2.append(
                {
                    "scenario": scen,
                    "side": side,
                    "type": typ,
                    "kills CV R², compartments": r2c,
                    "frozen law (refit)": r2p,
                }
            )
        print(f"## {scen} ({name} model), compartments")
        print(equations(models[scen]["compartments"]))
    pickle.dump(models, open(OUT, "wb"))
    json.dump({"h1": h1, "h2": h2}, open(RES, "w"), indent=1, default=float)
    H1 = pd.DataFrame(h1).pivot_table(
        index=["scenario", "side", "type", "law"], columns="variant", values="CV R²"
    )
    print(H1.round(4).to_string())
    print(pd.DataFrame(h2).set_index(["scenario", "side", "type"]).round(4).to_string())


# ------------------------------------------------------------------ variants and batches
def variants(scen: str) -> dict:
    frozen = pickle.load(open(MODELS, "rb"))[SETS[scen][1]]
    cm = pickle.load(open(OUT, "rb"))[scen]
    fc = pickle.load(open(VARIANTS, "rb"))[scen]
    out = {"frozen": frozen}
    for name, laws, use_hw, focus in (
        ("compartments", cm["compartments"], True, False),
        ("compartments + focus cost", cm["compartments"], True, True),
        ("without H_W", cm["without H_W"], False, False),
    ):
        m = copy.deepcopy(frozen)
        m.compart_model = {"laws": laws, "use_hw": use_hw}
        if focus:
            m.focus_cost, m.focus_norm = fc["c"], 1 - fc["c"] * fc["fbar"]
        out[name] = m
    return out


def ckeys(L) -> list[str]:
    return [f"{side}_{t}" for side in SIDES for t in L.types]


def attach(B: dict, states: pd.DataFrame, L) -> dict:
    keys = ckeys(L)
    for k0, n in (("W0", "W"), ("HW0", "HW"), ("h00", "h0")):
        cols = [f"{n}_{k}" for k in keys]
        arr = {ep: g[cols].to_numpy(float) for ep, g in states.groupby(level=0, sort=False)}
        B[k0] = np.array([arr[e][k] for e, k in zip(B["ep"], B["k0"], strict=True)])
    return B


def boot_by(x, grp, rng):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(N_BOOT):
        wts = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((wts * sums).sum() / (wts * cnt).sum())
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def start_ok(exp: str, name: str, ep: np.ndarray, offset: int = 0) -> np.ndarray:
    """Battles where both sides started at >= 80% of full health."""
    from sindy_wm.io import storage
    from sindy_wm.paths import RAW_DIR

    e = storage.load_episode_index(RAW_DIR / exp / name)
    good = set(
        e.loc[(e["start_health_blue"] >= 0.8) & (e["start_health_red"] >= 0.8), "episode_idx"] + offset
    )
    return np.isin(ep, list(good))


# ------------------------------------------------------------------ H3: val
def val() -> None:
    res = json.load(open(RES))
    rng = np.random.default_rng(0)
    rows = []
    for scen, (exp, _) in SETS.items():
        models = variants(scen)
        L = models["frozen"].L
        st = add_controls(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet"))
        st = st.join(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_compart" / "val.parquet"), how="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach(L.batch(st), st, L)
            Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
        T = B["team"]["Y"]
        good = start_ok(exp, "val", B["ep"])
        for subset, keep in (("both sides started >= 80%", good), ("all starts", np.ones_like(good))):
            for h in (1.0, 2.0, 4.0):
                n = round(h / DT)
                ok = ~np.isnan(T[:, n, 0]) & keep
                err = {k: 10 * np.abs(Y[ok, n, 2:4] - T[ok, n, 2:4]).sum(1) for k, Y in Ys.items()}
                herr = {k: np.abs(Y[ok, n, :2] - T[ok, n, :2]).sum(1) for k, Y in Ys.items()}
                for k in list(models)[1:]:
                    a = boot_by(err[k] - err["frozen"], B["ep"][ok], rng)
                    b = boot_by(herr[k] - herr["frozen"], B["ep"][ok], rng)
                    rows.append(
                        {
                            "scenario": scen,
                            "subset": subset,
                            "h": h,
                            "model": k,
                            "alive error, frozen": err["frozen"].mean(),
                            "alive error": err[k].mean(),
                            "difference": a[0],
                            "95%": f"[{a[1]:+.4f}, {a[2]:+.4f}]",
                            "read-out": "better" if a[2] < 0 else ("worse" if a[1] > 0 else "no difference"),
                            "health error difference": b[0],
                            "health 95%": f"[{b[1]:+.4f}, {b[2]:+.4f}]",
                        }
                    )
    res["val"] = rows
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(pd.DataFrame(rows).set_index(["scenario", "subset", "h", "model"]).round(4).to_string())


# ------------------------------------------------------------------ H4: counterfactual sets
def load_cf(exp: str) -> pd.DataFrame:
    parts = []
    for b, name in enumerate(BRANCHES):
        d = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / f"cf_{name}.parquet")
        c = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_compart" / f"cf_{name}.parquet")
        d = d.join(c, how="left").reset_index()
        d["branch"] = b
        d["episode_idx"] = d["episode_idx"] + 10_000 * (b + 1)
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_controls(pd.concat(parts))


def cf() -> None:
    res = json.load(open(RES))
    rng = np.random.default_rng(0)
    out_r, out_w = [], []
    for scen, (exp, _) in SETS.items():
        models = variants(scen)
        L = models["frozen"].L
        states, _ = from_branch(load_cf(exp), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach(L.batch(states, horizon=64, every=10**9, history=0, min_future=1), states, L)
            Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
            Bf = L.batch(states, horizon=720, every=10**9, history=0, min_future=1)
        TY, TF = B["team"]["Y"], Bf["team"]["Y"]
        g = states.groupby(level=0, sort=False)
        seed = g["seed"].first().loc[B["ep"]].to_numpy()
        branch = g["branch"].first().loc[B["ep"]].to_numpy()
        won = g["battle_won"].first().loc[B["ep"]].to_numpy().astype(float)
        end = (~np.isnan(TF[:, :, 0])).sum(1) - 1
        final = TF[np.arange(len(TF)), end, 0] - TF[np.arange(len(TF)), end, 1]

        def wide(v, seed=seed, branch=branch):
            f = pd.DataFrame({"s": seed, "b": branch, "v": v})
            return f.pivot(index="s", columns="b", values="v").to_numpy()

        Fm, W = wide(final), wide(won)
        idx = np.arange(len(Fm))
        for h in (1.0, 2.0, 4.0, 8.0):
            T = wide(margin_change(TY, TY, h))
            regs = {}
            for n, Y in Ys.items():
                M = wide(margin_change(Y, TY, h))
                table, _ = rank_table(T, M, rng)
                pick = M.argmax(1)
                regs[n] = T.max(1) - T[idx, pick]
                out_r.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": n,
                        "regret": regs[n].mean(),
                        "rank corr": table.loc["model", "rank correlation"],
                        "always adv/nearest": table.loc["always advance/nearest", "regret"],
                        "picks focus": float(np.isin(pick, [1, 3]).mean()),
                        "eventual won": W[idx, pick].mean(),
                        "eventual regret": (Fm.max(1) - Fm[idx, pick]).mean(),
                    }
                )
            for n in list(models)[1:]:
                d = regs[n] - regs["frozen"]
                mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
                lo, hi = np.quantile(mm, [0.025, 0.975])
                out_w.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": n,
                        "regret − frozen": d.mean(),
                        "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                    }
                )
    res["cf"] = {"table": out_r, "paired": out_w}
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(pd.DataFrame(out_r).set_index(["scenario", "h", "model"]).round(4).to_string())
    print(pd.DataFrame(out_w).set_index(["scenario", "h", "model"]).round(4).to_string())


# ------------------------------------------------------------------ H5: closed loop
def closed(n: int = 400, smoke: bool = False) -> None:
    import phase_b as PB

    arms = PB.POLICY_SETS["compart"] + ([] if smoke else PB.POLICY_SETS["fixed"])
    for scen in FRESH:
        PB.run(scen, arms, n, 9, SMOKE if smoke else FRESH[scen], "_smoke" if smoke else "", "compart_")


def h5() -> list:
    from analyse_phase_b import BEST_FIXED

    rng = np.random.default_rng(0)
    out = []
    for scen in FRESH:
        d = pd.read_csv(P / f"compart_{scen}.csv")
        good = d.groupby("seed")[["start_health_blue", "start_health_red"]].first().min(1) >= 0.8
        W_all = d.pivot(index="seed", columns="policy", values="won").astype(float)
        base = BEST_FIXED[scen]
        for subset, Wt in (("all starts", W_all), ("both sides started >= 80%", W_all.loc[good[good].index])):
            for arm, ref in (
                ("compart:h3", "model:h3"),
                ("compart-focuscost:h3", "model:h3"),
                ("model:h3", base),
                ("compart:h3", base),
                ("compart-focuscost:h3", base),
            ):
                s = Wt[[arm, ref]].dropna()
                x = s[arm].to_numpy() - s[ref].to_numpy()
                mm = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT * 2)]
                lo, hi = np.quantile(mm, [0.025, 0.975])
                out.append(
                    {
                        "scenario": scen,
                        "subset": subset,
                        "arm": arm,
                        "against": ref.removeprefix("fixed:"),
                        "arm won": s[arm].mean(),
                        "against won": s[ref].mean(),
                        "difference": x.mean(),
                        "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                        "read-out": "beats" if lo > 0 else ("loses" if hi < 0 else "ties"),
                        "n": len(s),
                        "flagged": scen not in MAIN,
                    }
                )
    return out


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def report() -> None:
    res = json.load(open(RES))
    res["closed"] = h5()
    json.dump(res, open(RES, "w"), indent=1, default=float)
    cm = pickle.load(open(OUT, "rb"))
    lines = [
        "# Health compartments in the world model: results",
        "",
        "Plan: `13_compartment_plan.md` (pre-registered, approved 21 Nov). Script: "
        "`experiments/planning/compartment_model.py`.",
        "States: `scripts/build_compartment_states.py`. Fitted on train only; tested on val, the "
        "counterfactual",
        "sets and 400 fresh seeds per scenario (16 x00 000 +).",
        "",
        "## The fitted laws (train)",
        "",
        "```",
    ]
    for scen in SETS:
        lines += [f"{scen} ({SETS[scen][1]} model)", equations(cm[scen]["compartments"]), ""]
    lines += [
        "```",
        "",
        "## H1: CV R² per law (train, 5 folds by battle)",
        "",
        md(
            pd.DataFrame(res["h1"])
            .pivot_table(index=["scenario", "side", "type", "law"], columns="variant", values="CV R²")
            .round(4)
        ),
        "",
        "## H2: kills over 1 s windows (CV R², d-hat for both)",
        "",
        md(pd.DataFrame(res["h2"]).set_index(["scenario", "side", "type"]).round(4)),
        "",
        "## H3: alive error on val (team, units; paired, bootstrap by battle)",
        "",
        md(pd.DataFrame(res["val"]).set_index(["scenario", "subset", "h", "model"]).round(4)),
        "",
        "## H4: ranking on the counterfactual sets",
        "",
        md(pd.DataFrame(res["cf"]["table"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        md(pd.DataFrame(res["cf"]["paired"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        "## H5: closed loop on fresh seeds (wins, paired)",
        "",
        md(pd.DataFrame(res["closed"]).set_index(["scenario", "subset", "arm", "against"]).round(3)),
        "",
    ]
    for scen in FRESH:
        d = pd.read_csv(P / f"compart_{scen}.csv")
        pl = d[d.policy.str.contains(":h")]
        pick = [c for c in pl if c.startswith("pick_")]
        lines += [
            f"{scen}: wins per policy and planner picks",
            "",
            md(d.groupby("policy")[["won", "margin", "blue_preserved"]].mean().round(3)),
            "",
            md(pl.groupby("policy")[[*pick, "switches", "replans"]].mean().round(3)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-60:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["fit", "val", "cf", "smoke", "closed", "report"])
    a = parser.parse_args()
    if a.what in ("smoke", "closed"):
        closed(5 if a.what == "smoke" else 400, a.what == "smoke")
        return
    {"fit": fit, "val": val, "cf": cf, "report": report}[a.what]()


if __name__ == "__main__":
    main()
