"""The health-spread state experiment (pre-registered in docs/planning/09_spread_state_plan.md).

uv run --extra explore experiments/planning/spread_state.py select   train only: c vs f, H1-H3 -> stdout
uv run --extra explore experiments/planning/spread_state.py fit      train only: final fit, equations, saved
uv run --extra explore experiments/planning/spread_state.py val      H4 (val)
uv run --extra explore experiments/planning/spread_state.py cf       H5 (counterfactual sets)
uv run --extra explore experiments/planning/spread_state.py closed   H6 runs (fresh seeds), then `report`
uv run --extra explore experiments/planning/spread_state.py report   -> docs/planning/10_spread_state.md
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
from model_regret import BRANCHES, from_branch, margin_change, rank_table  # noqa: E402

from sindy_wm.models.intent_model import CTRL, DT, add_controls, add_rates  # noqa: E402
from sindy_wm.models.pipeline import (  # noqa: E402
    OTHER,
    SIDES,
    group_stlsq,
    spread_alive_features,
    spread_rate_features,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
VARIANTS = DATA_DIR / "processed" / "planning" / "intent_variants.pkl"
OUT = DATA_DIR / "processed" / "planning" / "spread_models.pkl"
RES = DATA_DIR / "processed" / "planning" / "spread_results.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "10_spread_state.md"
SETS = {"S1": ("exp06_intents", "marines"), "S2": ("exp08_mixed", "mixed")}
THR_ALIVE, THR_RATE = 0.15, 0.05
N_BOOT = 2000


# ---------------------------------------------------------------- data
def s_cols(L) -> list[str]:
    """Spread state columns in the reach / front order (blue types, then red types)."""
    return [f"s_{side}_{t}" for side in SIDES for t in L.types]


def with_s(sp: pd.DataFrame, L, measure: str) -> pd.DataFrame:
    out = pd.DataFrame(index=sp.index)
    for side in SIDES:
        for t in L.types:
            out[f"s_{side}_{t}"] = (
                sp[f"even_{side}_{t}"] if measure == "even" else 1 - sp[f"fragile_{side}_{t}"]
            )
    return out


def train_rows(exp: str, L, measure: str) -> pd.DataFrame:
    rows = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "rates_train_w9.parquet")
    sp = with_s(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_spread" / "train.parquet"), L, measure)
    sp["t"] = sp.index.get_level_values("step") * DT
    sp = add_rates(sp, cols=tuple(s_cols(L)), window=9)
    out = rows.join(sp.drop(columns="t"), how="inner")
    assert len(out) == len(rows)
    return out


def health_rate(m, rows: pd.DataFrame, s: str, t: str) -> np.ndarray:
    """The frozen health equation's rate for (side, type) on measured reach (as in `Pipeline.fit`)."""
    o, tt = OTHER[s], m.L.types
    R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in tt}
    A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in tt}
    terms = m._terms(R_o, m._sigma_measured(rows, o), A_o, rows["withdraw"].to_numpy(), o, t)
    return np.column_stack(list(terms.values())) @ np.array(list(m.coef[(s, t)].values()))


def alive_design(m, rows, s, t, spread_on: bool):
    h, al = rows[f"health_{s}_{t}"].to_numpy(), rows[f"alive_{s}_{t}"].to_numpy()
    ok = (h > 0.03) & (al > 0)
    g = health_rate(m, rows, s, t) * al / np.maximum(h, 1e-9)
    U, d = rows[CTRL].to_numpy(), rows["distance"].to_numpy()
    if spread_on:
        F = spread_alive_features(rows[f"s_{s}_{t}"].to_numpy(), U, d)
        names = list(F)
        X = np.column_stack([g * F[k] for k in names])
    else:
        names = ["1|1", "1|focus"] if s == "red" else ["1|1"]
        X = np.column_stack([g, g * U[:, 3]]) if s == "red" else g[:, None]
    return (
        X[ok],
        rows[f"d_alive_{s}_{t}"].to_numpy()[ok],
        names,
        rows.index.get_level_values(0).to_numpy()[ok],
    )


def fit_alive(X, y, names, spread_on: bool) -> np.ndarray:
    if not spread_on:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    return group_stlsq(X, y, [n.split("|")[0] for n in names], THR_ALIVE, always=("1",))


def rate_design(m, rows, s, t):
    h, al = rows[f"health_{s}_{t}"].to_numpy(), rows[f"alive_{s}_{t}"].to_numpy()
    both = (rows[[c for c in rows if c.startswith("alive_blue_")]].sum(1) > 0) & (
        rows[[c for c in rows if c.startswith("alive_red_")]].sum(1) > 0
    )
    ok = (al >= 0.2 - 1e-9) & (h > 0.03) & both.to_numpy()
    r = np.maximum(-health_rate(m, rows, s, t) / np.maximum(h, 1e-9), 0)
    F = spread_rate_features(
        rows[f"s_{s}_{t}"].to_numpy(), r, rows[CTRL].to_numpy(), rows["distance"].to_numpy()
    )
    names = list(F)
    X = np.column_stack([F[k] for k in names])
    y = rows[f"d_s_{s}_{t}"].to_numpy()
    ok &= np.isfinite(y)
    return X[ok], y[ok], names, rows.index.get_level_values(0).to_numpy()[ok]


def cv(X, y, names, ep, fit, folds=5):
    pred = np.zeros_like(y)
    for k in range(folds):
        test = ep % folds == k
        c = fit(X[~test], y[~test], names)
        pred[test] = X[test] @ c
    return float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())


# ---------------------------------------------------------------- select / fit (train only)
def select() -> None:
    frozen = pickle.load(open(MODELS, "rb"))
    pooled, report = {}, []
    for measure in ("even", "fragile"):
        sse_f = sst_f = sse_s = sst_s = 0.0
        for scen, (exp, name) in SETS.items():
            m = frozen[name]
            rows = train_rows(exp, m.L, measure)
            for s in SIDES:
                for t in m.L.types:
                    X0, y0, n0, e0 = alive_design(m, rows, s, t, False)
                    a, b = cv(X0, y0, n0, e0, lambda X, y, n: fit_alive(X, y, n, False))
                    X1, y1, n1, e1 = alive_design(m, rows, s, t, True)
                    c, d = cv(X1, y1, n1, e1, lambda X, y, n: fit_alive(X, y, n, True))
                    sse_f, sst_f, sse_s, sst_s = sse_f + a, sst_f + b, sse_s + c, sst_s + d
                    report.append(
                        {
                            "measure": measure,
                            "scenario": scen,
                            "side": s,
                            "type": t,
                            "CV R² frozen form": 1 - a / b,
                            "CV R² with spread": 1 - c / d,
                        }
                    )
        pooled[measure] = {"frozen form": 1 - sse_f / sst_f, "with spread": 1 - sse_s / sst_s}
    table = pd.DataFrame(report).set_index(["measure", "scenario", "side", "type"])
    choice = (
        "even" if pooled["even"]["with spread"] >= pooled["fragile"]["with spread"] - 0.005 else "fragile"
    )
    print(table.round(4).to_string())
    print("pooled:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in pooled.items()})
    print("choice:", choice)
    json.dump(
        {"select": {"pooled": pooled, "table": table.reset_index().to_dict("records"), "choice": choice}},
        open(RES, "w"),
        indent=1,
        default=float,
    )


def fit() -> dict:
    res = json.load(open(RES))
    measure = res["select"]["choice"]
    frozen = pickle.load(open(MODELS, "rb"))
    out, rate_cv = {}, []
    for scen, (exp, name) in SETS.items():
        m = frozen[name]
        rows = train_rows(exp, m.L, measure)
        sm = {"measure": measure, "alive": {}, "rate": {}}
        for s in SIDES:
            for t in m.L.types:
                X, y, names, _ = alive_design(m, rows, s, t, True)
                c = fit_alive(X, y, names, True)
                sm["alive"][(s, t)] = ([n for n, v in zip(names, c, strict=True) if v], c[c != 0])
                X, y, names, ep = rate_design(m, rows, s, t)
                c = group_stlsq(X, y, [n.split("|")[0] for n in names], THR_RATE)
                sm["rate"][(s, t)] = ([n for n, v in zip(names, c, strict=True) if v], c[c != 0])
                a, b = cv(
                    X, y, names, ep, lambda X, y, n: group_stlsq(X, y, [k.split("|")[0] for k in n], THR_RATE)
                )
                med_s = float(np.median(rows.loc[rows[f"alive_{s}_{t}"] >= 0.2, f"s_{s}_{t}"]))
                rate_cv.append(
                    {"scenario": scen, "side": s, "type": t, "s' CV R²": 1 - a / b, "median s": med_s}
                )
        out[scen] = sm
        print(
            f"## {scen} ({name} model), measure s = "
            + ("evenness c" if measure == "even" else "1 − fragile share")
        )
        print(equations(sm, m))
    pickle.dump(out, open(OUT, "wb"))
    res["fit"] = {"rate_cv": rate_cv, "h2": h2(out, frozen), "h3": h3(out)}
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(pd.DataFrame(rate_cv).round(4).to_string())
    print("H2:", res["fit"]["h2"])
    print("H3:", res["fit"]["h3"])
    return out


def equations(sm: dict, m) -> str:
    def expr(names, c):
        groups = {}
        for n, v in zip(names, c, strict=True):
            g, u = n.split("|")
            groups.setdefault(g, []).append(f"{v:+.4f}" + ("" if u == "1" else f"·{u}"))
        return " + ".join(f"{g}·({' '.join(v)})" for g, v in groups.items()) or "0"

    lines = []
    for (s, t), (names, c) in sm["alive"].items():
        side = {"blue": "B", "red": "R"}[s]
        fz = m.p[(s, t)]
        lines.append(f"p_{side}^{t}(s,u) = {expr(names, c)}      [frozen: {fz[0]:.3f} + {fz[1]:.3f}·focus]")
    for (s, t), (names, c) in sm["rate"].items():
        side = {"blue": "B", "red": "R"}[s]
        lines.append(f"s_{side}^{t}' = {expr(names, c)}")
    return "\n".join(lines)


def p_of(names, c, s, focus, hold=0.0, wd=0.0, d=5.0) -> float:
    U = np.array([[1 - hold - wd, hold, wd, focus]])
    F = spread_alive_features(np.array([s]), U, np.array([d]))
    return float(sum(v * F[n][0] for n, v in zip(names, c, strict=True)))


def h2(models: dict, frozen: dict) -> list:
    out = []
    for scen, sm in models.items():
        m = frozen[SETS[scen][1]]
        for t in m.L.types:
            names, c = sm["alive"][("red", t)]
            for s in (1.0, 0.85, 0.7):
                out.append(
                    {
                        "scenario": scen,
                        "type": t,
                        "s": s,
                        "focus − nearest in p": p_of(names, c, s, 1) - p_of(names, c, s, 0),
                        "frozen focus term": m.p[("red", t)][1],
                    }
                )
    return out


def h3(models: dict) -> list:
    out = []
    for scen, sm in models.items():
        for (s, t), (names, c) in sm["rate"].items():
            if s != "red":
                continue
            terms = {n: float(v) for n, v in zip(names, c, strict=True) if "focus" in n.split("|")[1]}
            out.append({"scenario": scen, "type": t, "focus terms in Red's s'": terms})
    return out


# ---------------------------------------------------------------- variants and batches
def variants(scen: str) -> dict:
    frozen = pickle.load(open(MODELS, "rb"))[SETS[scen][1]]
    sm = pickle.load(open(OUT, "rb"))[scen]
    fc = pickle.load(open(VARIANTS, "rb"))[scen]
    sp = copy.deepcopy(frozen)
    sp.spread_model = sm
    spf = copy.deepcopy(sp)
    spf.focus_cost, spf.focus_norm = fc["c"], 1 - fc["c"] * fc["fbar"]
    return {"frozen": frozen, "spread": sp, "spread + focus cost": spf}


def attach(B: dict, states: pd.DataFrame, cols: list[str]) -> dict:
    arr = {ep: g[cols].to_numpy(float) for ep, g in states.groupby(level=0, sort=False)}
    B["spread0"] = np.array([arr[e][k] for e, k in zip(B["ep"], B["k0"], strict=True)])
    return B


def boot_by(x: np.ndarray, grp: np.ndarray, rng) -> tuple[float, float, float]:
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((w * sums).sum() / (w * cnt).sum())
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ---------------------------------------------------------------- H4: val
def val() -> None:
    res = json.load(open(RES))
    rng = np.random.default_rng(0)
    rows = []
    for scen, (exp, _) in SETS.items():
        models = variants(scen)
        L = models["frozen"].L
        st = add_controls(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet"))
        sp = with_s(
            pd.read_parquet(DATA_DIR / "processed" / f"{exp}_spread" / "val.parquet"),
            L,
            res["select"]["choice"],
        )
        st = st.join(sp, how="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach(L.batch(st), st, s_cols(L))
            Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
        T = B["team"]["Y"]
        for h in (1.0, 2.0, 4.0):
            n = round(h / DT)
            ok = ~np.isnan(T[:, n, 0])
            err = {k: 10 * np.abs(Y[ok, n, 2:4] - T[ok, n, 2:4]).sum(1) for k, Y in Ys.items()}
            herr = {k: np.abs(Y[ok, n, :2] - T[ok, n, :2]).sum(1) for k, Y in Ys.items()}
            for k in ("spread", "spread + focus cost"):
                a = boot_by(err[k] - err["frozen"], B["ep"][ok], rng)
                b = boot_by(herr[k] - herr["frozen"], B["ep"][ok], rng)
                rows.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": k,
                        "alive error, frozen (units)": err["frozen"].mean(),
                        "alive error": err[k].mean(),
                        "difference": a[0],
                        "95%": f"[{a[1]:+.4f}, {a[2]:+.4f}]",
                        "health error difference": b[0],
                        "health 95%": f"[{b[1]:+.4f}, {b[2]:+.4f}]",
                    }
                )
    res["val"] = rows
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(pd.DataFrame(rows).set_index(["scenario", "h", "model"]).round(4).to_string())


# ---------------------------------------------------------------- H5: counterfactual sets
def load_cf(exp: str, L, measure: str) -> pd.DataFrame:
    parts = []
    for b, name in enumerate(BRANCHES):
        d = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / f"cf_{name}.parquet")
        sp = with_s(
            pd.read_parquet(DATA_DIR / "processed" / f"{exp}_spread" / f"cf_{name}.parquet"), L, measure
        )
        d = d.join(sp, how="left").reset_index()
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
        states, _ = from_branch(load_cf(exp, L, res["select"]["choice"]), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach(L.batch(states, horizon=64, every=10**9, history=0, min_future=1), states, s_cols(L))
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
            return (
                pd.DataFrame({"s": seed, "b": branch, "v": v})
                .pivot(index="s", columns="b", values="v")
                .to_numpy()
            )

        Fm, W = wide(final), wide(won)
        idx = np.arange(len(Fm))
        for h in (1.0, 2.0, 4.0, 8.0):
            T = wide(margin_change(TY, TY, h))
            regs = {}
            for n, Y in Ys.items():
                M = wide(margin_change(Y, TY, h))
                table, info = rank_table(T, M, rng)
                regs[n] = T.max(1) - T[idx, M.argmax(1)]
                pick = M.argmax(1)
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
            for n in ("spread", "spread + focus cost"):
                d = regs[n] - regs["frozen"]
                mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
                out_w.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": n,
                        "regret − frozen": d.mean(),
                        "95%": f"[{np.quantile(mm, 0.025):+.4f}, {np.quantile(mm, 0.975):+.4f}]",
                    }
                )
    res["cf"] = {"table": out_r, "paired": out_w}
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(pd.DataFrame(out_r).set_index(["scenario", "h", "model"]).round(4).to_string())
    print(pd.DataFrame(out_w).set_index(["scenario", "h", "model"]).round(4).to_string())


# ---------------------------------------------------------------- H6: closed loop, fresh seeds
FRESH = {f"S{i + 1}": 15_000_000 + 100_000 * i for i in range(5)}
SMOKE = 19_800_000
MAIN = ("S1", "S2", "S5")


def closed(n: int = 400, smoke: bool = False) -> None:
    import phase_b as PB

    arms = PB.POLICY_SETS["spread"] + PB.POLICY_SETS["fixed"]
    for scen in FRESH:
        first = SMOKE if smoke else FRESH[scen]
        PB.run(
            scen,
            arms if not smoke else PB.POLICY_SETS["spread"],
            n,
            9,
            first,
            "_smoke" if smoke else "",
            "spread_",
        )


def paired(a, b, rng):
    x = a - b
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT * 2)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), "beats" if lo > 0 else ("loses" if hi < 0 else "ties")


def h6() -> list:
    from analyse_phase_b import BEST_FIXED

    rng = np.random.default_rng(0)
    out = []
    for scen in FRESH:
        d = pd.read_csv(DATA_DIR / "processed" / "planning" / f"spread_{scen}.csv")
        W = d.pivot(index="seed", columns="policy", values="won").astype(float)
        base = BEST_FIXED[scen]
        for arm, ref in (
            ("spread:h3", "model:h3"),
            ("spread-focuscost:h3", "model:h3"),
            ("model:h3", base),
            ("spread:h3", base),
            ("spread-focuscost:h3", base),
        ):
            s = W[[arm, ref]].dropna()
            mean, lo, hi, v = paired(s[arm].to_numpy(), s[ref].to_numpy(), rng)
            out.append(
                {
                    "scenario": scen,
                    "arm": arm,
                    "against": ref.removeprefix("fixed:"),
                    "arm won": s[arm].mean(),
                    "against won": s[ref].mean(),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": v,
                    "n": len(s),
                    "flagged": scen not in MAIN,
                }
            )
        same = (W["spread:h3"] == W["model:h3"]).mean()
        out[-5]["same outcome as frozen planner"] = float(same)
    return out


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def report() -> None:
    res = json.load(open(RES))
    res["closed"] = h6()
    json.dump(res, open(RES, "w"), indent=1, default=float)
    pooled = {k: {a: round(b, 4) for a, b in v.items()} for k, v in res["select"]["pooled"].items()}
    sel = pd.DataFrame(res["select"]["table"]).set_index(["measure", "scenario", "side", "type"])
    lines = [
        "# The health-spread state: results",
        "",
        "Plan: `09_spread_state_plan.md` (pre-registered 21 Nov). Script: "
        "`experiments/planning/spread_state.py`.",
        "Spread tables: `scripts/build_spread_states.py`. Fitted on train only; tested on val, the "
        "counterfactual",
        "sets and 400 fresh seeds per scenario (15 x00 000 +).",
        "",
        "## Choice of the measure and H1 (train, 5-fold CV by battle)",
        "",
        md(sel.round(4)),
        "",
        "Pooled CV R²: " + json.dumps(pooled) + ".",
        f"Chosen: **{res['select']['choice']}** (rule: larger pooled R², ties within 0.005 to evenness).",
        "",
        "## The fitted spread variant",
        "",
        "```",
    ]
    frozen = pickle.load(open(MODELS, "rb"))
    sms = pickle.load(open(OUT, "rb"))
    for scen, sm in sms.items():
        lines += [f"{scen} ({SETS[scen][1]} model)", equations(sm, frozen[SETS[scen][1]]), ""]
    lines += [
        "```",
        "",
        "Spread dynamics, CV R² (train):",
        "",
        md(pd.DataFrame(res["fit"]["rate_cv"]).set_index(["scenario", "side", "type"]).round(4)),
        "",
        "## H2: Red's focus term at three levels of evenness",
        "",
        md(pd.DataFrame(res["fit"]["h2"]).set_index(["scenario", "type", "s"]).round(4)),
        "",
        "## H3: focus terms in Red's evenness dynamics",
        "",
        "```",
        *[json.dumps(r) for r in res["fit"]["h3"]],
        "```",
        "",
        "## H4: alive error on val (team, units; paired by start point, bootstrap by battle)",
        "",
        md(pd.DataFrame(res["val"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        "## H5: ranking on the counterfactual sets",
        "",
        md(pd.DataFrame(res["cf"]["table"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        md(pd.DataFrame(res["cf"]["paired"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        "## H6: closed loop on fresh seeds (wins, paired over 400 seeds)",
        "",
        md(pd.DataFrame(res["closed"]).set_index(["scenario", "arm", "against"]).round(3)),
        "",
    ]
    for scen in FRESH:
        d = pd.read_csv(DATA_DIR / "processed" / "planning" / f"spread_{scen}.csv")
        pl = d[d.policy.str.contains(":h")]
        pick = [c for c in pl if c.startswith("pick_")]
        lines += [
            f"{scen} wins per policy and planner picks:",
            "",
            md(d.groupby("policy")[["won", "margin", "blue_preserved"]].mean().round(3)),
            "",
            md(pl.groupby("policy")[[*pick, "switches", "replans"]].mean().round(3)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-80:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["select", "fit", "val", "cf", "closed", "smoke", "report"])
    a = parser.parse_args()
    if a.what in ("closed", "smoke"):
        closed(5 if a.what == "smoke" else 400, a.what == "smoke")
        return
    {"select": select, "fit": fit, "val": val, "cf": cf, "report": report}[a.what]()


if __name__ == "__main__":
    main()
