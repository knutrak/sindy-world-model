"""Compartments discovered, not bolted on (pre-registered in docs/planning/15_compartment_discovered_plan.md).

uv run --extra explore experiments/planning/compartment_discovered.py fit     train: refit, equations, D1, D2
uv run --extra explore experiments/planning/compartment_discovered.py val     D3 (val)
uv run --extra explore experiments/planning/compartment_discovered.py cf      D4 (counterfactual sets)
uv run --extra explore experiments/planning/compartment_discovered.py smoke   5 seeds, 19 950 000 +
uv run --extra explore experiments/planning/compartment_discovered.py closed  D5 (fresh seeds 17 x00 000 +)
uv run --extra explore experiments/planning/compartment_discovered.py report  -> docs/planning/16_...md
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compartment_model as CM  # noqa: E402

from sindy_wm.models.discovered import DiscoveredPipeline, smooth_rate  # noqa: E402
from sindy_wm.models.intent_model import CTRL, DT, add_controls  # noqa: E402
from sindy_wm.models.pipeline import SIDES, compart_derived, compart_features  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

P = DATA_DIR / "processed" / "planning"
OUT = P / "discovered_models.pkl"
RES = P / "discovered_results.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "16_compartment_discovered.md"
CONFIG = PROJECT_ROOT / "configs" / "generalisation_pipeline.yaml"
SETS = {"S1": ("exp06_intents", "marines"), "S2": ("exp08_mixed", "mixed")}
SPEC = ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd")
FRESH = {f"S{i + 1}": 17_000_000 + 100_000 * i for i in range(5)}
SMOKE = 19_950_000
N_BOOT = 2000


def train_rows(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "rates_train_w9.parquet")
    comp = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_compart" / "train.parquet")
    ev = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_compart" / "train_events.parquet")
    out = rows.join(comp, how="inner")
    assert len(out) == len(rows)
    return out, ev


def new_model(name: str) -> DiscoveredPipeline:
    cfg = yaml.safe_load(CONFIG.read_text())
    kw = {k: cfg["pipeline"][k] for k in SPEC}
    s = cfg["scenarios"][name]
    return DiscoveredPipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **kw)


def describe(m: DiscoveredPipeline, frozen) -> str:
    side = {"blue": "B", "red": "R"}
    lines = []
    for (s, t), c in m.coef.items():
        terms = " ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v)
        fz = " ".join(f"{v:+.4f}·{k}" for k, v in frozen.coef[(s, t)].items() if v)
        lines.append(f"H_{side[s]}^{t}' = {terms}\n        [frozen: {fz}]")
    d = " ".join(f"{c:+.3f}·{n}" for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c)
    lines.append(f"d' = {d}")
    for (s, t), L in m.compart_model["laws"].items():
        for law, (names, c) in L.items():
            groups = {}
            for n, v in zip(names, c, strict=True):
                g, u = n.split("|")
                groups.setdefault(g, []).append(f"{v:+.4f}" + ("" if u == "1" else f"·{u}"))
            body = " + ".join(f"{g}·({' '.join(v)})" for g, v in groups.items())
            lines.append(f"{law}_{side[s]}^{t} = {body}      [in-sample R² {m.law_r2[(law, s, t)]:.3f}]")
    return "\n".join(lines)


def fit() -> None:
    frozen = pickle.load(open(CM.MODELS, "rb"))
    models, d1, d2 = {}, [], []
    for scen, (exp, name) in SETS.items():
        rows, ev = train_rows(exp)
        out = {}
        for variant, use_hw in (("discovered", True), ("discovered, no H_W", False)):
            out[variant] = new_model(name).fit_discovered(rows, ev, use_hw)
        m = out["discovered"]
        models[scen] = out
        for key, groups in m.selection.items():
            d1.append({"scenario": scen, "equation": " ".join(key), "compartment groups kept": groups})
        d2 += cv_rates(m, frozen[name], rows, ev, scen)
        print(f"## {scen} ({name}), discovered")
        print(describe(m, frozen[name]))
    pickle.dump(models, open(OUT, "wb"))
    json.dump({"d1": d1, "d2": d2}, open(RES, "w"), indent=1, default=str)
    print(pd.DataFrame(d1).to_string())
    print(pd.DataFrame(d2).set_index(["scenario", "side", "type"]).round(4).to_string())


def cv_rates(m, frozen, rows, ev, scen, folds: int = 5) -> list:
    """D2: CV R² (5 folds by battle) of the rates of A and W from the four laws (refitted per fold on the
    full-fit health equation), against the frozen alive law refitted per fold for A'."""
    from sindy_wm.models.discovered import THR_LAWS, _group
    from sindy_wm.models.pipeline import COMPART_ALWAYS, two_level_stlsq

    m._ctx = m._rows_context(rows)
    ep = rows.index.get_level_values(0).to_numpy()
    out = []
    for s in SIDES:
        for t in m.L.types:
            A10 = rows[f"alive_{s}_{t}"].to_numpy(float)
            H10 = rows[f"health_{s}_{t}"].to_numpy(float)
            A, Wn, w, mF, mW = compart_derived(
                A10,
                rows[f"W_{s}_{t}"].to_numpy(float),
                H10,
                rows[f"HW_{s}_{t}"].to_numpy(float),
                rows[f"h0_{s}_{t}"].to_numpy(float),
            )
            d = np.maximum(-m.health_rate_rows(rows, s, t), 0) * 10
            rho = np.divide(rows[f"reach_{s}_{t}"].to_numpy(), A10, out=np.zeros_like(A10), where=A10 > 1e-9)
            rate = {k: smooth_rate(ev[f"{k}_{s}_{t}"], rows.index) for k in ("I", "KW", "KF")}
            U = rows[CTRL].to_numpy(float)
            # the pipeline's own rows for the alive law: health above 0.03 and units alive
            ok = (A10 > 0) & (H10 > 0.03) & np.isfinite(rate["I"])
            feats = {law: compart_features(law, d, w, Wn, mF, mW, rho, U) for law in ("I", "KW", "KF")}
            pred = {law: np.zeros(len(rows)) for law in feats}
            for k in range(folds):
                tr, te = ok & (ep % folds != k), ok & (ep % folds == k)
                for law, F in feats.items():
                    nm = list(F)
                    X = np.column_stack([F[n] for n in nm])
                    gg = [_group(n) for n in nm]
                    c = two_level_stlsq(
                        X[tr],
                        rate[law][tr],
                        [x[0] for x in gg],
                        THR_LAWS,
                        [x[1] for x in gg],
                        always=(COMPART_ALWAYS[law],),
                    )
                    pred[law][te] = X[te] @ c
            # both laws are scored against the same target: the smoothed alive rate (d_alive), in units / s
            yA = rows[f"d_alive_{s}_{t}"].to_numpy()[ok] * 10
            pA = -(np.maximum(pred["KW"], 0) + np.maximum(pred["KF"], 0))[ok]
            yW = (rate["I"] - rate["KW"])[ok]
            pW = (np.maximum(pred["I"], 0) - np.maximum(pred["KW"], 0))[ok]
            # frozen alive law, refitted per fold: d_alive = p g (+ pf g focus for Red), g = H-hat' A / H
            g = -d / 10 * A10 / np.maximum(H10, 1e-9)
            Xf = np.column_stack([g, g * U[:, 3]]) if s == "red" else g[:, None]
            ya = rows[f"d_alive_{s}_{t}"].to_numpy() * 10
            pf = np.zeros(len(rows))
            for k in range(folds):
                tr, te = ok & (ep % folds != k), ok & (ep % folds == k)
                c = np.linalg.lstsq(Xf[tr], ya[tr], rcond=None)[0]
                pf[te] = Xf[te] @ c * 10

            def r2(y, p):
                return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

            out.append(
                {
                    "scenario": scen,
                    "side": s,
                    "type": t,
                    "A' from the laws": r2(yA, pA),
                    "A' frozen law (refit)": r2(ya[ok], pf[ok] / 10),
                    "W' from the laws (event rates)": r2(yW, pW),
                }
            )
    m._ctx = None
    return out


def d2_only() -> None:
    """Recompute D2 from the saved fit (after the comparator fix of 21 Nov)."""
    frozen = pickle.load(open(CM.MODELS, "rb"))
    models = pickle.load(open(OUT, "rb"))
    res = json.load(open(RES))
    res["d2"] = []
    for scen, (exp, name) in SETS.items():
        rows, ev = train_rows(exp)
        res["d2"] += cv_rates(models[scen]["discovered"], frozen[name], rows, ev, scen)
    json.dump(res, open(RES, "w"), indent=1, default=str)
    print(pd.DataFrame(res["d2"]).set_index(["scenario", "side", "type"]).round(4).to_string())


def variants(scen: str) -> dict:
    frozen = pickle.load(open(CM.MODELS, "rb"))[SETS[scen][1]]
    bolted = copy.deepcopy(frozen)
    bolted.compart_model = {"laws": pickle.load(open(CM.OUT, "rb"))[scen]["compartments"], "use_hw": True}
    disc = pickle.load(open(OUT, "rb"))[scen]
    return {
        "frozen": frozen,
        "bolted-on": bolted,
        "discovered": disc["discovered"],
        "discovered, no H_W": disc["discovered, no H_W"],
    }


def val() -> None:
    """D3: alive, health and margin errors at 1, 2, 4 s on val, against frozen (and bolted-on), paired."""
    res = json.load(open(RES))
    rng = np.random.default_rng(0)
    rows = []
    for scen, (exp, _) in SETS.items():
        models = variants(scen)
        L = models["frozen"].L
        st = add_controls(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet"))
        st = st.join(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_compart" / "val.parquet"), how="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            B = CM.attach(L.batch(st), st, L)
            Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
        T = B["team"]["Y"]
        good = CM.start_ok(exp, "val", B["ep"])
        for subset, keep in (("both sides started >= 80%", good), ("all starts", np.ones_like(good))):
            for h in (1.0, 2.0, 4.0):
                n = round(h / DT)
                ok = ~np.isnan(T[:, n, 0]) & keep
                err = {
                    "alive": {k: 10 * np.abs(Y[ok, n, 2:4] - T[ok, n, 2:4]).sum(1) for k, Y in Ys.items()},
                    "health": {k: np.abs(Y[ok, n, :2] - T[ok, n, :2]).sum(1) for k, Y in Ys.items()},
                    "margin": {
                        k: np.abs((Y[ok, n, 0] - Y[ok, n, 1]) - (T[ok, n, 0] - T[ok, n, 1]))
                        for k, Y in Ys.items()
                    },
                }
                for k in ("discovered", "discovered, no H_W", "bolted-on"):
                    for ref in ("frozen", "bolted-on"):
                        if k == ref or (ref == "bolted-on" and k == "bolted-on"):
                            continue
                        row = {"scenario": scen, "subset": subset, "h": h, "model": k, "against": ref}
                        for q, e in err.items():
                            a = CM.boot_by(e[k] - e[ref], B["ep"][ok], rng)
                            row[f"{q} error"] = e[k].mean()
                            row[f"{q} − ref"] = a[0]
                            row[f"{q} 95%"] = f"[{a[1]:+.4f}, {a[2]:+.4f}]"
                        rows.append(row)
    res["val"] = rows
    json.dump(res, open(RES, "w"), indent=1, default=str)
    t = pd.DataFrame(rows).set_index(["scenario", "subset", "h", "model", "against"])
    print(t.round(4).to_string())


def cf() -> None:
    """D4: regret and the picks' eventual wins on the counterfactual sets, against frozen and bolted-on."""
    res = json.load(open(RES))
    rng = np.random.default_rng(0)
    out_r, out_w = [], []
    for scen, (exp, _) in SETS.items():
        models = variants(scen)
        L = models["frozen"].L
        states, _ = CM.from_branch(CM.load_cf(exp), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = CM.attach(L.batch(states, horizon=64, every=10**9, history=0, min_future=1), states, L)
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
            T = wide(CM.margin_change(TY, TY, h))
            regs, picks = {}, {}
            for n, Y in Ys.items():
                M = wide(CM.margin_change(Y, TY, h))
                table, _ = CM.rank_table(T, M, rng)
                picks[n] = M.argmax(1)
                regs[n] = T.max(1) - T[idx, picks[n]]
                out_r.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": n,
                        "regret": regs[n].mean(),
                        "rank corr": table.loc["model", "rank correlation"],
                        "always adv/nearest": table.loc["always advance/nearest", "regret"],
                        "picks focus": float(np.isin(picks[n], [1, 3]).mean()),
                        "eventual won": W[idx, picks[n]].mean(),
                        "eventual regret": (Fm.max(1) - Fm[idx, picks[n]]).mean(),
                    }
                )
            for n in ("discovered", "discovered, no H_W", "bolted-on"):
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
                        "picks differ from frozen": float((picks[n] != picks["frozen"]).mean()),
                    }
                )
    res["cf"] = {"table": out_r, "paired": out_w}
    json.dump(res, open(RES, "w"), indent=1, default=str)
    print(pd.DataFrame(out_r).set_index(["scenario", "h", "model"]).round(4).to_string())
    print(pd.DataFrame(out_w).set_index(["scenario", "h", "model"]).round(4).to_string())


def report() -> None:
    """docs/planning/16: D1-D4 (D5 deferred, see the plan's amendments)."""
    res = json.load(open(RES))
    models = pickle.load(open(OUT, "rb"))
    frozen = pickle.load(open(CM.MODELS, "rb"))

    def md(t):
        return "```\n" + t.to_string() + "\n```"

    lines = [
        "# Compartments discovered, not bolted on: results (D1-D4)",
        "",
        "Plan: `15_compartment_discovered_plan.md` (approved 21 Nov; see its amendments: the D2 "
        "comparator fix,",
        "D5 not run, by Knut's decision after D4). Script: `experiments/planning/compartment_discovered.py`;",
        "model: `src/sindy_wm/models/discovered.py`. Fitted on train; D3 on val, D4 on the "
        "counterfactual sets.",
        "",
        "## The refitted model",
        "",
        "```",
    ]
    for scen, (_, name) in SETS.items():
        lines += [f"{scen} ({name})", describe(models[scen]["discovered"], frozen[name]), ""]
    lines += [
        "```",
        "",
        "## D1: compartment groups kept per equation",
        "",
        md(pd.DataFrame(res["d1"]).set_index(["scenario", "equation"])),
        "",
        "## D2: CV R² of the alive and wounded rates (train)",
        "",
        md(pd.DataFrame(res["d2"]).set_index(["scenario", "side", "type"]).round(4)),
        "",
        "## D3: rollout errors on val (paired, bootstrap by battle)",
        "",
        md(pd.DataFrame(res["val"]).set_index(["scenario", "subset", "h", "model", "against"]).round(4)),
        "",
        "## D4: ranking on the counterfactual sets",
        "",
        md(pd.DataFrame(res["cf"]["table"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
        md(pd.DataFrame(res["cf"]["paired"]).set_index(["scenario", "h", "model"]).round(4)),
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(DOC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["fit", "d2", "val", "cf", "smoke", "closed", "report"])
    a = parser.parse_args()
    {"fit": fit, "d2": d2_only, "val": val, "cf": cf, "report": report}[a.what]()


if __name__ == "__main__":
    main()
