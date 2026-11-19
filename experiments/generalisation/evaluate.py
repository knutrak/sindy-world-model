"""WP4: one evaluation table for every scenario on val, and WP5's decisions (docs/generalisation/PROGRESS.md).

For marines, marauders and mixed battles, the pipeline (`models.pipeline.Pipeline`) is fitted on train with
each option (withdraw: fitted / rule; mixed also target choice: closure / proportional) and scored on val
with the harness's metrics, against persistence, Lanchester square and (mixed) the team-level A*-2k. Also:
selected terms and signs (G1, G2), health coefficients against the game, a light G3a (10 battle-bootstrap
refits of the health equations), G4, G5 (paired 95% vs the baselines) and the reliable horizon (G6). The
decision rules fixed beforehand are applied at the end. Writes docs/generalisation/02_val_table.md (+ .json).

    uv run --extra explore experiments/generalisation/evaluate.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "mixed"))
import evaluate_final as E  # noqa: E402
from common import DOCS, SCENARIOS, layout, md, rows, states  # noqa: E402
from oracle import game_rate  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.evaluation.harness_models import Lanchester  # noqa: E402
from sindy_wm.models.intent_model import add_controls, r2, stlsq  # noqa: E402
from sindy_wm.models.pipeline import OTHER, SIDES, Pipeline, team_rows  # noqa: E402

METRICS = [
    "health_1s",
    "health_2s",
    "health_4s",
    "health_8s",
    "margin_r2_2s",
    "margin_r2_4s",
    "alive_4s",
    "distance_4s",
    "battle_health_nrmse",
    "winner_acc",
    "duration_err_s",
    "blowup_rate",
]
SPEEDS = {"close*adv": 6.3, "close*hold*contact": 6.3, "close*hold*far": 3.15, "close*wd": 3.15}
N_BOOT = 10


def team_rates(r: pd.DataFrame, L) -> pd.DataFrame:
    T = team_rows(r, L)
    ref = L.hp[L.types[0]]
    for s in SIDES:
        T[f"d_health_{s}"] = sum(T[f"d_health_{s}_{t}"] * L.hp[t] / ref for t in L.types)
        T[f"d_alive_{s}"] = sum(T[f"d_alive_{s}_{t}"] for t in L.types)
    return T


def health_fit(m: Pipeline, r: pd.DataFrame) -> dict:
    """Only the health equations (STLSQ), for the bootstrap."""
    out, wd = {}, r["withdraw"].to_numpy()
    for s in SIDES:
        o = OTHER[s]
        R_o = {a: r[f"reach_{o}_{a}"].to_numpy() for a in m.L.types}
        A_o = {a: r[f"alive_{o}_{a}"].to_numpy() for a in m.L.types}
        sig = m._sigma_measured(r, o)
        for t in m.L.types:
            T = m._terms(R_o, sig, A_o, wd, o, t)
            c = stlsq(np.column_stack(list(T.values())), r[f"d_health_{s}_{t}"].to_numpy(), m.thr_health)
            out |= {(f"health_{s}_{t}", k): v for k, v in zip(T, c, strict=True)}
    return out


def one_step_r2(m: Pipeline, r: pd.DataFrame) -> dict:
    out, wd = {}, r["withdraw"].to_numpy()
    for s in SIDES:
        o = OTHER[s]
        R_o = {a: r[f"reach_{o}_{a}"].to_numpy() for a in m.L.types}
        A_o = {a: r[f"alive_{o}_{a}"].to_numpy() for a in m.L.types}
        sig = m._sigma_measured(r, o)
        for t in m.L.types:
            T = m._terms(R_o, sig, A_o, wd, o, t)
            pred = sum(c * T[k] for k, c in m.coef[(s, t)].items() if c)
            out[f"{t} ({s})"] = r2(r[f"d_health_{s}_{t}"], pred)
    return out


def gates_terms(m: Pipeline) -> dict:
    bad, counts = [], {}
    for (s, t), c in m.coef.items():
        kept = {k: v for k, v in c.items() if v}
        counts[f"health_{s}_{t}"] = len(kept)
        for k, v in kept.items():
            if k.startswith("R[") and not k.endswith("*wd") and v >= 0:
                bad.append(f"{s} {t}: {k} {v:+.3f}")
            elif k.endswith("*wd"):
                main = abs(c.get(k[: -len("*wd")], 0.0))
                if not 0 < v < main * 1.05:
                    bad.append(f"{s} {t}: {k} {v:+.3f}")
            elif not k.startswith("R["):
                bad.append(f"{s} {t}: {k} {v:+.3f} (no rule)")
    counts["distance"] = int((m.dist_coef != 0).sum())
    g1 = (
        all(v <= 2 * len(m.L.types) for k, v in counts.items() if k.startswith("health"))
        and counts["distance"] <= 6
    )
    return {"G1": g1, "G1 detail": counts, "G2": not bad, "G2 detail": bad}


def physics(m: Pipeline) -> pd.DataFrame:
    rows_ = []
    for s in SIDES:
        o = OTHER[s]
        for a in m.L.types:
            for t in m.L.types:
                k = -m.coef[(s, t)].get(f"R[{a}]*s[{t}]", 0.0)
                g = game_rate(a, t)
                rows_.append({"term": f"{a} ({o}) → {t}", "fitted": k, "game": g, "fitted / game": k / g})
    for n, c in zip(m.dist_names, m.dist_coef, strict=True):
        if n in SPEEDS:
            rows_.append(
                {"term": f"distance {n}", "fitted": -c, "game": SPEEDS[n], "fitted / game": -c / SPEEDS[n]}
            )
    return pd.DataFrame(rows_).set_index("term")


def evaluate(scenario: str) -> dict:
    L = layout(scenario)
    types = SCENARIOS[scenario][1]
    tr, vr = rows(scenario, "train"), rows(scenario, "val")
    val = add_controls(states(scenario, "val"))
    Bh, Bb = L.batch(val), L.battle_batch(val)
    Th, Tb = Bh["team"], Bb["team"]
    options = [("wd fitted", {"withdraw": "fitted"}), ("wd rule", {"withdraw": "rule"})]
    if len(types) == 2:
        options = [
            (f"{w}, {al}", {**kw, "alloc": al2})
            for w, kw in options
            for al, al2 in (("closure", "closure"), ("proportional", "rule"))
        ]
    models = {name: Pipeline(types, **kw).fit(tr) for name, kw in options}
    Y = {n: (L.team_view(m.rollout(Bh)), L.team_view(m.rollout(Bb))) for n, m in models.items()}
    T = team_rates(tr, L)
    base = {"persistence": None, "Lanchester": Lanchester().fit(H.FitData(states=None, rates=T))}
    if len(types) == 2:
        base["team-level A*-2k"] = AliveVariantA(law="reach").fit(H.FitData(states=None, rates=T))
    for n, b in base.items():
        if b is None:
            Y[n] = tuple(np.repeat(B["X0"][:, None], B["U"].shape[1] + 1, 1) for B in (Th, Tb))
        else:
            Y[n] = (b.rollout(Th), b.rollout(Tb))
    table = pd.DataFrame(
        {n: {**H.horizon_metrics(Yh, Th), **H.battle_metrics(Yb, Tb)} for n, (Yh, Yb) in Y.items()}
    ).T
    detail = {}
    for n, m in models.items():
        d = gates_terms(m)
        ok4h, d4h = E.gate4(Y[n][0], Th)
        d["G4"], d["G4 detail"] = ok4h, d4h
        cis = {b: E.ci_health_diff(Y[b][0], Y[n][0], Th) for b in base}
        d["G5"], d["G5 detail"] = all(ci[1] < 0 for ci in cis.values()), cis
        d["G6 horizon (s)"], _ = E.horizon(Y[n][0], Th)
        d["one-step R² (val)"] = one_step_r2(m, vr)
        d["terms"] = {k: float(v) for k, v in m.coefficients().items()}
        detail[n] = d
    # light G3a on the health equations: 10 battle-bootstrap refits (first option's settings)
    boot = []
    for b in range(N_BOOT):
        eps = tr.index.get_level_values(0).unique().to_numpy()
        pick = np.random.default_rng(700 + b).choice(eps, len(eps), replace=True)
        rb = pd.concat([tr.loc[[e]] for e in pick])
        boot.append({n: health_fit(m, rb) for n, m in models.items()})
    for n, m in models.items():
        full = health_fit(m, tr)
        res = {}
        for key, v in full.items():
            if v:
                vals = np.array([bb[n].get(key, 0.0) for bb in boot])
                res[f"{key[0]}: {key[1]}"] = {
                    "kept": float((vals != 0).mean()),
                    "range": [float(vals.min()), float(vals.max())],
                }
        detail[n]["G3a (10 refits)"] = res
        detail[n]["G3a"] = all(
            r_["kept"] >= 0.9 and (r_["range"][0] > 0 or r_["range"][1] < 0) for r_ in res.values()
        )
    pickle.dump(models, open(H.CACHE / f"pipeline_{scenario}_val.pkl", "wb"))
    return {"table": table, "detail": detail, "physics": {n: physics(m) for n, m in models.items()}}


def main() -> None:
    results = {s: evaluate(s) for s in SCENARIOS}
    lines = [
        "# WP4: the pipeline in three scenarios on val (20 Nov)",
        "",
        "Script: `experiments/generalisation/evaluate.py`. Fitted on each scenario's train set, scored on",
        "its val set. Health in ten units of the scenario's first type.",
        "",
    ]
    ranks = []
    for s, res in results.items():
        t = res["table"]
        lines += [f"## {s}", "", md(t[METRICS].round(4)), ""]
        g = pd.DataFrame(
            {
                n: {k: v for k, v in d.items() if k in ("G1", "G2", "G3a", "G4", "G5", "G6 horizon (s)")}
                for n, d in res["detail"].items()
            }
        ).T
        lines += ["Gates (G3a light: 10 refits; G5 vs every baseline):", "", md(g), ""]
        for n, d in res["detail"].items():
            lines += [
                f"- **{n}**: one-step R² "
                f"{', '.join(f'{k} {v:.3f}' for k, v in d['one-step R² (val)'].items())}; "
                f"term counts {d['G1 detail']}; sign failures {d['G2 detail'] or 'none'}"
            ]
        first = next(iter(res["physics"]))
        lines += ["", f"Coefficients against the game ({first}):", "", md(res["physics"][first].round(3)), ""]
        opts = [n for n in res["detail"]]
        for metric in ("health_4s", "battle_health_nrmse"):
            for w in ("wd fitted", "wd rule"):
                cand = [n for n in opts if n.startswith(w)]
                best_alloc = min(cand, key=lambda n: t.loc[n, metric])
                ranks.append(
                    {"scenario": s, "metric": metric, "option": w, "value": float(t.loc[best_alloc, metric])}
                )
    R = pd.DataFrame(ranks)
    R["rank"] = R.groupby(["scenario", "metric"])["value"].rank(method="min")
    mean_rank = R.groupby("option")["rank"].mean()
    withdraw = "wd fitted" if mean_rank["wd fitted"] < mean_rank["wd rule"] else "wd rule"
    mixed = results["mixed"]["table"]
    pre = withdraw
    clos, prop = f"{pre}, closure", f"{pre}, proportional"
    alloc = (
        "closure"
        if (
            mixed.loc[clos, "health_4s"] < mixed.loc[prop, "health_4s"]
            and mixed.loc[clos, "battle_health_nrmse"] < mixed.loc[prop, "battle_health_nrmse"]
        )
        else "proportional"
    )
    lines += [
        "## WP5: applying the decision rules",
        "",
        md(R.round(4)),
        "",
        f"Mean rank: {mean_rank.round(3).to_dict()} → **withdraw: {withdraw}**.",
        f"Mixed, with {withdraw}: closure health 4 s {mixed.loc[clos, 'health_4s']:.4f} / nRMSE "
        f"{mixed.loc[clos, 'battle_health_nrmse']:.4f}; proportional {mixed.loc[prop, 'health_4s']:.4f} / "
        f"{mixed.loc[prop, 'battle_health_nrmse']:.4f} → **target choice: {alloc}**.",
        "",
    ]
    (DOCS / "02_val_table.md").write_text("\n".join(lines))
    (DOCS / "02_val_table.json").write_text(
        json.dumps(
            {s: {"table": r["table"].to_dict(), "detail": r["detail"]} for s, r in results.items()}
            | {"decision": {"withdraw": withdraw, "alloc": alloc, "mean_rank": mean_rank.to_dict()}},
            indent=1,
            default=E._plain,
        )
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
