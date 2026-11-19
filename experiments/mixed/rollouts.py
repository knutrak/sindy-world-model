"""Step 3: closures and rollouts for mixed battles. Does the per-type state pay off in rollouts?

Models (all fitted on exp08 `train`, scored on `val`):
  team A*-2k          A*-2k on team totals (health in ten marines' hp, alive and reach summed): the
                      mixed army treated as one unit type. Its distance equation is reused below.
  typed (closure)     `models.typed_model.TypedA`: per-type health, alive and reach closures, the
                      logistic allocation closure
  typed (rule)        the same with the proportional allocation rule
  persistence

Team-level metrics are the harness's (health in ten marines' hp, margin R², alive in units, distance,
whole-battle nRMSE, winner), so they compare with the marine-only numbers. Per-type metrics: health
error per type (ten full units of the type), alive error per type (units), and the error of the
marauders' share of the alive units. Writes docs/mixed/04_rollouts.md.

    uv run --extra explore experiments/mixed/rollouts.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.models.intent_model import DT, add_controls  # noqa: E402
from sindy_wm.models.typed_model import (  # noqa: E402
    SIDES,
    TYPES,
    TypedA,
    idx,
    team_battle_batch,
    team_horizon_batch,
    team_table,
    team_view,
    typed_batch,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

P = DATA_DIR / "processed" / "exp08_mixed"
DOC = PROJECT_ROOT / "docs" / "mixed" / "04_rollouts.md"


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def team_rates(r: pd.DataFrame) -> pd.DataFrame:
    T = team_table(r)
    for s in SIDES:
        T[f"d_health_{s}"] = T[f"d_hp_{s}"] / (10 * 45.0)
    return T


def per_type_metrics(Y: np.ndarray, B: dict) -> dict:
    out = {}
    for h in (1, 2, 4, 8):
        n = round(h / DT)
        ok = ~np.isnan(B["Y"][:, n, 0])
        Yt, Yp = B["Y"][ok, n], Y[ok, n]
        for t in TYPES:
            hcols = [idx("health", s, t) for s in SIDES]
            acols = [idx("alive", s, t) for s in SIDES]
            out[f"health_{t}_{h}s"] = float(np.abs(Yp[:, hcols] - Yt[:, hcols]).mean())
            out[f"alive_{t}_{h}s"] = float(10 * np.abs(Yp[:, acols] - Yt[:, acols]).mean())
        errs = []
        for s in SIDES:
            m_t, M_t = Yt[:, idx("alive", s, "marine")], Yt[:, idx("alive", s, "marauder")]
            m_p, M_p = Yp[:, idx("alive", s, "marine")], Yp[:, idx("alive", s, "marauder")]
            sh_t = np.divide(M_t, m_t + M_t, out=np.full(len(Yt), np.nan), where=(m_t + M_t) > 1e-9)
            sh_p = np.divide(M_p, m_p + M_p, out=np.full(len(Yp), np.nan), where=(m_p + M_p) > 1e-9)
            errs.append(np.abs(sh_p - sh_t))
        out[f"composition_{h}s"] = float(np.nanmean(np.concatenate(errs)))
    return out


def main() -> None:
    global P, DOC
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reach", type=float, default=None)
    max_reach = parser.parse_args().max_reach
    suffix = oracle.configure(max_reach)
    P, DOC = oracle.P, DOC.with_name(DOC.stem + suffix + ".md")
    tr = oracle.rates("train")
    val = add_controls(pd.read_parquet(P / "val.parquet"))
    Bh = typed_batch(val)
    Th = team_horizon_batch(Bh)
    Bb, Tb = team_battle_batch(val)

    team_data = H.FitData(states=None, rates=team_rates(tr))
    team = AliveVariantA(law="reach").fit(team_data)
    models = {"team A*-2k": team}
    for alloc in ("closure", "rule"):
        models[f"typed ({alloc})"] = TypedA(team, alloc=alloc, max_reach=max_reach).fit(tr)
    for alloc in ("closure", "rule"):  # Blue does not fire while withdrawing (the intent's rule, not fitted)
        m = TypedA(team, alloc=alloc, max_reach=max_reach, withdraw_rule=True)
        m.__dict__.update(
            {k: v for k, v in models[f"typed ({alloc})"].__dict__.items() if k != "withdraw_rule"}
        )
        models[f"typed ({alloc}) + wd rule"] = m
    pickle.dump(models, open(H.CACHE / f"mixed_models{suffix}.pkl", "wb"))

    rows, typed_rows, team_Y = {}, {}, {}
    pers_h = np.repeat(Th["X0"][:, None], Th["U"].shape[1] + 1, 1)
    pers_b = np.repeat(Tb["X0"][:, None], Tb["U"].shape[1] + 1, 1)
    rows["persistence"] = {**H.horizon_metrics(pers_h, Th), **H.battle_metrics(pers_b, Tb)}
    typed_rows["persistence"] = per_type_metrics(np.repeat(Bh["X0"][:, None], Bh["U"].shape[1] + 1, 1), Bh)
    for name, m in models.items():
        if name.startswith("typed"):
            Yh, Yb = m.rollout(Bh), m.rollout(Bb)
            typed_rows[name] = per_type_metrics(Yh, Bh)
            Yh, Yb = team_view(Yh), team_view(Yb)
        else:
            Yh, Yb = m.rollout(Th), m.rollout(Tb)
        rows[name] = {**H.horizon_metrics(Yh, Th), **H.battle_metrics(Yb, Tb)}
        team_Y[name] = Yh
        print(name, f"health 4 s {rows[name]['health_4s']:.4f}", flush=True)
    lab = __import__("sindy_wm.evaluation.harness_models", fromlist=["lab"]).lab()
    paired = {}
    for name in [n for n in models if n.startswith("typed")]:
        pb = lab.paired_bootstrap(team_Y["team A*-2k"], team_Y[name], Th, h=4, n_boot=1000)
        paired[name] = {k: f"[{v[0]:+.4f}, {v[1]:+.4f}]" for k, v in pb.items() if k.endswith("b-a")}
    team_cols = [
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
    T = pd.DataFrame(rows).T[team_cols]
    Ty = pd.DataFrame(typed_rows).T
    typed = models["typed (closure)"]
    clos = pd.DataFrame(
        {
            "reach closure R² (count)": {f"{s} {a}": typed.share_r2[(s, a)] for s in SIDES for a in TYPES},
            "lam, beta, g": {
                f"{s} {a}": np.round(typed.share[(s, a)], 3).tolist() for s in SIDES for a in TYPES
            },
            "allocation b0, b1, b2 (as attacker)": {
                f"{s} {a}": np.round(typed.alloc_b[(s, a)], 3).tolist() for s in SIDES for a in TYPES
            },
        }
    )
    coef = pd.Series(typed.coefficients()).round(4).to_frame("typed (closure)")
    lines = [
        "# Step 3: per-type closures and rollouts (exp08, 19 Nov)",
        "",
        "Script: `experiments/mixed/rollouts.py`; model: `src/sindy_wm/models/typed_model.py`. Fitted on",
        "exp08 `train`, scored on `val` (1000 battles). test3 not read.",
        "",
        "## Team-level metrics (harness: health in ten marines' hp)",
        "",
        md(T.round(3)),
        "",
        "Paired 95% bootstrap (by battle, 1000 resamples), model − team A*-2k, at 4 s:",
        "",
        md(pd.DataFrame(paired).T),
        "",
        "## Per-type metrics",
        "",
        "Health per type in ten full units of the type; alive in units; composition = |error| of the",
        "marauders' share of each side's alive units.",
        "",
        md(Ty[[c for c in Ty if c.endswith(("_2s", "_4s"))]].round(3)),
        "",
        "## Closures (typed model)",
        "",
        md(clos),
        "",
        "## Typed model coefficients",
        "",
        md(coef),
        "",
        "## Team model (A*-2k on team totals)",
        "",
        md(pd.Series(team.coefficients()).round(4).to_frame("team A*-2k")),
        "",
    ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
