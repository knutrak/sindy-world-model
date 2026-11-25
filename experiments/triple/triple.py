# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Three unit types (docs/triple/00_plan.md): fit the frozen pipeline on exp12 train, score it, check structure,
physics and the shared coefficients.

    uv run --extra explore experiments/triple/triple.py val       -> docs/triple/01_val.md (fit on train, score val)
    TRIPLE_TEST=1 uv run --extra explore experiments/triple/triple.py test     once -> docs/triple/02_test.md
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "experiments/mixed", "experiments/generalisation"):
    sys.path.insert(0, str(ROOT / p))
import evaluate_final as E  # noqa: E402
from common import md  # noqa: E402
from evaluate import gates_terms, one_step_r2, team_rates  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.evaluation.harness_models import Lanchester  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import OTHER, SIDES, Layout, Pipeline, prepare  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402
from sindy_wm.states.typed import unit_stats  # noqa: E402

PROC = DATA_DIR / "processed"
DOCS = ROOT / "docs" / "triple"
TYPES = ("MARINE", "MARAUDER", "STALKER")
BONUS = {"MARAUDER": 10.0, "STALKER": 5.0}
ARMORED = {"MARAUDER", "STALKER"}
SHOW = ["health_1s", "health_2s", "health_4s", "margin_r2_2s", "margin_r2_4s", "alive_2s", "alive_4s"]
SHOW += ["distance_2s", "battle_health_nrmse", "winner_acc", "duration_err_s", "blowup_rate", "n_battles"]


def config() -> dict:
    pc = yaml.safe_load((ROOT / "configs" / "generalisation_pipeline.yaml").read_text())["pipeline"]
    return {k: pc[k] for k in ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd")}


def game_rate(a: str, t: str) -> float:
    """Full units of t removed per second by one firing unit of a. Armor does not apply to shield damage."""
    a, t = a.upper(), t.upper()
    sa, st = unit_stats(a), unit_stats(t)
    dmg = sa["damage"] + (BONUS.get(a, 0.0) if t in ARMORED else 0.0)
    shots = st["shield"] / dmg + st["hp"] / (dmg - st["armor"])
    return 1 / (shots * np.ceil(sa["cooldown"] * 16) / 16)


def physics(m: Pipeline) -> pd.DataFrame:
    out = []
    for s in SIDES:
        for a in m.L.types:
            for t in m.L.types:
                k, g = -m.coef[(s, t)].get(f"R[{a}]*s[{t}]", 0.0), game_rate(a, t)
                out.append(
                    {"term": f"{a} ({OTHER[s]}) → {t}", "fitted": k, "game": g, "fitted / game": k / g}
                )
    return pd.DataFrame(out).set_index("term")


def table(exp: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{exp}_typed" / f"{name}.parquet")


def rates(exp: str, name: str, L: Layout) -> pd.DataFrame:
    path = PROC / f"{exp}_typed" / f"rates_{name}_w9.parquet"
    if not path.exists():
        prepare(table(exp, name), L).to_parquet(path)
    return pd.read_parquet(path)


def with_missing_types(t: pd.DataFrame, types: tuple[str, ...]) -> pd.DataFrame:
    """Add zero columns for types a scenario does not have, so the triple model can read its states."""
    t = t.copy()
    for s in SIDES:
        for a in types:
            for c in (f"health_{s}_{a}", f"alive_{s}_{a}", f"reach_{s}_{a}", f"front_{s}_{a}"):
                if c not in t:
                    t[c] = 0.0
    return t


def equations(m: Pipeline) -> str:
    from frozen_models import equations as eq

    return eq(m)


def score(m, st, baselines) -> tuple[pd.DataFrame, dict, tuple, dict]:
    L = m.L
    Bh, Bb = L.batch(st), L.battle_batch(st)
    Th, Tb = Bh["team"], Bb["team"]
    Y = {"pipeline": (L.team_view(m.rollout(Bh)), L.team_view(m.rollout(Bb)))}
    for b, bm in baselines.items():
        Y[b] = (
            tuple(np.repeat(B["X0"][:, None], B["U"].shape[1] + 1, 1) for B in (Th, Tb))
            if bm is None
            else (bm.rollout(Th), bm.rollout(Tb))
        )
    tab = pd.DataFrame(
        {k: {**H.horizon_metrics(a, Th), **H.battle_metrics(b, Tb)} for k, (a, b) in Y.items()}
    ).T
    ci = {(b, h): E.ci_health_diff(Y[b][0], Y["pipeline"][0], Th, h=h) for b in baselines for h in (2, 4)}
    return tab, ci, E.horizon(Y["pipeline"][0], Th), Th


def main(split: str) -> None:
    if split == "test":
        assert os.environ.get("TRIPLE_TEST") == "1", "held-out: set TRIPLE_TEST=1"
        assert not (DOCS / "02_test.md").exists(), "the held-out test has already been run"
    kw, L = config(), Layout(tuple(t.lower() for t in TYPES))
    tr = rates("exp12_triple", "train", L)
    m = Pipeline(TYPES, withdraw_radius=8.0, **kw).fit(tr)
    T = team_rates(tr, L)
    base = {
        "persistence": None,
        "Lanchester": Lanchester().fit(H.FitData(states=None, rates=T)),
        "team-level A*-2k": AliveVariantA(law="reach").fit(H.FitData(states=None, rates=T)),
    }
    st = add_controls(table("exp12_triple", split))
    tab, ci, (hor, _), _ = score(m, st, base)
    # structure and physics: the train fit on val, a refit on the held-out set itself on test (plan, clarification)
    judged = m if split == "val" else Pipeline(TYPES, withdraw_radius=8.0, **kw).fit(prepare(st, L))
    ph, g = physics(judged), gates_terms(judged)
    frozen = pickle.load(open(PROC / "final_exploration" / "generalisation_frozen.pkl", "rb"))["mixed"]
    shared = []
    for s in SIDES:
        for a in frozen.L.types:
            for t in frozen.L.types:
                n = f"R[{a}]*s[{t}]"
                k0, k1 = frozen.coef[(s, t)][n], judged.coef[(s, t)][n]
                shared.append(
                    {"term": f"{a} ({OTHER[s]}) → {t}", "mixed": -k0, "triple": -k1, "ratio": k1 / k0}
                )
    shared = pd.DataFrame(shared).set_index("term")
    val2 = None
    if split == "test":
        val2 = json.loads((DOCS / "01_val.json").read_text())["health_2s"]
    exact = all(
        sorted(k for k, v in judged.coef[(s, t)].items() if v) == sorted(f"R[{a}]*s[{t}]" for a in L.types)
        for s in SIDES
        for t in L.types
    )
    exps = {
        "T1 structure (3 own firepower terms each, no others; distance <= 6)": bool(
            exact and g["G2"] and g["G1 detail"]["distance"] <= 6
        ),
        "T2 physics (18 terms 0.75-1.0 x game)": bool(ph["fitted / game"].between(0.75, 1.0).all()),
        "T3 beats persistence and Lanchester at 2 and 4 s": all(
            ci[(b, h)][1] < 0 for b in ("persistence", "Lanchester") for h in (2, 4)
        ),
        "T4 beats team-level A*-2k at 2 s": ci[("team-level A*-2k", 2)][1] < 0,
        "T5 reliable horizon >= 2 s": hor is not None and hor >= 2,
        "T6 health 2 s <= 1.2 x val": None
        if val2 is None
        else tab.loc["pipeline", "health_2s"] <= 1.2 * val2,
        "T7 shared terms within 10% of frozen mixed": bool(shared["ratio"].between(0.9, 1.1).all()),
    }
    lines = [
        f"# Three unit types: {'held-out test (run once)' if split == 'test' else 'val'}",
        "",
        "Plan: `00_plan.md`. Script: `experiments/triple/triple.py`. Fitted on exp12 train (1500 battles).",
        "",
        "## Equations",
        "",
        "```",
        equations(m),
        "```",
        "",
        "## Prediction (health in ten marines' hp, mean per side)",
        "",
        md(tab[SHOW].round(4)),
        "",
        "Paired 95% (pipeline − baseline), health error: "
        + "; ".join(f"{b} {h} s [{v[0]:+.4f}, {v[1]:+.4f}]" for (b, h), v in ci.items()),
        "",
        f"Reliable horizon (G6): {hor} s. One-step R²: "
        + ", ".join(f"{k} {v:.2f}" for k, v in one_step_r2(m, rates("exp12_triple", "val", L)).items()),
        "",
        f"## Firepower against the game ({'fitted on train' if split == 'val' else 'refitted on test'})",
        "",
        md(ph.round(3)),
        "",
        "## Against the frozen mixed model (marines + marauders)",
        "",
        md(shared.round(3)),
        "",
        "Reach closure R²: " + ", ".join(f"{s} {t} {v:.2f}" for (s, t), v in m.share_r2.items()),
        "",
        f"Terms: {g['G1 detail']}; terms without a rule or with a wrong sign: {g['G2 detail'] or 'none'}",
        "",
        "## Expectations",
        "",
        md(pd.Series(exps).to_frame("met")),
        "",
    ]
    if split == "val":
        (DOCS / "01_val.json").write_text(json.dumps({"health_2s": float(tab.loc["pipeline", "health_2s"])}))
        pickle.dump(m, open(PROC / "exp12_triple_typed" / "triple_model.pkl", "wb"))
    (DOCS / ("02_test.md" if split == "test" else "01_val.md")).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
