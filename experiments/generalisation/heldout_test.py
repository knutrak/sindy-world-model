"""WP6: the pre-registered held-out test of the frozen pipeline (docs/generalisation/03_preregistration.md).

Runs ONCE. Needs GENERALISATION_TEST=1. Builds the held-out per-type tables (`build_typed_states.py
--held-out`), fits the frozen pipeline (configs/generalisation_pipeline.yaml) on each scenario's train set,
scores it and the baselines on the held-out sets, refits the pipeline on each held-out set (structure,
physics), and writes docs/generalisation/04_test_results.md / .json with every expectation met / not met.

    GENERALISATION_TEST=1 uv run --extra explore experiments/generalisation/heldout_test.py
    ... heldout_test.py --dry-run    (val in place of held-out; checks the script runs; no held-out read)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "mixed"))
import evaluate_final as E  # noqa: E402
from common import DOCS, md  # noqa: E402
from evaluate import gates_terms, physics, team_rates  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.evaluation.harness_models import Lanchester  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import Layout, Pipeline, prepare  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402

VAL_H2 = {"marines": 0.0277, "marauders": 0.0203, "mixed": 0.0346}  # amended pipeline (03, amendment)
SHOW = [
    "health_1s",
    "health_2s",
    "health_4s",
    "margin_r2_2s",
    "margin_r2_4s",
    "alive_2s",
    "alive_4s",
    "distance_2s",
    "battle_health_nrmse",
    "winner_acc",
    "duration_err_s",
    "blowup_rate",
    "n_battles",
]


def table_path(exp: str, name: str) -> Path:
    return DATA_DIR / "processed" / f"{exp}_typed" / f"{name}.parquet"


def main() -> None:
    assert os.environ.get("GENERALISATION_TEST") == "1", "held-out test: set GENERALISATION_TEST=1"
    cfg = yaml.safe_load((ROOT / "configs" / "generalisation_pipeline.yaml").read_text())
    pc = cfg["pipeline"]
    kw = {
        "withdraw": pc["withdraw"],
        "alloc": pc["alloc"],
        "thr_health": pc["thr_health"],
        "thr_distance": pc["thr_distance"],
        "max_reach": pc["max_reach"],
        "movement_wd": pc.get("movement_wd", "gates"),
    }
    sc = cfg["scenarios"]
    # 1. held-out tables
    dry = "--dry-run" in sys.argv  # val in place of the held-out sets, uneven skipped, output to scratch
    if dry:
        sc = {k: {**v, "held_out": "val"} for k, v in sc.items() if k != "uneven"}
    for _name, s in sc.items():
        if not table_path(s["experiment"], s["held_out"]).exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_typed_states.py"),
                    s["experiment"],
                    *s["types"],
                    s["held_out"],
                    "--held-out",
                ],
                check=True,
            )
    # 2. models fitted on train (frozen config)
    fitted, team_models = {}, {}
    for name in ("marines", "marauders", "mixed"):
        s = sc[name]
        L = Layout(tuple(t.lower() for t in s["types"]))
        tr = pd.read_parquet(DATA_DIR / "processed" / f"{s['experiment']}_typed" / "rates_train_w9.parquet")
        fitted[name] = Pipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **kw).fit(tr)
        T = team_rates(tr, L)
        team_models[name] = {
            "persistence": None,
            "Lanchester": Lanchester().fit(H.FitData(states=None, rates=T)),
        }
        if len(s["types"]) == 2:
            team_models[name]["team-level A*-2k"] = AliveVariantA(law="reach").fit(
                H.FitData(states=None, rates=T)
            )
    if "uneven" in sc:
        fitted["uneven"], team_models["uneven"] = fitted["mixed"], team_models["mixed"]
    report, lines = (
        {},
        [
            "# Generalisation test: held-out results (run once)",
            "",
            "Pre-registration: `03_preregistration.md`. Script: "
            "`experiments/generalisation/heldout_test.py`.",
            "",
        ],
    )
    exp_rows = []
    for name, s in sc.items():
        m = fitted[name]
        L = m.L
        st = add_controls(pd.read_parquet(table_path(s["experiment"], s["held_out"])))
        Bh, Bb = L.batch(st), L.battle_batch(st)
        Th, Tb = Bh["team"], Bb["team"]
        Y = {"pipeline": (L.team_view(m.rollout(Bh)), L.team_view(m.rollout(Bb)))}
        for b, bm in team_models[name].items():
            Y[b] = (
                tuple(np.repeat(B["X0"][:, None], B["U"].shape[1] + 1, 1) for B in (Th, Tb))
                if bm is None
                else (bm.rollout(Th), bm.rollout(Tb))
            )
        tab = pd.DataFrame(
            {k: {**H.horizon_metrics(a, Th), **H.battle_metrics(b, Tb)} for k, (a, b) in Y.items()}
        ).T
        cis = {
            (b, h): E.ci_health_diff(Y[b][0], Y["pipeline"][0], Th, h=h)
            for b in team_models[name]
            for h in (2, 4)
        }
        horizon, hrows = E.horizon(Y["pipeline"][0], Th)
        # refit on the held-out set itself
        refit = Pipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **kw).fit(prepare(st, L))
        g = gates_terms(refit)
        ph = physics(refit)
        fire = ph[~ph.index.str.startswith("distance")]["fitted / game"]
        checked = ["distance close*adv", "distance close*hold*far", "distance close*hold*contact"]
        speed = ph.loc[
            [c for c in checked if c in ph.index], "fitted / game"
        ]  # amendment 2: no withdraw term
        struct_ok = (
            g["G1"]
            and g["G2"]
            and all(
                sum(1 for k, v in refit.coef[(sd, t)].items() if v) == len(L.types)
                for sd in ("blue", "red")
                for t in L.types
            )
        )
        e = {
            "E1 structure": bool(struct_ok),
            "E2 physics": bool(fire.between(0.75, 1.0).all() and speed.between(0.8, 1.2).all()),
            "E3 beats persistence and Lanchester at 2 and 4 s": all(
                cis[(b, h)][1] < 0 for b in ("persistence", "Lanchester") for h in (2, 4)
            ),
        }
        if len(L.types) == 2:
            e["E4 per-type beats team-level at 2 s"] = cis[("team-level A*-2k", 2)][1] < 0
        if name != "uneven":
            e["E5 reliable horizon >= 2 s"] = horizon is not None and horizon >= 2
            e["E6 health 2 s <= 1.2 x val"] = tab.loc["pipeline", "health_2s"] <= 1.2 * VAL_H2[name]
        for k, v in e.items():
            exp_rows.append({"scenario": name, "expectation": k, "met": bool(v)})
        report[name] = {
            "table": tab.to_dict(),
            "ci_model_minus_baseline": {f"{b} {h}s": v for (b, h), v in cis.items()},
            "horizon": horizon,
            "horizon_detail": hrows,
            "refit_terms": {k: float(v) for k, v in refit.coefficients().items()},
            "refit_gates": g,
            "refit_physics": ph.to_dict(),
            "expectations": e,
        }
        lines += [
            f"## {name} (`{s['experiment']}/{s['held_out']}`)",
            "",
            md(tab[SHOW].round(4)),
            "",
            "Paired 95% (pipeline − baseline), health error:",
            "",
            md(
                pd.Series(
                    {f"{b}, {h} s": f"[{v[0]:+.4f}, {v[1]:+.4f}]" for (b, h), v in cis.items()}
                ).to_frame("interval")
            ),
            "",
            f"Reliable horizon (G6): {horizon} s.",
            "",
            "Refit on this held-out set: against the game:",
            "",
            md(ph.round(3)),
            "",
            f"Refit term counts {g['G1 detail']}; sign failures {g['G2 detail'] or 'none'}.",
            "",
        ]
    X = pd.DataFrame(exp_rows)
    lines += [
        "## Expectations",
        "",
        md(X.set_index(["scenario", "expectation"])),
        "",
        f"Met: {int(X['met'].sum())} of {len(X)}.",
        "",
    ]
    out = Path(os.environ.get("DRY_OUT", "/tmp")) if dry else DOCS
    (out / "04_test_results.md").write_text("\n".join(lines))
    (out / "04_test_results.json").write_text(json.dumps(report, indent=1, default=E._plain))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
