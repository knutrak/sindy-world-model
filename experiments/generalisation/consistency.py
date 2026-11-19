"""WP2: does the unified pipeline, run on marines alone, reproduce A*-2k?

A*-2k (`final_Astar2k.pkl`, fitted on exp06 train with `AliveVariantA(law="reach")`) against the pipeline
(`models.pipeline.Pipeline(("MARINE",))`, same train battles, per-type tables with the reach capped at 6,
which changes nothing for marines) with both withdraw options. Compared: coefficients, and val metrics on the
same start points. Pass: coefficients within 2% and val metrics within 0.002, or every difference explained.

    uv run --extra explore experiments/generalisation/consistency.py  (docs/generalisation/01_consistency.md)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS, layout, md, rows, states  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import Pipeline  # noqa: E402

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


def main() -> None:
    L = layout("marines")
    tr = rows("marines", "train")
    val = add_controls(states("marines", "val"))
    Bh, Bb = L.batch(val), L.battle_batch(val)
    # the same start points as the harness?
    Hh = H.horizon_batch(H.load_states("val"))
    same = len(Hh["X0"]) == len(Bh["X0"]) and np.allclose(Hh["X0"], Bh["team"]["X0"], atol=1e-9)
    astar = pickle.load(open(H.CACHE / "final_Astar2k.pkl", "rb"))
    models = {
        name: Pipeline(("MARINE",), withdraw=w).fit(tr)
        for name, w in (("pipeline, wd fitted", "fitted"), ("pipeline, wd rule", "rule"))
    }
    res = {
        "A*-2k": {
            **H.horizon_metrics(astar.rollout(Bh["team"]), Bh["team"]),
            **H.battle_metrics(astar.rollout(Bb["team"]), Bb["team"]),
        }
    }
    for name, m in models.items():
        res[name] = {
            **H.horizon_metrics(L.team_view(m.rollout(Bh)), Bh["team"]),
            **H.battle_metrics(L.team_view(m.rollout(Bb)), Bb["team"]),
        }
    R = pd.DataFrame(res).T[METRICS]
    # coefficients side by side
    a = astar.coefficients()
    rows_ = {
        "k_B (Blue's loss per Red unit in reach)": (a.get("health_blue: Rr"), "blue", "R[marine]*s[marine]"),
        "k_R (Red's loss per Blue unit in reach)": (a.get("health_red: Rb"), "red", "R[marine]*s[marine]"),
        "k_wd (Red's loss, Blue withdrawing)": (a.get("health_red: Rb*wd"), "red", "R[marine]*s[marine]*wd"),
    }
    C = {}
    for label, (va, side, term) in rows_.items():
        C[label] = {"A*-2k": va} | {n: m.coef[(side, "marine")].get(term, np.nan) for n, m in models.items()}
    for side, key in (("blue", "alive_blue: 1"), ("red", "alive_red: 1")):
        C[f"p_{side[0].upper()}"] = {"A*-2k": a.get(key)} | {
            n: m.p[(side, "marine")][0] for n, m in models.items()
        }
    C["p_focus (Red)"] = {"A*-2k": a.get("alive_red: focus")} | {
        n: m.p[("red", "marine")][1] for n, m in models.items()
    }
    for k, v in a.items():
        if k.startswith("distance: "):
            C[k] = {"A*-2k": v} | {n: m.coefficients().get(k, np.nan) for n, m in models.items()}
    extra = {
        n: {
            k: v
            for k, v in m.coefficients().items()
            if k.startswith("health") and "R[marine]*s[marine]" not in k
        }
        for n, m in models.items()
    }
    C = pd.DataFrame(C).T
    lines = [
        "# WP2: the unified pipeline on marines against A*-2k (20 Nov)",
        "",
        "Script: `experiments/generalisation/consistency.py`. Both fitted on exp06 train (6000 battles),",
        f"scored on exp06 val. Same start points as the harness: {same}.",
        "",
        "## Coefficients",
        "",
        md(C.round(4)),
        "",
        f"Other health terms kept by the pipeline: {extra}",
        "",
        "## Val metrics (team view, health in ten marines' hp)",
        "",
        md(R.round(4)),
        "",
        "## Differences, relative to A*-2k",
        "",
        md((C.drop(columns="A*-2k").div(C["A*-2k"], axis=0) - 1).round(4)),
        "",
        md((R.drop(index="A*-2k") - R.loc["A*-2k"]).round(4)),
        "",
    ]
    (DOCS / "01_consistency.md").write_text("\n".join(lines))
    pickle.dump(models, open(H.CACHE / "pipeline_marines.pkl", "wb"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
