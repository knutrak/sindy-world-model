"""Counterfactual ranking counted over distinct choices (exploratory, docs/planning/28_distinct_regret.md).

In contact, hold and advance with the same target play identical battles (99%, `21` section 4), so a pick of
hold/focus and one of advance/focus are the same choice there. Here, in states where Blue is in contact at
the branch,
the twins are merged (a pick counts by its target rule; withdraw stays withdraw) before comparing picks
between
models; regret is unchanged by construction (twins have the same outcome) and is reported as a check.

    uv run --extra explore experiments/planning/distinct_regret.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compartment_discovered as D  # noqa: E402
import compartment_model as CM  # noqa: E402

from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "28_distinct_regret.md"
MERGE = np.array(
    [0, 1, 0, 1, 2]
)  # advance/nearest, advance/focus, hold/nearest, hold/focus, withdraw -> 3 choices


def main() -> None:
    rows = []
    for scen, (exp, _) in CM.SETS.items():
        models = D.variants(scen)
        L = models["frozen"].L
        states, _ = CM.from_branch(CM.load_cf(exp), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = CM.attach(L.batch(states, horizon=64, every=10**9, history=0, min_future=1), states, L)
            Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
        TY = B["team"]["Y"]
        g = states.groupby(level=0, sort=False)
        seed = g["seed"].first().loc[B["ep"]].to_numpy()
        branch = g["branch"].first().loc[B["ep"]].to_numpy()
        reach = g[[c for c in states if c.startswith("reach_blue_")]].first().sum(1).loc[B["ep"]].to_numpy()

        def wide(v, seed=seed, branch=branch):
            f = pd.DataFrame({"s": seed, "b": branch, "v": v})
            return f.pivot(index="s", columns="b", values="v")

        contact = wide(reach).iloc[:, 0].to_numpy() > 0
        for h in (1.0, 2.0, 4.0):
            T = wide(CM.margin_change(TY, TY, h)).to_numpy()
            picks = {n: wide(CM.margin_change(Y, TY, h)).to_numpy().argmax(1) for n, Y in Ys.items()}
            idx = np.arange(len(T))
            for n in ("bolted-on", "discovered"):
                raw = picks[n] != picks["frozen"]
                merged = np.where(contact, MERGE[picks[n]] != MERGE[picks["frozen"]], raw)
                rows.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "model": n,
                        "states": len(T),
                        "in contact": contact.mean(),
                        "picks differ (five candidates)": raw.mean(),
                        "picks differ (distinct choices)": merged.mean(),
                        "regret, frozen": (T.max(1) - T[idx, picks["frozen"]]).mean(),
                        f"regret, {n}": (T.max(1) - T[idx, picks[n]]).mean(),
                    }
                )
    t = pd.DataFrame(rows).set_index(["scenario", "h", "model"])
    lines = [
        "# Counterfactual picks over distinct choices",
        "",
        "Script: `experiments/planning/distinct_regret.py` (see its docstring). "
        "Counterfactual sets of S1 and S2.",
        "",
        "```",
        t.round(4).to_string(),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
