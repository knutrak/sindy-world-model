"""Focus cost on the μ, σ planner, team level, fresh seeds (plan: docs/planning/46_focus_cost_plan.md).

uv run experiments/planning/fc46.py run S2       four policies, 400 fresh seeds
uv run experiments/planning/fc46.py analyse      -> docs/planning/47_focus_cost.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed34 as C34  # noqa: E402
import phase_b as PB  # noqa: E402

from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

FIRST = {f"S{i + 1}": 23_000_000 + 100_000 * i for i in range(5)}
MAIN = ("S1", "S2", "S5")
DOC = PROJECT_ROOT / "docs" / "planning" / "47_focus_cost.md"


def policies(scen):
    return ["musigma5-focuscost:h3", "musigma5:h3", C34.BEST_FIXED[scen], "oracle5-h3"]


def analyse():
    rng = np.random.default_rng(0)
    rows, wins = [], []
    for scen in FIRST:
        p = C34.A.OUT / f"fc46_{scen}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        wins.append(d.groupby("policy")["won"].mean().rename(scen))
        best = C34.BEST_FIXED[scen]
        res = {}
        for label, a, b in (
            ("F1", "musigma5-focuscost:h3", "musigma5:h3"),
            ("F2", "musigma5-focuscost:h3", best),
            ("ceiling", "oracle5-h3", best),
        ):
            mean, lo, hi, read, n = C34.paired(d, a, b, rng)
            res[label] = mean
            rows.append(
                {
                    "scenario": scen,
                    "flag": "" if scen in MAIN else "flagged",
                    "rule": label,
                    "policy": a,
                    "against": b.removeprefix("fixed:"),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": read,
                    "n": n,
                }
            )
        rows.append(
            {
                "scenario": scen,
                "rule": "F3 share of ceiling",
                "difference": res["F2"] / res["ceiling"] if res["ceiling"] > 0 else np.nan,
            }
        )
        pk = d[d.policy.str.startswith("musigma5")]
        cols = [c for c in pk if c.startswith("pick_")]
        print(scen, pk.groupby("policy")[cols].mean().round(3).to_string())
    t = pd.DataFrame(rows)
    f1 = t[(t.rule == "F1") & t.scenario.isin(MAIN)]["read-out"]
    holds = bool((f1 == "beats").any() and not (f1 == "loses").any())
    pd.set_option("display.width", 250)
    lines = [
        "# Focus cost on the μ, σ planner, team level: results",
        "",
        "Plan: `46_focus_cost_plan.md`. Script: `experiments/planning/fc46.py`. Fresh seeds 23 x00 000 +.",
        "",
        "## Wins",
        "",
        "```",
        pd.concat(wins, axis=1).round(3).to_string(),
        "```",
        "",
        "## Comparisons",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**Statement (F1 beats in at least one of S1, S2, S5 and loses in none): "
        f"{'holds' if holds else 'does not hold'}.**",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "analyse"])
    p.add_argument("scenario", nargs="?")
    a = p.parse_args()
    if a.what == "analyse":
        return analyse()
    PB.run(a.scenario, policies(a.scenario), 400, 8, FIRST[a.scenario], prefix="fc46_")


if __name__ == "__main__":
    main()
