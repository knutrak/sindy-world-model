"""Planning at team level only (approved plan: docs/planning/42_team_level_plan.md).

uv run experiments/planning/team42.py smoke              2 battles per new policy and scenario, timed
uv run experiments/planning/team42.py run S2 --n 400     the three new policies on 34's seeds
uv run experiments/planning/team42.py analyse            -> docs/planning/43_team_level.md
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

NEW = ["musigma5:h3", "oracle5-h3", "oracle5-h8"]
MAIN = ("S1", "S2", "S5")
DOC = PROJECT_ROOT / "docs" / "planning" / "43_team_level.md"


def analyse():
    rng = np.random.default_rng(0)
    rows, wins = [], []
    for scen in C34.FIRST:
        d = pd.read_csv(C34.A.OUT / f"closed34_{scen}.csv")
        keep = [C34.BEST_FIXED[scen], "model:h3", *NEW]
        wins.append(d[d.policy.isin(keep)].groupby("policy")["won"].mean().rename(scen))
        best = C34.BEST_FIXED[scen]
        res = {}
        for label, a, b in (
            ("T1", "musigma5:h3", best),
            ("T2", "musigma5:h3", "model:h3"),
            ("T3", "oracle5-h3", best),
            ("T4", "oracle5-h8", best),
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
        rows[-1]["share of 3 s ceiling (T1 / T3)"] = res["T1"] / res["T3"] if res["T3"] > 0 else np.nan
    t = pd.DataFrame(rows)
    t1 = t[(t.rule == "T1") & t.scenario.isin(MAIN)]["read-out"]
    holds = bool((t1 == "beats").any() and not (t1 == "loses").any())
    pd.set_option("display.width", 250)
    lines = [
        "# Planning at team level only: results",
        "",
        "Plan: `42_team_level_plan.md` (approved 23 Nov). Script: `experiments/planning/team42.py`.",
        "Five team-level candidates; scripted Red; `34`'s seeds, 400 per scenario, paired.",
        "",
        "## Wins",
        "",
        "```",
        pd.concat(wins, axis=1).round(3).to_string(),
        "```",
        "",
        "## Paired comparisons",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**Statement (T1 beats in at least one of S1, S2, S5 and loses in none): "
        f"{'holds' if holds else 'does not hold'}.**",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["smoke", "run", "analyse"])
    p.add_argument("scenario", nargs="?")
    p.add_argument("--n", type=int, default=400)
    a = p.parse_args()
    if a.what == "analyse":
        return analyse()
    if a.what == "smoke":
        for scen in ("S2", "S4"):
            PB.run(scen, NEW, 2, 6, C34.SMOKE, "_smoke42", prefix="closed34_")
            d = pd.read_csv(C34.A.OUT / f"closed34_{scen}_smoke42.csv")
            print(d.groupby("policy")["wall_s"].mean().round(2).to_string(), flush=True)
        return
    PB.run(a.scenario, NEW, a.n, 8, C34.FIRST[a.scenario], prefix="closed34_")


if __name__ == "__main__":
    main()
