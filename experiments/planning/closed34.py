"""Closed-loop test of the μ, σ planner (approved plan: docs/planning/34_mu_sigma_closed_loop_plan.md).

uv run experiments/planning/closed34.py smoke            5 battles per scenario and policy (19 950 000 +)
uv run experiments/planning/closed34.py run S2 --n 400   main run (fresh seeds 21 x00 000 +)
uv run experiments/planning/closed34.py analyse          -> docs/planning/35_mu_sigma_closed_loop.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402
import phase_b as PB  # noqa: E402
from analyse_phase_b import BEST_FIXED  # noqa: E402

from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

FIRST = {f"S{i + 1}": 21_000_000 + 100_000 * i for i in range(5)}
SMOKE = 19_950_000
BEST_KITE = {
    "S1": "fixed:kite/nearest",
    "S2": "fixed:kite/nearest",
    "S3": "fixed:kite/focus",
    "S4": "fixed:kite/nearest",
    "S5": "fixed:kite/nearest",
}
MAIN = ("S2", "S4", "S5")
DOC = PROJECT_ROOT / "docs" / "planning" / "35_mu_sigma_closed_loop.md"
N_BOOT = 4000


def policies(scen):
    return ["musigma:h3", "model:h3", "fixed:kite/nearest", "fixed:kite/focus", BEST_FIXED[scen]]


def paired(d, a, b, rng):
    W = d.pivot_table(index="seed", columns="policy", values="won", aggfunc="first").astype(float)
    x = (W[a] - W[b]).dropna().to_numpy()
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    read = "beats" if lo > 0 else ("loses" if hi < 0 else "ties")
    return x.mean(), lo, hi, read, len(x)


def analyse():
    rng = np.random.default_rng(0)
    rows, diag, wins = [], [], []
    for scen in FIRST:
        path = A.OUT / f"closed34_{scen}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        wins.append(d.groupby("policy")["won"].mean().rename(scen))
        comps = [("C1" if scen in MAIN else "C3", BEST_KITE[scen]), ("C2", "model:h3")]
        if scen not in MAIN:
            comps.append(("C3", BEST_FIXED[scen]))
        for label, ref in comps:
            mean, lo, hi, read, n = paired(d, "musigma:h3", ref, rng)
            rows.append(
                {
                    "scenario": scen,
                    "rule": label,
                    "against": ref.removeprefix("fixed:"),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": read,
                    "n": n,
                }
            )
        p = d[d.policy == "musigma:h3"]
        cols = [c for c in p if c.startswith("pick_")] + [
            "near_tie_share",
            "clip_share",
            "switches",
            "wall_s",
        ]
        diag.append({"scenario": scen, **p[[c for c in cols if c in p]].mean().to_dict()})
    t = pd.DataFrame(rows)
    main = t[t.rule == "C1"]
    useful = bool((main["read-out"] == "beats").any() and not (main["read-out"] == "loses").any())
    pd.set_option("display.width", 250)
    lines = [
        "# Closed-loop test of the μ, σ planner: results",
        "",
        "Plan: `34_mu_sigma_closed_loop_plan.md` (approved 22 Nov).",
        "Script: `experiments/planning/closed34.py`.",
        "Fresh seeds 21 x00 000 +, paired by seed.",
        "",
        "## Wins per policy",
        "",
        "```",
        pd.concat(wins, axis=1).round(3).to_string(),
        "```",
        "",
        "## Paired comparisons (μ, σ planner − reference)",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**RQ5 statement (C1 beats in at least one of S2, S4, S5 and loses in none): "
        f"{'holds' if useful else 'does not hold'}.**",
        "",
        "## Planner diagnostics (means per battle)",
        "",
        "```",
        pd.DataFrame(diag).round(3).to_string(index=False),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["smoke", "run", "analyse"])
    p.add_argument("scenario", nargs="?")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--mlp", action="store_true", help="only the black-box planner (docs/planning/36)")
    p.add_argument("--hybrid", action="store_true", help="only the hybrid planner (docs/generalisation/20)")
    a = p.parse_args()
    if a.what == "analyse":
        return analyse()
    if a.what == "smoke":
        for scen in FIRST:
            PB.run(scen, policies(scen), 5, a.workers, SMOKE, "_smoke", prefix="closed34_")
            d = pd.read_csv(A.OUT / f"closed34_{scen}_smoke.csv")
            print(d.groupby("policy")[["won", "wall_s"]].mean().round(2).to_string(), flush=True)
        return
    pols = ["mlp:h3"] if a.mlp else ["hybrid:h3"] if a.hybrid else policies(a.scenario)
    PB.run(a.scenario, pols, a.n, a.workers, FIRST[a.scenario], prefix="closed34_")


if __name__ == "__main__":
    main()
