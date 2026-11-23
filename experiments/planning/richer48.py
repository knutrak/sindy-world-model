"""A richer decision space: per-type roles and target priority (docs/planning/48_richer_intents_plan.md).

uv run experiments/planning/richer48.py smoke          2 seeds, S4: timing
uv run experiments/planning/richer48.py run S2         27 fixed combinations + oracle27-h3
uv run experiments/planning/richer48.py analyse        -> docs/planning/49_richer_intents.md
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed34 as C34  # noqa: E402
import phase_b as PB  # noqa: E402

from sindy_wm.envs.roles import ROLE_NAMES  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

SCEN = ("S2", "S4", "S5")
N = 200
DOC = PROJECT_ROOT / "docs" / "planning" / "49_richer_intents.md"
N_BOOT = 4000


def smoke():
    t0 = time.time()
    PB.run("S4", [*ROLE_NAMES, "oracle27-h3"], 2, 8, C34.SMOKE, "_smoke48", prefix="closed34_")
    d = pd.read_csv(C34.A.OUT / "closed34_S4_smoke48.csv")
    print(f"{len(d)} battles in {time.time() - t0:.0f} s")
    print(d.assign(o=d.policy == "oracle27-h3").groupby("o")["wall_s"].mean().round(2).to_string())


def boot_diff(W, a, b, rng):
    x = (W[a] - W[b]).dropna().to_numpy()
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return x.mean(), *np.quantile(m, [0.025, 0.975])


def analyse():
    rng = np.random.default_rng(0)
    rows, fixed_tables = [], {}
    for scen in SCEN:
        d = pd.read_csv(C34.A.OUT / f"closed34_{scen}.csv")
        d = d[d.seed < C34.FIRST[scen] + N]
        W = d.pivot_table(index="seed", columns="policy", values="won", aggfunc="first").astype(float)
        fixed = W[list(ROLE_NAMES)].mean().sort_values(ascending=False)
        fixed_tables[scen] = fixed
        best27, best5 = fixed.index[0], C34.BEST_FIXED[scen]
        r27 = boot_diff(W, "oracle27-h3", best27, rng)
        r5 = boot_diff(W, "oracle5-h3", best5, rng)
        n3 = boot_diff(W, "oracle27-h3", "oracle5-h3", rng)
        rows.append(
            {
                "scenario": scen,
                "best fixed of 27": best27,
                "its wins": fixed.iloc[0],
                "oracle27-h3 wins": W["oracle27-h3"].mean(),
                "Room27": r27[0],
                "Room27 95%": f"[{r27[1]:+.3f}, {r27[2]:+.3f}]",
                "best fixed team intent": best5,
                "its wins ": W[best5].mean(),
                "oracle5-h3 wins": W["oracle5-h3"].mean(),
                "Room5": r5[0],
                "Room5 95%": f"[{r5[1]:+.3f}, {r5[2]:+.3f}]",
                "N1 = Room27 − Room5": r27[0] - r5[0],
                "N3 oracle27 − oracle5": n3[0],
                "N3 95%": f"[{n3[1]:+.3f}, {n3[2]:+.3f}]",
            }
        )
    t = pd.DataFrame(rows)
    F = pd.DataFrame(fixed_tables)
    gap = F.max() - F  # each combination's gap to that scenario's best fixed combination
    near = gap[(gap <= 0.05).all(axis=1)].index.tolist()
    holds = int((t["N1 = Room27 − Room5"] >= 0.05).sum()) >= 2
    pd.set_option("display.width", 250)
    lines = [
        "# A richer decision space: per-type roles and target priority: results",
        "",
        "Plan: `48_richer_intents_plan.md` (approved 23 Nov). Script: `experiments/planning/richer48.py`.",
        f"`34`'s first {N} seeds per scenario, paired.",
        "",
        "## Room for a planner",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**Statement (N1 ≥ 5 points in at least two of S2, S4, S5): "
        f"{'holds' if holds else 'does not hold'}.**",
        "",
        f"**N2 (one fixed combination within 5 points of the best in all three scenarios):** "
        f"{', '.join(near) if near else 'none'}.",
        "",
        "## Fixed combinations (wins)",
        "",
        "```",
        F.round(3).sort_values("S2", ascending=False).to_string(),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["smoke", "run", "analyse"])
    p.add_argument("scenario", nargs="?")
    a = p.parse_args()
    if a.what == "smoke":
        return smoke()
    if a.what == "analyse":
        return analyse()
    PB.run(a.scenario, [*ROLE_NAMES, "oracle27-h3"], N, 8, C34.FIRST[a.scenario], prefix="closed34_")


if __name__ == "__main__":
    main()
