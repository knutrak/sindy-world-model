"""Perfect geometry, SINDy damage (plan: docs/planning/50_geometry_oracle_plan.md).

uv run experiments/planning/geo50.py run        geo5:h3 on 34's first 200 seeds, S1-S5
uv run experiments/planning/geo50.py analyse    -> docs/planning/51_geometry_oracle.md
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

N = 200
DOC = PROJECT_ROOT / "docs" / "planning" / "51_geometry_oracle.md"


def analyse():
    rng = np.random.default_rng(0)
    rows, shares = [], {}
    for scen in C34.FIRST:
        d = pd.read_csv(C34.A.OUT / f"closed34_{scen}.csv")
        d = d[d.seed < C34.FIRST[scen] + N]
        best = C34.BEST_FIXED[scen]
        W = d.pivot_table(index="seed", columns="policy", values="won", aggfunc="first").astype(float)
        res = {}
        for a, b in (
            ("geo5:h3", best),
            ("oracle5-h3", best),
            ("geo5:h3", "oracle5-h3"),
            ("geo5:h3", "musigma5:h3"),
        ):
            x = (W[a] - W[b]).dropna().to_numpy()
            m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            lo, hi = np.quantile(m, [0.025, 0.975])
            res[(a, b)] = x.mean()
            rows.append(
                {
                    "scenario": scen,
                    "policy": a,
                    "against": b.removeprefix("fixed:"),
                    "wins": W[a].mean(),
                    "its wins": W[b].mean(),
                    "difference": x.mean(),
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                }
            )
        ceiling = res[("oracle5-h3", best)]
        shares[scen] = res[("geo5:h3", best)] / ceiling if ceiling > 0 else np.nan
    t = pd.DataFrame(rows)
    sh = pd.Series(shares, name="share of matched ceiling")
    geo = int((sh >= 0.7).sum())
    dmg = int((sh <= 0.3).sum())
    verdict = (
        "geometry is the bottleneck"
        if geo >= 3
        else "damage and death equations are a second bottleneck"
        if dmg >= 3
        else "mixed"
    )
    pd.set_option("display.width", 250)
    lines = [
        "# Perfect geometry, SINDy damage: results",
        "",
        "Plan: `50_geometry_oracle_plan.md`. Script: `experiments/planning/geo50.py`.",
        f"`34`'s first {N} seeds per scenario, paired.",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        "Share of the matched ceiling captured, (geo5 − best fixed) / (oracle5-h3 − best fixed):",
        "",
        "```",
        sh.round(2).to_string(),
        "```",
        "",
        f"**Read-out fixed in the plan: {verdict}** (share ≥ 0.7 in {geo} of 5 scenarios, ≤ 0.3 in {dmg}).",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "analyse"])
    if p.parse_args().what == "analyse":
        return analyse()
    for scen in C34.FIRST:
        PB.run(scen, ["geo5:h3"], N, 8, C34.FIRST[scen], prefix="closed34_")


if __name__ == "__main__":
    main()
