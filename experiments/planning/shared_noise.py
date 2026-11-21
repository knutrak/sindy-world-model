"""The perturbed oracle with one noise draw per re-plan shared by the candidates
(docs/planning/29_shared_noise.md).

Compares it, paired on Phase A's 200 oracle seeds, with the independent-noise perturbed oracle, the exact
oracle and
the best fixed intent; reports intent use and how often its top two candidates are within 1 hp.

    uv run experiments/planning/shared_noise.py
"""

import numpy as np
import pandas as pd

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

P = DATA_DIR / "processed" / "planning"
DOC = PROJECT_ROOT / "docs" / "planning" / "29_shared_noise.md"
BEST = {
    "S1": "fixed:hold/focus",
    "S2": "fixed:advance/focus",
    "S3": "fixed:hold/focus",
    "S4": "fixed:hold/focus",
    "S5": "fixed:advance/focus",
}


def main() -> None:
    rng = np.random.default_rng(0)
    rows, use = [], []
    for scen in BEST:
        a = pd.read_csv(P / f"headroom_{scen}.csv")
        s = pd.read_csv(P / f"headroom_{scen}_shared.csv")
        d = pd.concat([a, s])
        W = d.pivot_table(index="seed", columns="policy", values="won", aggfunc="first").astype(float)
        W = W.dropna(subset=["oracle-perturbed-shared"])
        for ref in ("oracle-perturbed", "oracle", BEST[scen]):
            x = (W["oracle-perturbed-shared"] - W[ref]).dropna().to_numpy()
            m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            lo, hi = np.quantile(m, [0.025, 0.975])
            rows.append(
                {
                    "scenario": scen,
                    "against": ref.removeprefix("fixed:"),
                    "shared won": W["oracle-perturbed-shared"].mean(),
                    "against won": W[ref].mean(),
                    "difference": x.mean(),
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "n": len(x),
                }
            )
        exact, base = W["oracle"].mean(), W[BEST[scen]].mean()
        rows[-1]["share of exact-oracle headroom"] = (W["oracle-perturbed-shared"].mean() - base) / (
            exact - base
        )
        sh = s[s.policy == "oracle-perturbed-shared"]
        sc = [c for c in sh if c.startswith("share_")]
        use.append(
            {
                "scenario": scen,
                **{c.removeprefix("share_"): sh[c].fillna(0).mean() for c in sc},
                "top two within 1 hp": sh["near_tie_share"].mean(),
                "median gap (hp)": sh["gap_median"].median(),
            }
        )
    lines = [
        "# The perturbed oracle with shared noise",
        "",
        "Script: `experiments/planning/shared_noise.py`. Phase A's perturbed oracle "
        "(positions ± 0.2, health ± 5%)",
        "with one draw per re-plan applied to all five candidates' copies, instead of one draw per copy; "
        "Phase A's",
        "200 oracle seeds per scenario, paired.",
        "",
        "## Wins",
        "",
        "```",
        pd.DataFrame(rows).set_index(["scenario", "against"]).round(3).to_string(),
        "```",
        "",
        "## Intent use (share of steps) and how close the top two candidates are",
        "",
        "```",
        pd.DataFrame(use).set_index("scenario").round(3).to_string(),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
