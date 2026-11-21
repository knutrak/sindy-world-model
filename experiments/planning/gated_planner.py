"""A confidence-gated planner (pre-registered in docs/planning/22_gated_planner_plan.md).

uv run --extra explore experiments/planning/gated_planner.py threshold   tau per model, from the cf sets
uv run experiments/planning/gated_planner.py closed                      fresh seeds 18 x00 000 +, 400 each
uv run experiments/planning/gated_planner.py report                      -> docs/planning/23_gated_planner.md
"""

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

P = DATA_DIR / "processed" / "planning"
TAU = P / "gated_tau.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "23_gated_planner.md"
FRESH = {f"S{i + 1}": 18_000_000 + 100_000 * i for i in range(5)}
MAIN = ("S1", "S2", "S5")
H = 3.0


def threshold() -> None:
    from model_regret import MODELS, SETS, from_branch, load, margin_change

    from sindy_wm.models.intent_model import DT

    models = pickle.load(open(MODELS, "rb"))
    out = {}
    for _scen, (folder, name) in SETS.items():
        m = models[name]
        L = m.L
        states, _ = from_branch(load(folder), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = L.batch(states, horizon=round(H / DT), every=10**9, history=0, min_future=1)
            Y = L.team_view(m.rollout(B))
        TY = B["team"]["Y"]
        g = states.groupby(level=0, sort=False)
        f = pd.DataFrame(
            {
                "s": g["seed"].first().loc[B["ep"]].to_numpy(),
                "b": g["branch"].first().loc[B["ep"]].to_numpy(),
                "t": margin_change(TY, TY, H),
                "m": margin_change(Y, TY, H),
            }
        )
        T = f.pivot(index="s", columns="b", values="t").to_numpy()
        M = f.pivot(index="s", columns="b", values="m").to_numpy()
        err = np.concatenate(
            [np.abs((M[:, i] - M[:, j]) - (T[:, i] - T[:, j])) for i, j in combinations(range(5), 2)]
        )
        out[name] = {
            "tau": float(np.median(err)),
            "states": len(T),
            "pairs": len(err),
            "p25": float(np.quantile(err, 0.25)),
            "p75": float(np.quantile(err, 0.75)),
        }
        print(name, out[name])
    json.dump(out, open(TAU, "w"), indent=1)


def closed() -> None:
    import phase_b as PB

    arms = ["gated:h3", "model:h3", *PB.POLICY_SETS["fixed"]]
    for scen, first in FRESH.items():
        PB.run(scen, arms, 400, 9, first, "", "gated_")


def extra() -> None:
    """docs/planning/25: the terminal-value and withdraw-firing arms on the same seeds."""
    import phase_b as PB

    for scen, first in FRESH.items():
        PB.run(scen, ["terminal:h3", "wfire:h3"], 400, 9, first, "", "gated_")


def report() -> None:
    from analyse_phase_b import BEST_FIXED

    rng = np.random.default_rng(0)
    tau = json.load(open(TAU))
    rows, use = [], []
    for scen in FRESH:
        d = pd.read_csv(P / f"gated_{scen}.csv")
        W = d.pivot(index="seed", columns="policy", values="won").astype(float)
        base = BEST_FIXED[scen]
        for arm, ref in (("gated:h3", base), ("gated:h3", "model:h3"), ("model:h3", base)):
            s = W[[arm, ref]].dropna()
            x = s[arm].to_numpy() - s[ref].to_numpy()
            mm = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            lo, hi = np.quantile(mm, [0.025, 0.975])
            rows.append(
                {
                    "scenario": scen,
                    "arm": arm,
                    "against": ref.removeprefix("fixed:"),
                    "arm won": s[arm].mean(),
                    "against won": s[ref].mean(),
                    "difference": x.mean(),
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": "beats" if lo > 0 else ("loses" if hi < 0 else "ties"),
                    "flagged": scen not in MAIN,
                    "n": len(s),
                }
            )
        g = d[d.policy == "gated:h3"]
        use.append(
            {
                "scenario": scen,
                "default": base.removeprefix("fixed:"),
                "share of re-plans off the default": g["deviate_share"].mean(),
                **{c.removeprefix("share_"): g[c].fillna(0).mean() for c in g if c.startswith("share_")},
            }
        )
        fixed = d[d.policy.str.startswith("fixed:")].groupby("policy")["won"].mean()
        use[-1]["best fixed on these seeds"] = f"{fixed.idxmax().removeprefix('fixed:')} {fixed.max():.3f}"
    lines = [
        "# The confidence-gated planner: results",
        "",
        "Plan: `22_gated_planner_plan.md` (pre-registered). Script: `experiments/planning/gated_planner.py`.",
        f"Thresholds tau (team margin, ten units of the first type): {json.dumps(tau)}.",
        "",
        "## Wins, paired over 400 fresh seeds (18 x00 000 +)",
        "",
        "```",
        pd.DataFrame(rows).set_index(["scenario", "arm", "against"]).round(3).to_string(),
        "```",
        "",
        "## What the gated planner plays (share of steps)",
        "",
        "```",
        pd.DataFrame(use).set_index("scenario").round(3).to_string(),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["threshold", "closed", "extra", "report"])
    {"threshold": threshold, "closed": closed, "extra": extra, "report": report}[parser.parse_args().what]()


if __name__ == "__main__":
    main()
