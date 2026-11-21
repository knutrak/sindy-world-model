# ruff: noqa: E501  (long text lines in the report)
"""Analyse Phase A of the headroom test (docs/planning/00_headroom_plan.md) -> docs/planning/01_headroom.md.

For each scenario and objective: the best fixed intent (chosen on its 400 seeds), the oracles, and the headroom
(perturbed oracle - best fixed) with a paired 95% bootstrap interval on the oracles' 200 seeds; the foresight share;
the intents the oracle uses; and the go / no-go decision fixed in the plan.

    uv run experiments/planning/analyse.py
"""

import numpy as np
import pandas as pd

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

IN = DATA_DIR / "processed" / "planning"
DOC = PROJECT_ROOT / "docs" / "planning" / "01_headroom.md"
FIG = PROJECT_ROOT / "docs" / "planning" / "01_headroom.png"
OBJECTIVES = {
    "won": "O1 win",
    "margin": "O2 health margin",
    "blue_preserved": "O3 units preserved",
    "log_exchange": "O4 exchange ratio (log)",
}
NATURAL = {"S4": "blue_preserved"}
N_BOOT = 1000


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def paired_ci(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = b - a
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    lines = [
        "# Headroom test, Phase A: results (run 20-21 Nov 2026)",
        "",
        "Plan: `00_headroom_plan.md`. Data: `data/processed/planning/headroom_<scenario>.csv`. Script:",
        "`experiments/planning/analyse.py`. Fixed intents: 400 seeds; oracles: 200 seeds (paired comparisons on",
        "those). Oracles re-plan every 1 s, commit 2 s, horizon 8 s, optimise the health margin (the survival",
        "oracle optimises Blue units preserved). Health margin = Blue − Red health left, as shares of full health.",
        "",
        "Notes on the run:",
        "",
        "- **Duplicates removed.** A stopped run kept going beside its restart, so S3-S5 oracles (and the S4",
        "  survival oracle) were played twice. Every duplicate pair had identical outcomes (the games are",
        "  deterministic per seed); one copy of each was kept.",
        "- **O3 (units preserved) is won by always withdrawing,** which never wins a battle. Read O3 together",
        "  with O1.",
        "- **The survival oracle cannot value a sustained withdraw:** each candidate is played for 2 s, then",
        "  advance/nearest to the 8 s horizon, so withdrawing only ever looks like a 2 s delay. Its O3 check",
        "  below says little about how much a survival-minded planner could gain.",
        "- **Withdraw ignores the target rule,** so the oracles have five distinct candidates; withdraw is",
        "  logged as withdraw/nearest.",
        "",
    ]
    summary, decisions = [], []
    for path in sorted(IN.glob("headroom_S*.csv")):
        t = pd.read_csv(path)
        assert not t.duplicated(["seed", "policy"]).any(), f"duplicate rows in {path}"
        scen = t["scenario"].iloc[0]
        t["log_exchange"] = np.log(t["exchange"])
        t["won"] = t["won"].astype(float)
        fixed = t[t.policy.str.startswith("fixed:") | (t.policy == "random")]
        means = t.groupby("policy")[list(OBJECTIVES)].mean()
        oracle_seeds = set(t.loc[t.policy == "oracle-perturbed", "seed"])
        lines += [
            f"## {scen}",
            "",
            f"Battles: {t.groupby('policy').size().to_dict()}",
            "",
            md(means.round(3)),
            "",
        ]
        for obj, label in OBJECTIVES.items():
            fm = fixed[fixed.policy != "random"].groupby("policy")[obj].mean()
            best = fm.idxmax()
            row = {
                "scenario": scen,
                "objective": label,
                "best fixed": best.removeprefix("fixed:"),
                "best fixed value": fm.max(),
            }
            base = t[(t.policy == best) & t.seed.isin(oracle_seeds)].set_index("seed")[obj]
            oracles = ["oracle", "oracle-perturbed"] + (["oracle-survival"] if scen == "S4" else [])
            for o in oracles:
                if o not in set(t.policy):
                    continue
                val = t[t.policy == o].set_index("seed")[obj].reindex(base.index)
                lo, hi = paired_ci(base.to_numpy(), val.to_numpy())
                row[f"{o}"] = float(val.mean())
                row[f"{o} − best fixed"] = float((val - base).mean())
                row[f"{o} − best fixed, 95%"] = f"[{lo:+.3f}, {hi:+.3f}]"
            if "oracle" in row and "oracle-perturbed" in row:
                gap = row["oracle − best fixed"]
                row["foresight share"] = (
                    (row["oracle"] - row["oracle-perturbed"]) / gap if abs(gap) > 1e-9 else np.nan
                )
            summary.append(row)
        # go / no-go: O1, or the natural objective (S4: O3, survival oracle)
        go = []
        for obj, o in [("won", "oracle-perturbed")] + (
            [(NATURAL[scen], "oracle-survival")] if scen in NATURAL else []
        ):
            r = next(x for x in summary if x["scenario"] == scen and x["objective"] == OBJECTIVES[obj])
            if f"{o} − best fixed" in r:
                diff = r[f"{o} − best fixed"]
                lo = float(r[f"{o} − best fixed, 95%"].strip("[]").split(",")[0])
                thr = 0.05
                go.append((obj, o, diff, lo, diff >= thr and lo > 0))
        decisions.append({"scenario": scen, "checks": go, "go": any(g[-1] for g in go)})
        shares = (
            t[t.policy.str.startswith("oracle")]
            .groupby("policy")[[c for c in t if c.startswith("share_")]]
            .mean()
        )
        lines += ["Intent use by the oracles (share of steps):", "", md(shares.round(3)), ""]
    S = pd.DataFrame(summary)
    lines += [
        "## Headroom per scenario and objective",
        "",
        md(S.set_index(["scenario", "objective"]).round(3)),
        "",
    ]
    lines += [
        "## Go / no-go for Phase B (rule: perturbed oracle ≥ best fixed + 5 points, 95% interval above 0)",
        "",
    ]
    for d in decisions:
        checks = "; ".join(
            f"{o} on {obj}: {diff:+.3f} (lower {lo:+.3f})" for obj, o, diff, lo, _ in d["checks"]
        )
        lines.append(f"- **{d['scenario']}: {'GO' if d['go'] else 'no go'}** — {checks}")
    DOC.write_text("\n".join(lines) + "\n")
    try:
        import matplotlib.pyplot as plt

        w = S[S.objective == OBJECTIVES["won"]].set_index("scenario")
        x = np.arange(len(w))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.25, w["best fixed value"], 0.25, label="best fixed intent", color="#999999")
        ax.bar(x, w["oracle-perturbed"], 0.25, label="oracle, perturbed copies", color="#1f77b4")
        ax.bar(x + 0.25, w["oracle"], 0.25, label="oracle, exact copies", color="#aec7e8")
        ax.set_xticks(x, w.index)
        ax.set_ylabel("Blue win rate (fixed: 400 seeds; oracles: 200)")
        ax.set_title("How much can choosing intents gain? (O1 win)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG, dpi=120)
    except Exception as err:  # the figure is optional
        print("figure skipped:", err)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
