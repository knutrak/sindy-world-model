"""Analyse Phase B (docs/planning/07_phase_b_plan.md) -> docs/planning/08_phase_b.md.

Read-out as pre-registered: primary = model planner (frozen, horizon 3 s) − Phase A's best fixed intent on O1
(won), paired over 400 seeds, bootstrap 95%: beats (interval above 0) / ties (contains 0) / loses (below 0).
Share of headroom = (model − best fixed) / (exact oracle − best fixed) on the oracle's 200 paired seeds.
Main scenarios S1, S2, S5; S3, S4 flagged (model outside its tested range).

    uv run experiments/planning/analyse_phase_b.py
"""

import numpy as np
import pandas as pd

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

P = DATA_DIR / "processed" / "planning"
DOC = PROJECT_ROOT / "docs" / "planning" / "08_phase_b.md"
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
MAIN = ("S1", "S2", "S5")
BEST_FIXED = {  # Phase A, O1 (01_headroom.md), fixed before Phase B
    "S1": "fixed:hold/focus",
    "S2": "fixed:advance/focus",
    "S3": "fixed:hold/focus",
    "S4": "fixed:hold/focus",
    "S5": "fixed:advance/focus",
}
PRIMARY = "model:h3"
PLANNERS = ("model:h3", "model:h2", "model:h4", "model-focuscost:h3")
N_BOOT = 4000
OBJ = {
    "won": "O1 won",
    "margin": "O2 margin",
    "blue_preserved": "O3 preserved",
    "log_exchange": "O4 log exchange",
}


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def wide(d: pd.DataFrame, col: str) -> pd.DataFrame:
    return d.pivot(index="seed", columns="policy", values=col)


def paired(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    x = a - b
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def share(model: np.ndarray, base: np.ndarray, oracle: np.ndarray, rng) -> tuple[float, float, float]:
    def r(i):
        den = (oracle[i] - base[i]).mean()
        return (model[i] - base[i]).mean() / den if den != 0 else np.nan

    idx = np.arange(len(model))
    m = [r(rng.integers(0, len(idx), len(idx))) for _ in range(N_BOOT)]
    return float(r(idx)), float(np.nanquantile(m, 0.025)), float(np.nanquantile(m, 0.975))


def verdict(lo: float, hi: float) -> str:
    return "beats" if lo > 0 else ("loses" if hi < 0 else "ties")


def main() -> None:
    rng = np.random.default_rng(0)
    lines = [
        "# Phase B: the frozen model as a planner (results)",
        "",
        "Plan: `07_phase_b_plan.md` (pre-registered, approved 21 Nov). Data: "
        "`data/processed/planning/phase_b_<S>.csv`.",
        "Script: `experiments/planning/analyse_phase_b.py`. Planners re-plan every 1 s, commit 2 s; "
        "the primary has",
        "a 3 s horizon. Seeds 14 x00 000 +; planners, fixed intents and random 400 per scenario, "
        "exact oracle 200.",
        "Main scenarios S1, S2, S5; S3 and S4 are outside the mixed model's tested range.",
        "",
    ]
    primary, shares, second = [], [], []
    for scen in SCENARIOS:
        d = pd.read_csv(P / f"phase_b_{scen}.csv")
        d["log_exchange"] = np.log(d["exchange"])
        counts = d.groupby("policy").size().to_dict()
        lines += [f"## {scen}" + ("" if scen in MAIN else " (flagged: outside the model's tested range)"), ""]
        lines += [f"Battles: {counts}", ""]
        lines += [md(d.groupby("policy")[list(OBJ)].mean().rename(columns=OBJ).round(3)), ""]
        W = wide(d, "won").astype(float)
        base = BEST_FIXED[scen]
        seeds400 = W[[PRIMARY, base]].dropna().index
        for p in PLANNERS:
            s = W.loc[seeds400, [p, base]].dropna()
            mean, lo, hi = paired(s[p].to_numpy(), s[base].to_numpy(), rng)
            row = {
                "scenario": scen,
                "planner": p,
                "baseline (Phase A best fixed)": base.removeprefix("fixed:"),
                "planner won": s[p].mean(),
                "baseline won": s[base].mean(),
                "difference": mean,
                "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                "read-out": verdict(lo, hi),
                "n": len(s),
            }
            (primary if p == PRIMARY else second).append(row)
        if "oracle" in W:
            s = W[[PRIMARY, base, "oracle"]].dropna()
            sh, lo, hi = share(s[PRIMARY].to_numpy(), s[base].to_numpy(), s["oracle"].to_numpy(), rng)
            shares.append(
                {
                    "scenario": scen,
                    "oracle won": s["oracle"].mean(),
                    "baseline won": s[base].mean(),
                    "planner won": s[PRIMARY].mean(),
                    "share captured": sh,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "n": len(s),
                }
            )
        fixed = d[d.policy.str.startswith("fixed:")].groupby("policy")["won"].mean()
        lines += [f"Best fixed intent on Phase B's seeds (check): {fixed.idxmax()} {fixed.max():.3f}.", ""]
        pl = d[d.policy.isin(PLANNERS)]
        pick = [c for c in pl if c.startswith("pick_")]
        diag = pl.groupby("policy")[[*pick, "replans", "switches", "gap_best_second"]].mean()
        diag["switches per re-plan"] = (
            pl.groupby("policy")["switches"].sum() / pl.groupby("policy")["replans"].sum()
        )
        diag.columns = [c.removeprefix("pick_") for c in diag.columns]
        lines += [
            "Planner diagnostics (share of re-plans per pick; means per battle):",
            "",
            md(diag.round(3)),
            "",
        ]
        # secondary objectives for the primary planner
        sec = []
        for o in ("margin", "blue_preserved", "log_exchange"):
            Wo = wide(d, o)
            s = Wo[[PRIMARY, base]].dropna()
            mean, lo, hi = paired(s[PRIMARY].to_numpy(), s[base].to_numpy(), rng)
            sec.append({"objective": OBJ[o], "difference": mean, "95%": f"[{lo:+.3f}, {hi:+.3f}]"})
        lines += [
            "Primary planner − baseline, secondary objectives:",
            "",
            md(pd.DataFrame(sec).set_index("objective").round(3)),
            "",
        ]
    lines = (
        lines[:7]
        + [
            "## Primary result (O1 won, model:h3 − Phase A best fixed, 400 paired seeds)",
            "",
            md(pd.DataFrame(primary).set_index("scenario").round(3)),
            "",
            "## Share of the headroom captured (O1, on the exact oracle's 200 paired seeds)",
            "",
            md(pd.DataFrame(shares).set_index("scenario").round(3)),
            "",
            "## Secondary planners (O1, against the same baseline)",
            "",
            md(pd.DataFrame(second).set_index(["scenario", "planner"]).round(3)),
            "",
        ]
        + lines[7:]
    )
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
