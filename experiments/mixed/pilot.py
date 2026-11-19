"""exp08 pilot (300 mixed battles): what the mix does, before the full collection (checkpoint 1).

Reads data/raw/exp08_mixed/pilot, builds per-type states (`states.typed`) and reports:
battle length and Blue win rate by marauder share; who dies first; the measured fire allocation
against the proportional rule; and a *preview* of the damage matrix (one-step least squares of each
type's health rate on the measured fire counts). Writes docs/mixed/01_pilot.md and the states to
data/processed/exp08_mixed/pilot.parquet.

    uv run experiments/mixed/pilot.py
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sindy_wm.io import storage
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR
from sindy_wm.states.typed import load_typed_states, unit_stats

RAW = RAW_DIR / "exp08_mixed" / "pilot"
OUT = DATA_DIR / "processed" / "exp08_mixed"
DOC = PROJECT_ROOT / "docs" / "mixed" / "01_pilot.md"
TYPES = ("MARINE", "MARAUDER")
T = [t.lower() for t in TYPES]
DT = 0.125


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def game_rate(attacker: str, target: str) -> float:
    """Target units removed per second per firing attacker (unit files; cooldown rounded up to ticks)."""
    a, t = unit_stats(attacker), unit_stats(target)
    bonus = 10.0 if (attacker == "MARAUDER" and target == "MARAUDER") else 0.0  # +10 vs ARMORED
    dmg = max(a["damage"] + bonus - t["armor"], 0.5)
    cooldown = np.ceil(a["cooldown"] * 16) / 16
    return dmg / cooldown / (t["hp"] + t["shield"])


def main() -> None:
    path = OUT / "pilot.parquet"
    if path.exists():
        s = pd.read_parquet(path)
    else:
        s = load_typed_states(RAW, TYPES)
        OUT.mkdir(parents=True, exist_ok=True)
        s.to_parquet(path)
    idx = storage.load_episode_index(RAW).set_index("episode_idx")
    idx["share"] = idx["n_marauder_blue"] / idx["n_blue"]
    idx["share_bin"] = pd.cut(idx["share"], [0, 0.29, 0.42, 1], labels=["~1/4", "~1/3", "~1/2"])
    last = s.groupby(level=0).tail(1).droplevel(1)
    idx["duration_s"] = s.groupby(level=0)["t"].max()
    lines = ["# exp08 pilot: 300 mixed marine / marauder battles (19 Nov)", ""]
    lines += ["Script: `experiments/mixed/pilot.py`. Exploration only.", ""]

    # 1. length and outcome by share
    by = idx.groupby("share_bin", observed=True).agg(
        battles=("n_blue", "size"), duration_s=("duration_s", "mean"), blue_win=("battle_won", "mean")
    )
    lines += ["## Battle length and Blue win rate by marauder share", "", md(by.round(3)), ""]

    # 2. who dies first: share of each type's starting units alive at the end, on the winning side
    rows = []
    for ep, r in idx.iterrows():
        side = "blue" if r["battle_won"] else "red"
        first = s.loc[ep].iloc[0]
        for t in T:
            n0 = first[f"alive_{side}_{t}"]
            if n0 > 0:
                rows.append({"type": t, "survive": last.loc[ep, f"alive_{side}_{t}"] / n0})
    surv = pd.DataFrame(rows).groupby("type")["survive"].mean()
    lines += [
        "## Survival on the winning side (share of starting units alive at the end)",
        "",
        md(surv.round(3).to_frame()),
        "",
    ]

    # 3. fire allocation: share of each attacker type's fire on marauders vs the proportional rule
    alloc = []
    for side, other in (("blue", "red"), ("red", "blue")):
        for a in T:
            f_m, f_M = s[f"fire_{side}_{a}_marine"], s[f"fire_{side}_{a}_marauder"]
            fire = f_m + f_M
            share_alive = s[f"alive_{other}_marauder"] / (
                s[f"alive_{other}_marine"] + s[f"alive_{other}_marauder"]
            )
            ok = (fire > 0) & share_alive.notna()
            alloc.append(
                {
                    "side": side,
                    "attacker": a,
                    "fire on marauders (measured)": float(f_M[ok].sum() / fire[ok].sum()),
                    "marauder share of enemies alive (same steps, fire-weighted)": float(
                        (share_alive[ok] * fire[ok]).sum() / fire[ok].sum()
                    ),
                }
            )
    lines += [
        "## Fire allocation",
        "",
        "Share of each attacker type's fire (units with a target in range) that is on marauders, against",
        "the marauder share among the enemy's alive units at the same steps. Blue's rule: nearest or",
        "focus (weakest by absolute hp, so marines); Red's AI: nearest new target, keeps its target.",
        "",
        md(pd.DataFrame(alloc).round(3)),
        "",
    ]

    # 4. preview of the damage matrix: d(health of target type t)/dt = -sum_a k_at * fire_{a->t}
    rates = []
    for _, g in s.groupby(level=0):
        if len(g) < 11:
            continue
        r = g.copy()
        for side in ("blue", "red"):
            for t in T:
                r[f"d_{side}_{t}"] = savgol_filter(g[f"health_{side}_{t}"], 9, 2, deriv=1, delta=DT)
        rates.append(r)
    r = pd.concat(rates)
    mat = []
    for side, other in (("blue", "red"), ("red", "blue")):
        for t in T:
            both = (r[[f"alive_{side}_{x}" for x in T]].sum(axis=1) > 0) & (
                r[[f"alive_{other}_{x}" for x in T]].sum(axis=1) > 0
            )
            X = np.column_stack([r.loc[both, f"fire_{other}_{a}_{t}"] for a in T])
            y = r.loc[both, f"d_{side}_{t}"].to_numpy()
            k, *_ = np.linalg.lstsq(X, y, rcond=None)
            for a, kk in zip(TYPES, k, strict=True):
                mat.append(
                    {
                        "attacker (side)": f"{a.lower()} ({other})",
                        "target": t,
                        "fitted": -kk,
                        "game": game_rate(a, t.upper()),
                        "fitted / game": -kk / game_rate(a, t.upper()),
                    }
                )
    lines += [
        "## Preview: damage matrix from measured fire (one-step least squares, pilot only)",
        "",
        "Each type's health rate (SG 9) regressed on the measured number of attackers of each type that",
        "are shooting a unit of that type (in range). Units: target units removed per second per firing",
        "attacker. Game value: (damage + bonus − armor) / cooldown rounded up to ticks / target hp.",
        "This is a preview of checkpoint 2 with *measured* allocation; it says nothing yet about closures.",
        "",
        md(pd.DataFrame(mat).round(3)),
        "",
    ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
