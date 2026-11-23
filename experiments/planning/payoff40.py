"""Mirrored Red and the payoff table (approved plan: docs/planning/40_mirrored_red_payoff_plan.md).

uv run experiments/planning/payoff40.py gate               gate S: symmetry on the S1 sampler (400 seeds)
uv run experiments/planning/payoff40.py time               2 seeds per cell in S4, timed
uv run experiments/planning/payoff40.py run S2             the 7 x 7 table, 200 seeds per cell
uv run experiments/planning/payoff40.py analyse            -> docs/planning/41_mirrored_red_payoff.md
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.intents import IntentPolicy  # noqa: E402
from sindy_wm.envs.mirror import MirrorRed  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

BLUE = [
    "advance/nearest",
    "advance/focus",
    "hold/nearest",
    "hold/focus",
    "withdraw/nearest",
    "kite/nearest",
    "kite/focus",
]
RED = [
    "scripted",
    "advance/nearest",
    "advance/focus",
    "hold/focus",
    "withdraw/nearest",
    "kite/nearest",
    "kite/focus",
]
FIRST = {"S1": 22_000_000, "S2": 22_100_000, "S4": 22_300_000}
GATE_SEED = 22_900_000
DOC = PROJECT_ROOT / "docs" / "planning" / "41_mirrored_red_payoff.md"
N_BOOT = 4000


def play(job) -> dict:
    scen, seed, blue, red = job
    mapname, sampler, wd, max_s, _ = A.SCENARIOS[scen]
    s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        mapname,
        step_mul=A.STEP_MUL,
        spawn=s.spawn,
        unit_counts=None if types else s.unit_counts,
        unit_types=types,
    )
    env.reset(seed=seed)
    _set_start_health(env, *s.start_health)
    bp = IntentPolicy(*blue.split("/"), withdraw_radius=wd)
    bp.reset(seed)
    rp = None if red == "scripted" else MirrorRed(IntentPolicy(*red.split("/"), withdraw_radius=wd))
    if rp is not None:
        rp.reset(seed)
    t0, steps, won = time.time(), 0, False
    while steps < max_s * 16 // A.STEP_MUL:
        steps += 1
        if rp is not None:
            rp.apply(env)
        _, _, term, trunc, info = env.step(bp.act(env))
        if term or trunc:
            won = bool(info.get("battle_won", False))
            break
    g = env.unwrapped
    return {
        "scenario": scen,
        "seed": seed,
        "blue": blue,
        "red": red,
        "won": won,
        "seconds": steps * A.STEP_MUL / 16,
        "timeout": not (term or trunc) if steps else False,
        "blue_left": len(g.agents),
        "red_left": len(g.enemies),
        "wall_s": time.time() - t0,
    }


def pool_run(jobs, path):
    with Pool(8) as pool:
        rows = pool.map(play, jobs, chunksize=4)
    t = pd.DataFrame(rows)
    t.to_csv(path, index=False)
    return t


def gate():
    jobs = [
        ("S1", GATE_SEED + i, b, r)
        for i in range(400)
        for b, r in (("advance/nearest", "advance/nearest"), ("kite/nearest", "kite/nearest"))
    ]
    t = pool_run(jobs, A.OUT / "payoff40_gate.csv")
    rng = np.random.default_rng(0)
    for (b, r), g in t.groupby(["blue", "red"]):
        x = g.won.to_numpy(float)
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
        lo, hi = np.quantile(m, [0.025, 0.975])
        print(
            f"Blue {b} vs mirrored Red {r}: Blue wins {x.mean():.3f} [{lo:.3f}, {hi:.3f}], "
            f"timeouts {g.timeout.mean():.3f}; covers 0.5: {lo <= 0.5 <= hi}"
        )


def time_it():
    jobs = [("S4", FIRST["S4"] + i, b, r) for i in range(2) for b in BLUE for r in RED]
    t0 = time.time()
    t = pool_run(jobs, A.OUT / "payoff40_time.csv")
    print(f"{len(jobs)} battles in {time.time() - t0:.0f} s; mean wall per battle {t.wall_s.mean():.2f} s")


def run(scen, n=200):
    jobs = [(scen, FIRST[scen] + i, b, r) for i in range(n) for b in BLUE for r in RED]
    t0 = time.time()
    pool_run(jobs, A.OUT / f"payoff40_{scen}.csv")
    print(f"{scen}: {len(jobs)} battles in {time.time() - t0:.0f} s", flush=True)


def analyse():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    lines = [
        "# Mirrored Red and the payoff table: results",
        "",
        "Plan: `40_mirrored_red_payoff_plan.md` (approved 23 Nov).",
        "Script: `experiments/planning/payoff40.py`.",
        "",
    ]
    g = A.OUT / "payoff40_gate.csv"
    if g.exists():
        t = pd.read_csv(g)
        lines += [
            "## Gate S (S1 sampler, 400 seeds, both sides the same intent)",
            "",
            "```",
            t.groupby(["blue", "red"])[["won", "timeout", "seconds"]].mean().round(3).to_string(),
            "```",
            "",
        ]
    summary = []
    for scen in FIRST:
        p = A.OUT / f"payoff40_{scen}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        M = t.pivot_table(index="blue", columns="red", values="won", aggfunc="mean").reindex(
            index=BLUE, columns=RED
        )
        short = (M.max(0) - M).max(1)  # each row's largest gap to the column best
        best_row = M.mean(1).idxmax()
        headroom = float(M.max(0).mean() - M.mean(1).max())
        mirrored = [c for c in RED if c != "scripted"]
        # bootstrap P2 by seed
        W = t.pivot_table(index=["seed"], columns=["blue", "red"], values="won", aggfunc="first").astype(
            float
        )
        hs = []
        for _ in range(1000):
            s = W.iloc[rng.integers(0, len(W), len(W))].mean()
            Mb = s.unstack("red").reindex(index=BLUE, columns=RED)
            hs.append(Mb.max(0).mean() - Mb.mean(1).max())
        lo, hi = np.quantile(hs, [0.025, 0.975])
        p1 = bool((short > 0.05).all())
        summary.append(
            {
                "scenario": scen,
                "best single Blue row": best_row,
                "its mean over Red columns": float(M.mean(1).max()),
                "near-dominant rows (shortfall ≤ 5)": ", ".join(short[short <= 0.05].index) or "none",
                "P1: no single best rule": p1,
                "P2 headroom (points)": 100 * headroom,
                "P2 95%": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
                "P3 best row vs scripted": float(M.loc[best_row, "scripted"]),
                "P3 best row vs mirrored (mean)": float(M.loc[best_row, mirrored].mean()),
                "statement (P1 and P2 ≥ 10)": bool(p1 and headroom >= 0.10),
            }
        )
        lines += [
            f"## {scen}: Blue win rate (rows Blue, columns Red), 200 paired seeds per cell",
            "",
            "```",
            M.round(3).to_string(),
            "```",
            "",
            "Shortfall of each Blue row (largest gap to a column's best):",
            "",
            "```",
            short.round(3).to_string(),
            "```",
            "",
        ]
    if summary:
        lines += ["## Read-outs", "", "```", pd.DataFrame(summary).round(3).to_string(index=False), "```", ""]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gate", "time", "run", "analyse"])
    p.add_argument("scenario", nargs="?")
    a = p.parse_args()
    if a.what == "run":
        return run(a.scenario)
    {"gate": gate, "time": time_it, "analyse": analyse}[a.what]()


if __name__ == "__main__":
    main()
