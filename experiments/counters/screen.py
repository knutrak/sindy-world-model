"""Stage 1: payoff screens for match-ups with counter mechanics (plan: docs/counters/00_plan.md).

uv run experiments/counters/screen.py calibrate     Red counts for a balanced advance-vs-advance match-up
uv run experiments/counters/screen.py table Z       the 6 x 6 payoff table, 150 paired seeds per cell
uv run experiments/counters/screen.py analyse       -> docs/counters/01_screen.md
"""

import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.envs.episode_sampling import MixedBattleSampler  # noqa: E402
from sindy_wm.envs.intents import IntentPolicy  # noqa: E402
from sindy_wm.envs.mirror import MirrorRed  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

OUT = DATA_DIR / "processed" / "counters"
DOC = PROJECT_ROOT / "docs" / "counters" / "01_screen.md"
MAP, STEP_MUL, MAX_S, WD = "mixed_m_mrd", 2, 90, 8.0
BLUE = ["MARINE"] * 7 + ["MARAUDER"] * 3
MATCHUPS = {  # docs/counters/00, amendment 1: both unit counts vary
    "Z": {
        "red": lambda c: ["ZERGLING"] * c[0] + ["BANELING"] * c[1],
        "grid": ((8, 3), (10, 2), (12, 2), (6, 3)),
    },
    "P": {
        "red": lambda c: ["STALKER"] * c[0] + ["ZEALOT"] * c[1],
        "grid": ((5, 1), (5, 2), (6, 1), (3, 3)),
    },
}
MENU2 = [  # docs/counters/02: withdraw dropped, flank added
    "advance/nearest",
    "advance/focus",
    "stand/nearest",
    "kite/nearest",
    "spread/nearest",
    "flank/nearest",
]
STAGE2 = {  # docs/counters/02: match-up -> (Red units, spawn distance, seed0)
    "M": (["MARINE"] * 7 + ["MARAUDER"] * 3, (8.5, 16.0), 24_200_000),
    "Z2": (["ZERGLING"] * 10 + ["BANELING"] * 2, (8.5, 16.0), 24_300_000),
}
MENU = [
    "advance/nearest",
    "advance/focus",
    "stand/nearest",
    "withdraw/nearest",
    "kite/nearest",
    "spread/nearest",
]
SEED0 = {"calibrate": 24_900_000, "Z": 24_000_000, "P": 24_100_000}
N_CELL = 150


def play(job) -> dict:
    red_types, seed, blue, red, tag, *dist = job
    bounds = walkable_bounds(load_map_info(MAP))
    sampler = MixedBattleSampler() if not dist else replace(MixedBattleSampler(), spawn_distance=dist[0])
    s = sampler.sample(seed, bounds)
    env = make_env(MAP, step_mul=STEP_MUL, spawn=s.spawn, unit_types=(BLUE, red_types))
    env.reset(seed=seed)
    bp = IntentPolicy(*blue.split("/"), withdraw_radius=WD, bounds=bounds)
    bp.reset(seed)
    rp = MirrorRed(IntentPolicy(*red.split("/"), withdraw_radius=WD, bounds=bounds))
    rp.reset(seed + 7_000_000)  # Red's own stream (docs/planning/40 caution)
    won, steps, done = False, 0, False
    t0 = time.time()
    while steps < MAX_S * 16 // STEP_MUL:
        steps += 1
        rp.apply(env)
        _, _, term, trunc, info = env.step(bp.act(env))
        if term or trunc:
            won, done = bool(info.get("battle_won", False)), True
            break
    g = env.unwrapped
    return {
        "tag": tag,
        "seed": seed,
        "blue": blue,
        "red": red,
        "won": won,
        "timeout": not done,
        "seconds": steps * STEP_MUL / 16,
        "blue_left": len(g.agents),
        "red_left": len(g.enemies),
        "wall_s": time.time() - t0,
    }


def calibrate():
    jobs = []
    for mu, spec in MATCHUPS.items():
        for n in spec["grid"]:
            jobs += [
                (
                    spec["red"](n),
                    SEED0["calibrate"] + i,
                    "advance/nearest",
                    "advance/nearest",
                    f"{mu}{n[0]}-{n[1]}",
                )
                for i in range(100)
            ]
    with Pool(8) as pool:
        t = pd.DataFrame(pool.map(play, jobs, chunksize=4))
    t.to_csv(OUT / "calibrate3.csv", index=False)
    s = t.groupby("tag")[["won", "timeout", "seconds"]].mean()
    print(s.round(3).to_string())
    pick = {}
    for mu, spec in MATCHUPS.items():
        w = {n: s.loc[f"{mu}{n[0]}-{n[1]}", "won"] for n in spec["grid"]}
        pick[mu] = list(min(w, key=lambda n: abs(w[n] - 0.5)))
    json.dump(pick, open(OUT / "calibration.json", "w"))
    print("picked", pick)


def table(mu):
    if mu in STAGE2:
        red_types, dist, seed0 = STAGE2[mu]
        jobs = [(red_types, seed0 + i, b, r, mu, dist) for i in range(N_CELL) for b in MENU2 for r in MENU2]
        t0 = time.time()
        with Pool(8) as pool:
            t = pd.DataFrame(pool.map(play, jobs, chunksize=4))
        t.to_csv(OUT / f"table_{mu}.csv", index=False)
        print(f"{mu}: {len(t)} battles in {time.time() - t0:.0f} s")
        return
    n = json.load(open(OUT / "calibration.json"))[mu]
    red_types = MATCHUPS[mu]["red"](tuple(n))
    jobs = [(red_types, SEED0[mu] + i, b, r, mu) for i in range(N_CELL) for b in MENU for r in MENU]
    t0 = time.time()
    with Pool(8) as pool:
        t = pd.DataFrame(pool.map(play, jobs, chunksize=4))
    t.to_csv(OUT / f"table_{mu}.csv", index=False)
    print(f"{mu}: {len(t)} battles in {time.time() - t0:.0f} s")


def analyse():
    rng = np.random.default_rng(0)
    pick = json.load(open(OUT / "calibration.json"))
    cal = pd.concat(
        [
            pd.read_csv(OUT / f)
            for f in ("calibrate.csv", "calibrate2.csv", "calibrate3.csv")
            if (OUT / f).exists()
        ]
    )
    cal = cal.groupby("tag")[["won", "timeout"]].mean()
    lines = [
        "# Counter match-ups, stage 1: payoff screens",
        "",
        "Plan: `00_plan.md` (approved 24 Nov). Script: `experiments/counters/screen.py`.",
        "",
        "## Calibration (both sides advance/nearest, 100 battles per setting)",
        "",
        "```",
        cal.round(3).to_string(),
        "```",
        "",
        f"Picked Red counts: {pick}.",
        "",
    ]
    summary = []
    for mu in MATCHUPS:
        p = OUT / f"table_{mu}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        M = t.pivot_table(index="blue", columns="red", values="won", aggfunc="mean").reindex(
            index=MENU, columns=MENU
        )
        short = (M.max(0) - M).max(1)
        best_row = M.mean(1).idxmax()
        head = float(M.max(0).mean() - M.mean(1).max())
        W = t.pivot_table(index="seed", columns=["blue", "red"], values="won", aggfunc="first").astype(float)
        hs = []
        for _ in range(1000):
            Mb = (
                W.iloc[rng.integers(0, len(W), len(W))]
                .mean()
                .unstack("red")
                .reindex(index=MENU, columns=MENU)
            )
            hs.append(Mb.max(0).mean() - Mb.mean(1).max())
        lo, hi = np.quantile(hs, [0.025, 0.975])
        blue_reply = M.idxmax(0)  # Blue's best reply to each Red column
        red_reply = M.idxmin(1)  # Red's best reply to each Blue row
        p1 = bool((short > 0.05).all())
        summary.append(
            {
                "match-up": mu,
                "best single Blue row": best_row,
                "its mean": float(M.mean(1).max()),
                "near-dominant rows": ", ".join(short[short <= 0.05].index) or "none",
                "P1": p1,
                "P2 headroom (points)": 100 * head,
                "P2 95%": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
                "gate 1 (P1 and P2 >= 10)": bool(p1 and head >= 0.10),
                "timeouts": float(t.timeout.mean()),
            }
        )
        lines += [
            f"## {mu}: Blue win rate (rows Blue, columns Red), {N_CELL} paired seeds per cell",
            "",
            "```",
            M.round(3).to_string(),
            "```",
            "",
            "Blue's best reply to each Red strategy:",
            "",
            "```",
            blue_reply.to_string(),
            "```",
            "",
            "Red's best reply to each Blue strategy:",
            "",
            "```",
            red_reply.to_string(),
            "```",
            "",
        ]
    if summary:
        lines += ["## Read-outs", "", "```", pd.DataFrame(summary).round(3).to_string(index=False), "```", ""]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def readout(t, menu, rng, n_boot=1000):
    """P1, P2 headroom with a paired bootstrap interval, and best replies, for one table on `menu`."""
    t = t[t.blue.isin(menu) & t.red.isin(menu)]
    M = t.pivot_table(index="blue", columns="red", values="won", aggfunc="mean").reindex(
        index=menu, columns=menu
    )
    short = (M.max(0) - M).max(1)
    head = float(M.max(0).mean() - M.mean(1).max())
    W = t.pivot_table(index="seed", columns=["blue", "red"], values="won", aggfunc="first").astype(float)
    hs = []
    for _ in range(n_boot):
        Mb = W.iloc[rng.integers(0, len(W), len(W))].mean().unstack("red").reindex(index=menu, columns=menu)
        hs.append(Mb.max(0).mean() - Mb.mean(1).max())
    lo, hi = np.quantile(hs, [0.025, 0.975])
    p1 = bool((short > 0.05).all())
    return M, {
        "best single Blue row": M.mean(1).idxmax(),
        "its mean": float(M.mean(1).max()),
        "near-dominant rows": ", ".join(short[short <= 0.05].index) or "none",
        "P1": p1,
        "P2 headroom (points)": 100 * head,
        "P2 95%": f"[{100 * lo:.1f}, {100 * hi:.1f}]",
        "gate 1": bool(p1 and head >= 0.10),
        "timeouts": float(t.timeout.mean()),
    }


def analyse2():
    """docs/counters/02: the flank screen -> docs/counters/03_flank_screen.md."""
    rng = np.random.default_rng(0)
    lines = ["# Counters, step 2: the committed flank, payoff screens", "", "Plan: `02_flank_plan.md`.", ""]
    summary = []
    for mu in STAGE2:
        p = OUT / f"table_{mu}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        for name, menu in [
            ("full menu", MENU2),
            ("without flank", [m for m in MENU2 if m != "flank/nearest"]),
        ]:
            M, r = readout(t, menu, rng)
            summary.append({"match-up": mu, "menu": name, **r})
            if name == "full menu":
                T = t.pivot_table(index="blue", columns="red", values="timeout", aggfunc="mean")
                lines += [
                    f"## {mu}: Blue win rate (rows Blue, columns Red), {N_CELL} paired seeds per cell",
                    "",
                    "```",
                    M.round(3).to_string(),
                    "```",
                    "",
                    "Blue's best reply to each Red column / Red's best reply to each Blue row:",
                    "",
                    "```",
                    M.idxmax(0).to_string(),
                    "",
                    M.idxmin(1).to_string(),
                    "```",
                    "",
                    f"Timeout share per cell, max {T.values.max():.2f}.",
                    "",
                ]
    lines += ["## Read-outs", "", "```", pd.DataFrame(summary).round(3).to_string(index=False), "```", ""]
    doc = PROJECT_ROOT / "docs" / "counters" / "03_flank_screen.md"
    doc.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["calibrate", "table", "analyse", "analyse2"])
    p.add_argument("matchup", nargs="?")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "table":
        return table(a.matchup)
    {"calibrate": calibrate, "analyse": analyse, "analyse2": analyse2}[a.what]()


if __name__ == "__main__":
    main()
