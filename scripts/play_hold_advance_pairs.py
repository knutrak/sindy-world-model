"""Play the same close-battle starts with always hold/focus and always advance/focus (exp06, exploratory).

Seeds 6,000,000-6,001,999 (the planner test's seeds and more). Saves full team states per step,
plus the share of each side's alive units with a weapon ready, to
data/processed/exp06_intents/pairs_hold_advance.parquet. Used by sindyc_05_why_hold_wins.

    uv run scripts/play_hold_advance_pairs.py
"""

from multiprocessing import Pool

import pandas as pd

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentPolicy
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.paths import DATA_DIR
from sindy_wm.states.aggregate import level_a
from sindy_wm.states.geometry import MARINE_HP, REF_UNITS, geometry

FIRST_SEED, N = 6_000_000, 2000
OUT = DATA_DIR / "processed" / "exp06_intents" / "pairs_hold_advance.parquet"


def play(args) -> pd.DataFrame:
    seed, movement = args
    s = CloseBattleSampler().sample(seed, walkable_bounds(load_map_info("10m_vs_10m")))
    env = make_env("10m_vs_10m", step_mul=2, spawn=s.spawn, unit_counts=s.unit_counts)
    ep = play_episode(
        env, IntentPolicy(movement, "focus"), seed=seed, max_steps=480, start_health=s.start_health
    )
    snap = ep.snapshots
    states = level_a(snap).join(geometry(snap))
    states[["health_blue", "health_red"]] /= REF_UNITS * MARINE_HP
    states[["alive_blue", "alive_red", "reach_blue", "reach_red"]] /= REF_UNITS
    alive = snap[snap["alive"]]
    ready = alive.assign(r=alive["cooldown"] <= 1e-9).groupby(["step", "team"])["r"].mean().unstack("team")
    states["ready_blue"], states["ready_red"] = ready.get("blue"), ready.get("red")
    states["t"] = states.index * 0.125
    return states.assign(seed=seed, movement=movement, won=ep.battle_won)


if __name__ == "__main__":
    jobs = [(FIRST_SEED + k, m) for k in range(N) for m in ("hold", "advance")]
    with Pool(9) as pool:
        tables = pool.map(play, jobs, chunksize=20)
    pd.concat(tables).rename_axis("step").reset_index().to_parquet(OUT)
    print("saved", OUT)
