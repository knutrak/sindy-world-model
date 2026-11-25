"""Collect exp13: marines with one unit property changed (docs/properties/00_plan.md).

Both sides have the same unit (configs/units/MARINE_<variant>.json). Sampler, intents, switching schedule,
withdraw radius (7) and step as exp06; the same 1000 seeds for every variant. Committed code only.

    uv run scripts/collect_properties.py V0
    uv run scripts/collect_properties.py --timing      times 20 battles per variant, saves nothing
"""

import argparse
import time

from sindy_wm.envs import units
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp13_properties"
STEP_MUL, MAX_SECONDS, N = 2, 90, 1000
FIRST_SEED, TIMING_SEED = 31_000_000, 31_800_000
VARIANTS = ("V0", "DM", "DP", "CM", "CP", "HM", "HP", "A1", "A3", "SM", "SP", "R4", "R3")


def timing(n: int = 20) -> None:
    sampler = CloseBattleSampler()
    for v in VARIANTS:
        scenario = f"10m_{v.lower()}"
        bounds = walkable_bounds(load_map_info(scenario))
        policy, t0 = IntentSchedule(withdraw_radius=7.0), time.time()
        for seed in range(TIMING_SEED, TIMING_SEED + n):
            s = sampler.sample(seed, bounds)
            env = make_env(scenario, step_mul=STEP_MUL, spawn=s.spawn, unit_counts=s.unit_counts)
            play_episode(
                env,
                policy,
                seed=seed,
                max_steps=MAX_SECONDS * 16 // STEP_MUL,
                start_health=s.start_health,
            )
        print(f"{v}: {(time.time() - t0) / n:.2f} s per battle", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.timing:
        return timing()
    if args.variant is None:
        parser.error("give a variant or --timing")
    s = units.stats(f"MARINE_{args.variant}")
    collect(
        dataset_dir=RAW / args.variant.lower(),
        policy=IntentSchedule(withdraw_radius=7.0),
        description=f"exp13 {args.variant}: marines on both sides with hp {s.hp}, armor {s.armor}, damage "
        f"{s.damage}, cooldown {s.cooldown}, speed {s.speed}, range {s.attack_range}; "
        "intents switch every 1-4 s.",
        scenario=f"10m_{args.variant.lower()}",
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=N,
        first_seed=FIRST_SEED,
        allow_dirty=False,
        sampler=CloseBattleSampler(),
    )


if __name__ == "__main__":
    main()
