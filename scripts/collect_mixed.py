"""Collect exp08: mixed marine / marauder battles, the same mix on both sides (branch mixed-marauders).

Same intents and switching schedule as exp06 (`collect_intents.py`), on `mixed_m_mrd` with
`MixedBattleSampler`: unit counts, health and spawn as `CloseBattleSampler`, then one marauder
share from {1/4, 1/3, 1/2} for both sides and a random order of the units in each formation.
Withdraw radius 8 (marauder reach against a marauder is 7.125). Collected from committed code only.

    uv run scripts/collect_mixed.py pilot          300 battles, looked at before anything else
"""

import argparse

from sindy_wm.envs.episode_sampling import MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp08_mixed"
SCENARIO, STEP_MUL, MAX_SECONDS = "mixed_m_mrd", 2, 90
WITHDRAW_RADIUS = 8.0
SETS = {"pilot": (10_000_000, 300)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=list(SETS))
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    first_seed, n = SETS[args.what]
    collect(
        dataset_dir=RAW / args.what,
        policy=IntentSchedule(withdraw_radius=WITHDRAW_RADIUS),
        description=f"exp08 {args.what}: marines + marauders (same mix both sides), intents switch every "
        f"1-4 s (withdraw radius {WITHDRAW_RADIUS}).",
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=max(1, round(n * args.scale)),
        first_seed=first_seed,
        allow_dirty=False,
        sampler=MixedBattleSampler(),
    )


if __name__ == "__main__":
    main()
