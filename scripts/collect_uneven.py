"""Collect exp10: marines + marauders with *different* mixes per side, a scenario the generalisation
pipeline (docs/generalisation/00_plan.md) was never tuned on. Held out: only read in the pre-registered
test (WP6). One side has 2/3 marauders, the other 1/4 (which side is drawn per battle). Same intents,
switching schedule and withdraw radius (8) as exp08. Collected from committed code only.

    uv run scripts/collect_uneven.py test
"""

import argparse

from sindy_wm.envs.episode_sampling import UnevenMixSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp10_uneven"
SCENARIO, STEP_MUL, MAX_SECONDS = "mixed_m_mrd", 2, 90
SETS = {"test": (12_000_000, 1000)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=list(SETS))
    args = parser.parse_args()
    first_seed, n = SETS[args.what]
    collect(
        dataset_dir=RAW / args.what,
        policy=IntentSchedule(withdraw_radius=8.0),
        description="exp10 test: marines + marauders, uneven mixes (2/3 vs 1/4 marauders); held out for the "
        "pre-registered generalisation test.",
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=n,
        first_seed=first_seed,
        allow_dirty=False,
        sampler=UnevenMixSampler(),
    )


if __name__ == "__main__":
    main()
