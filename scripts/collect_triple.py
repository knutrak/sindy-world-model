"""Collect exp12: marines + marauders + stalkers, the same mix on both sides (docs/triple/00_plan.md).

Same intents, switching schedule and withdraw radius (8) as exp08, on `mixed_m_mrd_s` with
`TripleMixSampler`. Small on purpose: train 3 x 500, val 500, test 500 (held out, read once).

    uv run scripts/collect_triple.py train --shard 0
    uv run scripts/collect_triple.py val
"""

import argparse

from sindy_wm.envs.episode_sampling import TripleMixSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp12_triple"
SCENARIO, STEP_MUL, MAX_SECONDS = "mixed_m_mrd_s", 2, 90
SETS = {"train": (30_000_000, 500), "val": (30_100_000, 500), "test": (30_900_000, 500)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=list(SETS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true", help="collect from uncommitted code")
    args = parser.parse_args()
    first_seed, n = SETS[args.what]
    name = f"train_{args.shard}" if args.what == "train" else args.what
    collect(
        dataset_dir=RAW / name,
        policy=IntentSchedule(withdraw_radius=8.0),
        description=f"exp12 {name}: marines + marauders + stalkers (same mix both sides), withdraw radius 8; "
        "intents switch every 1-4 s.",
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=n,
        first_seed=first_seed + (args.shard * n if args.what == "train" else 0),
        allow_dirty=args.allow_dirty,
        sampler=TripleMixSampler(),
    )


if __name__ == "__main__":
    main()
