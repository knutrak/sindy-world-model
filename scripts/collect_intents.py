"""Collect exp06: exploratory datasets with team intents as control input (SINDYc experiments).

Blue follows a (movement, target) intent (`envs.intents`); battles are close
(`CloseBattleSampler`). Sets, each in its own folder under data/raw/exp06_intents/:

- pilot_<intent>   the same seeds under each fixed intent held for the first 4 s, then
                   advance/nearest: does the intent change the outcome? (leverage pilot)
- train_k          intents switch every 1-4 s: for fitting models with control (shards)
- val              same, new seeds: for choosing models
- cftrain_<intent> like cf_<intent>, more seeds: for learning which intent is best directly
- cf_<intent>      counterfactual test: a switching battle up to a drawn moment, then
                   <intent> for 4 s, then advance/nearest. Same seeds for every intent.

Exploratory: collected with allow_dirty. Run one set per process, e.g.

    uv run scripts/collect_intents.py pilot
    uv run scripts/collect_intents.py train --shard 0
"""

import argparse

from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import INTENTS, IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

EXPERIMENT_DIR = RAW_DIR / "exp06_intents"
SCENARIO, STEP_MUL, MAX_SECONDS = "10m_vs_10m", 2, 60
SAMPLER = CloseBattleSampler()
HOLD_SECONDS = 4.0  # how long a pilot / counterfactual intent is played before reverting to advance/nearest

N_PILOT, N_TRAIN_SHARD, N_SHARDS, N_VAL, N_CF, N_CFTRAIN = 300, 1000, 6, 1000, 600, 2000
SEEDS = {"pilot": 2_000_000, "train": 0, "val": 3_000_000, "cf": 4_000_000, "cftrain": 5_000_000}


def run(name: str, policy: IntentSchedule, first_seed: int, n: int, description: str, scale: float) -> None:
    dataset_dir = EXPERIMENT_DIR / name
    if (dataset_dir / "manifest.json").exists():
        print(f"Skipping {name}: already exists")
        return
    collect(
        dataset_dir=dataset_dir,
        policy=policy,
        description=description,
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=max(1, round(n * scale)),
        first_seed=first_seed,
        allow_dirty=True,
        sampler=SAMPLER,
    )
    print(f"done {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["pilot", "train", "val", "cf", "cftrain"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--intent", type=int, default=None, help="index into INTENTS (pilot, cf); default all"
    )
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    intents = INTENTS if args.intent is None else [INTENTS[args.intent]]
    if args.what == "pilot":
        for movement, target in intents:
            policy = IntentSchedule(branch_intent=(movement, target), branch_hold_seconds=HOLD_SECONDS)
            run(
                f"pilot_{movement}_{target}",
                policy,
                SEEDS["pilot"],
                N_PILOT,
                f"Leverage pilot: {movement}/{target} for the first {HOLD_SECONDS} s, then advance/nearest.",
                args.scale,
            )
    elif args.what == "cf":
        for movement, target in intents:
            policy = IntentSchedule(
                branch_intent=(movement, target), branch_seconds=(0.5, 8.0), branch_hold_seconds=HOLD_SECONDS
            )
            run(
                f"cf_{movement}_{target}",
                policy,
                SEEDS["cf"],
                N_CF,
                f"Counterfactual test: switching, then {movement}/{target} for {HOLD_SECONDS} s, "
                "then advance/nearest.",
                args.scale,
            )
    elif args.what == "cftrain":
        for movement, target in intents:
            policy = IntentSchedule(
                branch_intent=(movement, target), branch_seconds=(0.0, 8.0), branch_hold_seconds=HOLD_SECONDS
            )
            run(
                f"cftrain_{movement}_{target}",
                policy,
                SEEDS["cftrain"],
                N_CFTRAIN,
                f"Counterfactual training set: switching, then {movement}/{target} for {HOLD_SECONDS} s, "
                "then advance/nearest.",
                args.scale,
            )
    elif args.what == "train":
        run(
            f"train_{args.shard}",
            IntentSchedule(),
            SEEDS["train"] + args.shard * N_TRAIN_SHARD,
            N_TRAIN_SHARD,
            "Training: intents switch every 1-4 s.",
            args.scale,
        )
    else:
        run(
            "val",
            IntentSchedule(),
            SEEDS["val"],
            N_VAL,
            "Validation: intents switch every 1-4 s.",
            args.scale,
        )


if __name__ == "__main__":
    main()
