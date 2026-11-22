"""Collect exp11: mixed battles with kite in the intent schedule (docs/planning/32_kite_refit_plan.md).

As exp08 (`collect_mixed.py`): `mixed_m_mrd`, `MixedBattleSampler`, withdraw radius 8, intents switch every
1-4 s; the movements are advance / hold / withdraw / kite with weights 0.35 / 0.20 / 0.15 / 0.30.

- train_k          4 shards of 500, seeds 20 000 000 + 500 k
- val              300, seeds 20 100 000 +
- cf_kite_<target> exp08's counterfactual design on exp08's cf seeds (10 600 000 +, 600): exp08's
                   switching up to the drawn moment, then kite/<target> for 4 s, then advance/nearest.
                   Pairs with exp08's cf sets.

    uv run scripts/collect_kite.py train --shard 0
    uv run scripts/collect_kite.py val
    uv run scripts/collect_kite.py cf --target focus
"""

import argparse

from sindy_wm.envs.episode_sampling import MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp11_kite"
SCENARIO, STEP_MUL, MAX_SECONDS = "mixed_m_mrd", 2, 90
WITHDRAW_RADIUS = 8.0
WEIGHTS = (0.35, 0.20, 0.15, 0.30)
SETS = {"train": (20_000_000, 500), "val": (20_100_000, 300), "cf": (10_600_000, 600)}


def run(name: str, policy: IntentSchedule, first_seed: int, n: int, description: str) -> None:
    if (RAW / name / "manifest.json").exists():
        print(f"Skipping {name}: already exists")
        return
    collect(
        dataset_dir=RAW / name,
        policy=policy,
        description=description,
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=n,
        first_seed=first_seed,
        allow_dirty=False,
        sampler=MixedBattleSampler(),
    )
    print(f"done {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=list(SETS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--target", choices=["nearest", "focus"], default="focus")
    args = parser.parse_args()
    first, n = SETS[args.what]
    base = f"marines + marauders (same mix both sides), withdraw radius {WITHDRAW_RADIUS}"
    if args.what == "cf":
        policy = IntentSchedule(
            withdraw_radius=WITHDRAW_RADIUS,
            branch_intent=("kite", args.target),
            branch_seconds=(0.5, 8.0),
            branch_hold_seconds=4.0,
        )
        run(
            f"cf_kite_{args.target}",
            policy,
            first,
            n,
            f"exp11 counterfactual on exp08's cf seeds: {base}; switching, then kite/{args.target} for 4 s, "
            "then advance/nearest.",
        )
        return
    policy = IntentSchedule(withdraw_radius=WITHDRAW_RADIUS, movement_weights=WEIGHTS)
    if args.what == "train":
        run(
            f"train_{args.shard}",
            policy,
            first + args.shard * n,
            n,
            f"exp11 train shard {args.shard}: {base}; intents (with kite, weights {WEIGHTS}) "
            "switch every 1-4 s.",
        )
    else:
        run("val", policy, first, n, f"exp11 val: {base}; as train, new seeds.")


if __name__ == "__main__":
    main()
