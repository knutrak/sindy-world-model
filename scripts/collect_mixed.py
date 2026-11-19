"""Collect exp08: mixed marine / marauder battles, the same mix on both sides (branch mixed-marauders).

Same intents and switching schedule as exp06 (`collect_intents.py`), on `mixed_m_mrd` with
`MixedBattleSampler`: unit counts, health and spawn as `CloseBattleSampler`, then one marauder
share from {1/4, 1/3, 1/2} for both sides and a random order of the units in each formation.
Withdraw radius 8 (marauder reach against a marauder is 7.125). Collected from committed code only.

- pilot            300 battles, looked at before anything else (checkpoint 1)
- train_k          intents switch every 1-4 s (6 shards of 1000)
- val              same, new seeds
- cf_<intent>      counterfactual: switching up to a drawn moment, then <intent> for 4 s, then
                   advance/nearest. Same seeds for every intent (6 x 600)
- test3            held out like val; NOT read until a pre-registered evaluation

    uv run scripts/collect_mixed.py train --shard 0
    uv run scripts/collect_mixed.py cf --intent 2
    uv run scripts/collect_mixed.py build          per-type state tables -> data/processed/exp08_mixed/
    uv run scripts/collect_mixed.py build --max-reach 6    reach capped at 6 -> data/processed/exp08_mixed_r6/
"""

import argparse

import pandas as pd

from sindy_wm.envs.episode_sampling import MixedBattleSampler
from sindy_wm.envs.intents import INTENTS, IntentSchedule
from sindy_wm.io import storage
from sindy_wm.io.collect import collect
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.typed import load_typed_states

RAW, OUT = RAW_DIR / "exp08_mixed", DATA_DIR / "processed" / "exp08_mixed"
SCENARIO, STEP_MUL, MAX_SECONDS = "mixed_m_mrd", 2, 90
WITHDRAW_RADIUS = 8.0
HOLD_SECONDS = 4.0
TYPES = ("MARINE", "MARAUDER")
N_SHARDS = 6
SETS = {
    "pilot": (10_000_000, 300),
    "train": (10_100_000, 1000),  # per shard: seeds 10_100_000 + 1000 * shard
    "val": (10_500_000, 1000),
    "cf": (10_600_000, 600),
    "test3": (10_900_000, 1000),
}


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


def build(name: str, max_reach: float | None = None) -> pd.DataFrame:
    dataset_dir = RAW / name
    states = load_typed_states(dataset_dir, TYPES, max_reach=max_reach)
    controls = storage.load_controls(dataset_dir).set_index(["episode_idx", "step"])
    cols = [
        c for c in ("segment", "movement", "target", "branched", "reverted", "branch_at") if c in controls
    ]
    states = states.join(controls[cols])
    states[cols] = states.groupby(level="episode_idx")[cols].ffill()
    index = storage.load_episode_index(dataset_dir).set_index("episode_idx")
    keep = ["seed", "battle_won", "n_blue", "n_red"] + [
        c for c in index if c.startswith(("n_marine", "n_marauder"))
    ]
    return states.join(index[keep], on="episode_idx")


def build_all(max_reach: float | None = None) -> None:
    out = OUT if max_reach is None else OUT.with_name(f"{OUT.name}_r{max_reach:g}")
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(
        p.name
        for p in RAW.iterdir()
        if (p / "manifest.json").exists() and storage.load_manifest(p)["status"] == "complete"
    )
    names = [n for n in names if n != "test3"]  # held out
    shards = [n for n in names if n.startswith("train_")]
    if len(shards) == N_SHARDS and not (out / "train.parquet").exists():
        parts = []
        for name in shards:
            t = build(name, max_reach).reset_index()
            t["episode_idx"] += int(name.split("_")[1]) * 1000
            parts.append(t.set_index(["episode_idx", "step"]))
        pd.concat(parts).to_parquet(out / "train.parquet")
        print("built train", flush=True)
    for name in names:
        if not name.startswith("train_") and not (out / f"{name}.parquet").exists():
            build(name, max_reach).to_parquet(out / f"{name}.parquet")
            print("built", name, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=[*SETS, "build"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--intent", type=int, default=None, help="index into INTENTS (cf); default all")
    parser.add_argument(
        "--max-reach",
        type=float,
        default=None,
        help="build: cap every pair reach (e.g. 6) -> exp08_mixed_r6/",
    )
    args = parser.parse_args()
    if args.what == "build":
        build_all(args.max_reach)
        return
    first_seed, n = SETS[args.what]
    base = f"marines + marauders (same mix both sides), withdraw radius {WITHDRAW_RADIUS}"
    if args.what == "train":
        run(
            f"train_{args.shard}",
            IntentSchedule(withdraw_radius=WITHDRAW_RADIUS),
            first_seed + args.shard * n,
            n,
            f"exp08 train shard {args.shard}: {base}; intents switch every 1-4 s.",
        )
    elif args.what == "cf":
        for movement, target in INTENTS if args.intent is None else [INTENTS[args.intent]]:
            policy = IntentSchedule(
                withdraw_radius=WITHDRAW_RADIUS,
                branch_intent=(movement, target),
                branch_seconds=(0.5, 8.0),
                branch_hold_seconds=HOLD_SECONDS,
            )
            run(
                f"cf_{movement}_{target}",
                policy,
                first_seed,
                n,
                f"exp08 counterfactual: {base}; switching, then {movement}/{target} for {HOLD_SECONDS} s, "
                "then advance/nearest.",
            )
    else:
        run(
            args.what,
            IntentSchedule(withdraw_radius=WITHDRAW_RADIUS),
            first_seed,
            n,
            f"exp08 {args.what}: {base}; intents switch every 1-4 s."
            + (" Held-out test: not read until a pre-registered evaluation." if args.what == "test3" else ""),
        )


if __name__ == "__main__":
    main()
