"""Collect exp07: the exp06 intent battles with stalkers instead of marines (generalisation test).

Same intents, switching schedule and close-battle sampler as exp06 (`collect_intents.py`), on
10 stalkers a side (`configs/scenarios/10s_vs_10s.json`). The withdraw radius is scaled with the
reach (7.0 for marines, reach 5.75; stalkers reach 6 + 2 x 0.625 = 7.25). Exploratory
(allow_dirty). Builds the state tables afterwards.

    uv run scripts/collect_stalkers.py train
    uv run scripts/collect_stalkers.py val
    uv run scripts/collect_stalkers.py build
"""

import argparse

import pandas as pd

from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io import storage
from sindy_wm.io.collect import collect
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.geometry import load_team_states

RAW, OUT = RAW_DIR / "exp07_stalkers", DATA_DIR / "processed" / "exp07_stalkers"
SCENARIO, STEP_MUL, MAX_SECONDS = "10s_vs_10s", 2, 90
STALKER_REACH, STALKER_HEALTH = 6 + 2 * 0.625, 80 + 80
WITHDRAW_RADIUS = 7.0 * STALKER_REACH / 5.75
SETS = {"train": (7_000_000, 1500), "val": (7_500_000, 300)}


def build(name: str) -> None:
    dataset_dir = RAW / name
    states = load_team_states(dataset_dir, reach=STALKER_REACH, unit_health=STALKER_HEALTH)
    controls = storage.load_controls(dataset_dir).set_index(["episode_idx", "step"])
    cols = ["segment", "movement", "target"]
    states = states.join(controls[cols])
    states[cols] = states.groupby(level="episode_idx")[cols].ffill()
    index = storage.load_episode_index(dataset_dir).set_index("episode_idx")
    states = states.join(index[["seed", "battle_won", "n_blue", "n_red"]], on="episode_idx")
    OUT.mkdir(parents=True, exist_ok=True)
    states.to_parquet(OUT / f"{name}.parquet")
    print("built", name, len(states), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=[*SETS, "build"])
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.what == "build":
        for name in SETS:
            build(name)
        return
    first_seed, n = SETS[args.what]
    collect(
        dataset_dir=RAW / args.what,
        policy=IntentSchedule(withdraw_radius=WITHDRAW_RADIUS),
        description=f"exp07 {args.what}: stalkers, intents switch every 1-4 s (withdraw radius "
        f"{WITHDRAW_RADIUS:.2f}).",
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=max(1, round(n * args.scale)),
        first_seed=first_seed,
        allow_dirty=True,
        sampler=CloseBattleSampler(),
    )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
