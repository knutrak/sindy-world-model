"""Collect exp09: the exp06 intent battles with marauders instead of marines (step 1b, mixed-marauders).

A single-type stepping stone before mixed marine / marauder battles: does the marine model
transfer to marauders when scaled without refitting, and does A*-2k's recipe find the same
structure when refitted? Same intents, switching schedule and close-battle sampler as exp06
(`collect_intents.py`), on marauders (`configs/scenarios/10mrd_vs_10mrd.json`). Marauder
against marauder: reach 6 + 2 x 0.5625 = 7.125; damage 10 + 10 (vs armored) - 1 armor = 19 per
1.07 s; 125 hp. The withdraw radius is scaled with the reach as for stalkers (7.0 x 7.125 / 5.75).
Collected from committed code only (allow_dirty=False).

    uv run scripts/collect_marauders.py train     (and val, test)
    uv run scripts/collect_marauders.py build
"""

import argparse

from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io import storage
from sindy_wm.io.collect import collect
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.geometry import load_team_states

RAW, OUT = RAW_DIR / "exp09_marauders", DATA_DIR / "processed" / "exp09_marauders"
SCENARIO, STEP_MUL, MAX_SECONDS = "10mrd_vs_10mrd", 2, 90
MARAUDER_REACH, MARAUDER_HEALTH = 6 + 2 * 0.5625, 125.0
WITHDRAW_RADIUS = 7.0 * MARAUDER_REACH / 5.75
SETS = {
    "train": (11_000_000, 1500),
    "val": (11_500_000, 500),
    "test": (11_800_000, 500),  # held out: not read until a pre-registered evaluation
}


def build(name: str) -> None:
    dataset_dir = RAW / name
    states = load_team_states(dataset_dir, reach=MARAUDER_REACH, unit_health=MARAUDER_HEALTH)
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
            if (
                name != "test"
                and (RAW / name / "manifest.json").exists()
                and not (OUT / f"{name}.parquet").exists()
            ):
                build(name)
        return
    first_seed, n = SETS[args.what]
    collect(
        dataset_dir=RAW / args.what,
        policy=IntentSchedule(withdraw_radius=WITHDRAW_RADIUS),
        description=f"exp09 {args.what}: marauders, intents switch every 1-4 s (withdraw radius "
        f"{WITHDRAW_RADIUS:.2f}).",
        scenario=SCENARIO,
        step_mul=STEP_MUL,
        max_steps=MAX_SECONDS * 16 // STEP_MUL,
        n_episodes=max(1, round(n * args.scale)),
        first_seed=first_seed,
        allow_dirty=False,
        sampler=CloseBattleSampler(unit_size=1.125),
    )


if __name__ == "__main__":
    main()
