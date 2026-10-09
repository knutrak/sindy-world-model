"""Collect the datasets of one experiment. Existing datasets are skipped."""

from itertools import product
from pathlib import Path

from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.paths import RAW_DIR

EXPERIMENT = "exp03_10m_random_aggression_stepmul"
EXPERIMENT_DESCRIPTION = (
    "How Blue's aggression and decision interval (step_mul) affect outcomes "
    "and battle dynamics in 10m_vs_10m, with AttackNearestPolicy against Red's built-in AI."
)

EXPERIMENT_DIR = RAW_DIR / EXPERIMENT

SCENARIO = "10m_vs_10m"
STEP_MULS = [1, 2, 4, 8]
AGGRESSIONS = [0.5, 0.7, 0.9, 1.0]
MAX_SECONDS = 500  # battle time limit in game seconds, converted to steps per step_mul
N_EPISODES = 500
FIRST_SEED = 0
ALLOW_DIRTY = True  # fine while exploring; set False for thesis datasets

RANDOMIZE_SPAWN = True
SPAWN_DISTANCE_RANGE = (8.0, 16.0)  # map units; original 5m_vs_5m distance is 14

DATASETS = [
    {
        "name": f"10m_attack_a{round(a * 100):03d}_s{s}_v1",
        "scenario": SCENARIO,
        "step_mul": s,
        "policy": AttackNearestPolicy(aggression=a),
        "description": f"AttackNearest, aggression {a}, step_mul {s}",
    }
    for s, a in product(STEP_MULS, AGGRESSIONS)
]


def dataset_exists(dataset_dir: Path) -> bool:
    """True if the dataset has been started (complete or not)."""
    return (dataset_dir / "manifest.json").exists()


def max_steps_for(step_mul: int) -> int:
    """Number of steps that gives MAX_SECONDS of game time (16 ticks per second)."""
    return MAX_SECONDS * 16 // step_mul


def write_experiment_readme() -> None:
    """Write a short README for the experiment folder, unless one exists."""
    readme = EXPERIMENT_DIR / "README.md"
    if readme.exists():
        return
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {EXPERIMENT}\n\n{EXPERIMENT_DESCRIPTION}\n\n"
        f"- Scenario: {SCENARIO}\n"
        f"- step_mul: {STEP_MULS}\n"
        f"- aggression: {AGGRESSIONS}\n"
        f"- randomized spawn: distance {SPAWN_DISTANCE_RANGE[0]}-{SPAWN_DISTANCE_RANGE[1]} "
        f"map units, random bearing, resampled per episode\n"
        f"- {N_EPISODES} episodes per dataset, seeds from {FIRST_SEED}\n",
        encoding="utf-8",
    )


def main() -> None:
    write_experiment_readme()
    for ds in DATASETS:
        dataset_dir = EXPERIMENT_DIR / ds["name"]
        if dataset_exists(dataset_dir):
            print(f"Skipping {ds['name']}: already exists")
            continue

        collect(
            dataset_dir=dataset_dir,
            policy=ds["policy"],
            description=ds["description"],
            scenario=ds["scenario"],
            step_mul=ds["step_mul"],
            max_steps=max_steps_for(ds["step_mul"]),
            n_episodes=N_EPISODES,
            first_seed=FIRST_SEED,
            allow_dirty=ALLOW_DIRTY,
            randomize_spawn=RANDOMIZE_SPAWN,
            spawn_distance_range=SPAWN_DISTANCE_RANGE,
        )


if __name__ == "__main__":
    main()
