"""Collect the datasets of one experiment. Existing datasets are skipped."""

from itertools import product
from pathlib import Path

from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.paths import RAW_DIR

EXPERIMENT = "exp02_5m_random_position_aggression_stepmul"
EXPERIMENT_DESCRIPTION = (
    "How Blue's aggression and decision interval (step_mul) affect outcomes and "
    "battle dynamics in 5m_vs_5m, with AttackNearestPolicy against Red's built-in AI. "
    "I am also randomizing the initial positions of the units to get more varied battles."
)

EXPERIMENT_DIR = RAW_DIR / EXPERIMENT

SCENARIO = "5m_vs_5m"
STEP_MULS = [1, 2, 4, 8]
AGGRESSIONS = [0.5, 0.7, 0.9, 1.0]
MAX_SECONDS = 100  # battle time limit in game seconds, converted to steps per step_mul
N_EPISODES = 100
FIRST_SEED = 0
ALLOW_DIRTY = True  # fine while exploring; set False for thesis datasets

DATASETS = [
    {
        "name": f"5m_attack_a{round(a * 100):03d}_s{s}_v1",
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
        )


if __name__ == "__main__":
    main()
