"""Collect the datasets listed in DATASETS. Existing datasets are skipped."""

from itertools import product

from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.paths import RAW_DIR

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


def max_steps_for(step_mul: int) -> int:
    """Number of steps that gives MAX_SECONDS of game time (16 ticks per second)."""
    return MAX_SECONDS * 16 // step_mul


def main() -> None:

    for ds in DATASETS:
        if (RAW_DIR / ds["name"] / "manifest.json").exists():
            print(f"Skipping {ds['name']}: already exists")
            continue
        collect(
            dataset_name=ds["name"],
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
