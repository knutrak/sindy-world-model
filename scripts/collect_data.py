"""Collect the datasets listed in DATASETS. Existing datasets are skipped."""

from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy, RandomPolicy
from sindy_wm.paths import RAW_DIR

N_EPISODES = 100
STEP_MUL = 8
MAX_STEPS = 200
FIRST_SEED = 0
ALLOW_DIRTY = True  # fine while exploring; set False for thesis datasets

DATASETS = [  # (dataset name, scenario, policy, description)
    ("5m_attack_a090_v1", "5m_vs_5m", AttackNearestPolicy(aggression=0.9), "Baseline, aggression 0.9"),
    ("5m_random_v1", "5m_vs_5m", RandomPolicy(), "Random actions, for contrast"),
]


def main() -> None:

    for dataset_name, scenario, policy, description in DATASETS:
        if (RAW_DIR / dataset_name / "manifest.json").exists():
            print(f"Skipping {dataset_name}: already exists")
            continue

        collect(
            dataset_name=dataset_name,
            policy=policy,
            description=description,
            scenario=scenario,
            step_mul=STEP_MUL,
            max_steps=MAX_STEPS,
            n_episodes=N_EPISODES,
            first_seed=FIRST_SEED,
            allow_dirty=ALLOW_DIRTY,
        )


if __name__ == "__main__":
    main()
