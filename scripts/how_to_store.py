from sindy_wm.data import storage
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.envs.scenarios import SCENARIO_DIR, make_env
from sindy_wm.paths import RAW_DIR
from sindy_wm.provenance import file_sha256

# 1. Choose the configuration
scenario, step_mul, max_steps, aggression = "5m_vs_5m", 8, 200, 0.8
seeds = list(range(10))
dataset_dir = RAW_DIR / "5m_attack_a080_v1"

env = make_env(scenario, step_mul=step_mul)
policy = AttackNearestPolicy(aggression=aggression)

# 2. Create the dataset: writes manifest.json, refuses if the folder exists
storage.create_dataset(
    dataset_dir,
    config={
        "scenario": {"name": scenario, "sha256": file_sha256(SCENARIO_DIR / f"{scenario}.json")},
        "env": {"step_mul": step_mul, "seconds_per_step": env.seconds_per_step, "max_steps": max_steps},
        "policy": {"class": "AttackNearestPolicy", "params": {"aggression": aggression}},
        "seeds": {"first": seeds[0], "last": seeds[-1]},
    },
    description="Small test dataset",
    allow_dirty=True,  # fine while exploring; drop it for thesis datasets
)

# 3. Play and save each episode: snapshots, actions, and one line in episodes.jsonl
for idx, seed in enumerate(seeds):
    episode = play_episode(env, policy, seed=seed, max_steps=max_steps)
    storage.save_episode(dataset_dir, idx, episode)

# 4. Check all seeds are saved and mark the dataset complete
storage.finalize_dataset(dataset_dir)
