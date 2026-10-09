"""Collecting datasets of episodes: play episodes with a policy and save them."""

from pathlib import Path

import numpy as np

from sindy_wm.data import storage
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import Policy
from sindy_wm.envs.scenarios import load_map_info, make_env, seconds_per_step, walkable_bounds
from sindy_wm.envs.spawn_sampling import sample_opposing_spawns
from sindy_wm.paths import SCENARIO_DIR
from sindy_wm.provenance import file_sha256


def collect(
    dataset_dir: Path,
    policy: Policy,
    description: str,
    scenario: str = "5m_vs_5m",
    step_mul: int = 8,
    max_steps: int = 200,
    n_episodes: int = 100,
    first_seed: int = 0,
    allow_dirty: bool = False,
    randomize_spawn: bool = False,
    spawn_distance_range: tuple[float, float] = (8.0, 16.0),
) -> None:
    """Play n_episodes with consecutive seeds and save them as a new dataset in dataset_dir.

    Args:
        dataset_dir: Folder to create the dataset in.
        policy: Policy choosing Blue's actions.
        description: Free text: why this dataset was collected.
        scenario: Name of a built-in map, or one of configs/scenarios/.
        step_mul: Game ticks per env.step (see `envs.scenarios.make_env`).
        max_steps: Episodes are truncated after this many env.step calls.
        n_episodes: Number of episodes to play.
        first_seed: Seeds used are first_seed, first_seed + 1, ...
        allow_dirty: Passed to `storage.create_dataset`.
        randomize_spawn: If True, sample each episode's ally/enemy starting
            positions instead of using the scenario's fixed ones (see
            `envs.spawn_sampling.sample_opposing_spawns`), using the same seed
            as the env and the policy. Both teams keep their own formation;
            only the distance and bearing between them vary. Needs a scenario
            with exactly one ALLY and one ENEMY group. The realized positions
            end up in each episode's step-0 snapshot, so nothing extra needs
            to be recorded per episode.
        spawn_distance_range: (min, max) distance sampled between the two
            group centers, in map units, when randomize_spawn is True.
    """
    seeds = list(range(first_seed, first_seed + n_episodes))
    env = None if randomize_spawn else make_env(scenario, step_mul=step_mul)
    bounds = walkable_bounds(load_map_info(scenario)) if randomize_spawn else None

    config = dataset_config(
        scenario, step_mul, policy, seeds, max_steps, randomize_spawn, spawn_distance_range
    )
    storage.create_dataset(dataset_dir, config=config, description=description, allow_dirty=allow_dirty)
    for idx, seed in enumerate(seeds):
        if randomize_spawn:
            # Same seed as the env and the policy, so one number still describes
            # the whole episode (see the docstring above).
            spawn = sample_opposing_spawns(np.random.default_rng(seed), bounds, *spawn_distance_range)
            episode_env = make_env(scenario, step_mul=step_mul, spawn=spawn)
        else:
            episode_env = env
        episode = play_episode(episode_env, policy, seed=seed, max_steps=max_steps)
        storage.save_episode(dataset_dir, idx, episode)
    storage.finalize_dataset(dataset_dir)


def dataset_config(
    scenario,
    step_mul,
    policy,
    seeds,
    max_steps,
    randomize_spawn: bool = False,
    spawn_distance_range: tuple[float, float] | None = None,
) -> dict:
    """Everything needed to regenerate a dataset, for its manifest."""
    scenario_info = {"name": scenario}
    scenario_file = SCENARIO_DIR / f"{scenario}.json"
    if scenario_file.exists():  # custom scenario: record its contents
        scenario_info["sha256"] = file_sha256(scenario_file)
    if randomize_spawn:
        scenario_info["spawn_randomization"] = {
            "distance_range": list(spawn_distance_range),
            "note": (
                "ally/enemy group centers resampled per episode from "
                "np.random.default_rng(seed), the same seed as the env and the "
                "policy; realized positions are in each episode's step-0 snapshot"
            ),
        }
    return {
        "scenario": scenario_info,
        "env": {
            "step_mul": step_mul,
            "seconds_per_step": seconds_per_step(step_mul),
            "max_steps": max_steps,
            "use_cpp_rvo2": False,
        },
        "policy": policy.config(),
        "seeds": {"first": seeds[0], "last": seeds[-1]},
    }
