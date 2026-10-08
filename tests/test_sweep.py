"""
Does load_all_indexes combine datasets correctly, and are the summary stats sane?
"""

import numpy as np
import pandas as pd
import pytest

from sindy_wm.data import storage
from sindy_wm.evaluation.sweep import load_all_indexes, summarize, wilson_interval


def _dataset(tmp_path, name, step_mul, aggression, outcomes):
    # outcomes: list of (seed, n_steps, battle_won) to fake as saved episodes
    d = tmp_path / name
    storage.create_dataset(
        d,
        config={
            "env": {"step_mul": step_mul, "seconds_per_step": step_mul / 16},
            "policy": {"class": "AttackNearestPolicy", "params": {"aggression": aggression}},
            "seeds": [seed for seed, *_ in outcomes],
        },
        allow_dirty=True,
    )
    for idx, (seed, n_steps, battle_won) in enumerate(outcomes):
        snaps = pd.DataFrame(
            {
                "step": [0, n_steps - 1],
                "team": ["blue", "blue"],
                "unit_id": [0, 0],
                "hp": [45.0, 45.0 if battle_won else 0.0],
                "alive": [True, battle_won],
            }
        )
        acts = pd.DataFrame({"step": [0], "unit_id": [0], "action": [6]})
        episode = type(
            "FakeEpisode",
            (),
            {
                "snapshots": snaps,
                "actions": acts,
                "seed": seed,
                "n_steps": n_steps,
                "battle_won": battle_won,
                "truncated": False,
            },
        )()
        storage.save_episode(d, idx, episode)
    storage.finalize_dataset(d)
    return d


def test_load_all_indexes_skips_incomplete_and_combines_params(tmp_path):
    _dataset(tmp_path, "a", step_mul=1, aggression=0.5, outcomes=[(0, 10, True), (1, 20, False)])
    _dataset(tmp_path, "b", step_mul=2, aggression=0.9, outcomes=[(0, 5, True)])
    storage.create_dataset(
        tmp_path / "in_progress_ds",
        config={"env": {"step_mul": 1, "seconds_per_step": 1 / 16}},
        allow_dirty=True,
    )

    episodes = load_all_indexes(tmp_path)
    assert set(episodes["dataset"]) == {"a", "b"}
    assert episodes[episodes["dataset"] == "a"]["step_mul"].unique().tolist() == [1]
    assert episodes[episodes["dataset"] == "b"]["aggression"].unique().tolist() == [0.9]
    # duration_s = n_steps * seconds_per_step
    row = episodes[(episodes["dataset"] == "a") & (episodes["seed"] == 0)].iloc[0]
    assert row["duration_s"] == pytest.approx(10 * (1 / 16))


def test_load_all_indexes_raises_if_nothing_complete(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_indexes(tmp_path)


def test_wilson_interval_contains_point_estimate():
    low, high = wilson_interval(wins=7, n=10)
    assert low < 0.7 < high


def test_wilson_interval_handles_zero_n():
    low, high = wilson_interval(wins=0, n=0)
    assert np.isnan(low) and np.isnan(high)


def test_summarize_counts_distinct_outcomes():
    group = pd.DataFrame(
        {
            "battle_won": [True, True, False],
            "duration_s": [1.0, 1.0, 2.0],
            "n_steps": [10, 10, 20],
            "truncated": [False, False, True],
            "final_hp_blue": [50, 50, 0],
            "final_hp_red": [0, 0, 60],
        }
    )
    s = summarize(group)
    assert s["n"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["distinct_outcomes"] == 2  # the two identical winning rows collapse to one
