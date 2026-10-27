"""
Per-episode sampling: unit counts, starting health, switching policies, and how collect() records them.
"""

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.episode_sampling import EpisodeSampler
from sindy_wm.envs.policies import TARGET_RULES, PolicySpace, SwitchingPolicy
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.io import storage
from sindy_wm.io.collect import collect

BOUNDS = walkable_bounds(load_map_info("10m_vs_10m"))


def test_unit_counts_and_start_health():
    env = make_env("10m_vs_10m", step_mul=4, unit_counts=(7, 12))
    episode = play_episode(env, SwitchingPolicy(), seed=0, max_steps=3, start_health=(1.0, 0.5))
    first = episode.snapshots[episode.snapshots["step"] == 0]
    assert first.groupby("team").size().to_dict() == {"blue": 7, "red": 12}
    assert (first.loc[first["team"] == "blue", "hp"] == 45.0).all()
    assert (first.loc[first["team"] == "red", "hp"] == 22.5).all()


def test_policy_space_draws_within_its_ranges():
    space = PolicySpace(unit_retreat_share=0.5, team_retreat_share=0.5)
    rng = np.random.default_rng(0)
    draws = pd.DataFrame([space.sample(rng).params() for _ in range(300)])
    assert set(draws["rule"]) == set(TARGET_RULES)
    assert draws["aggression"].between(*space.aggression).all()
    retreating = draws["unit_below"] > 0
    assert 0.35 < retreating.mean() < 0.65
    assert draws.loc[retreating, "unit_below"].between(*space.unit_below).all()


def test_switching_policy_logs_a_setting_per_step_and_switches_within_its_durations():
    env = make_env("10m_vs_10m", step_mul=2)  # 0.125 s per step
    policy = SwitchingPolicy(segment_seconds=(1.0, 2.0))
    episode = play_episode(env, policy, seed=4, max_steps=200)
    controls = episode.controls
    assert list(controls["step"]) == list(range(episode.n_steps))

    lengths = controls.groupby("segment").size()
    assert len(lengths) > 1
    assert lengths.iloc[:-1].between(8, 16).all()  # 1-2 s; the last segment may be cut short
    per_segment = controls.groupby("segment")[["rule", "aggression", "unit_below", "team_below"]].nunique()
    assert (per_segment == 1).all().all()  # one setting per segment

    again = play_episode(env, SwitchingPolicy(segment_seconds=(1.0, 2.0)), seed=4, max_steps=200)
    pd.testing.assert_frame_equal(controls, again.controls)
    pd.testing.assert_frame_equal(episode.actions, again.actions)


def test_episode_sampler_is_deterministic_and_keeps_formations_apart():
    sampler = EpisodeSampler(red_units=(4, 16), force_ratio=(0.75, 1.33), spawn_distance=(2.0, 16.0))
    assert sampler.sample(3, BOUNDS) == sampler.sample(3, BOUNDS)
    ratios = []
    for seed in range(30):
        settings = sampler.sample(seed, BOUNDS)
        n_blue, n_red = settings.unit_counts
        assert 4 <= n_red <= 16
        assert round(n_red * 0.75) <= n_blue <= round(n_red * 1.33)  # within the ratio range, after rounding
        ratios.append(n_blue / n_red)
        assert all(0.4 <= h <= 1.0 for h in settings.start_health)
        env = make_env("10m_vs_10m", unit_counts=settings.unit_counts, spawn=settings.spawn)
        env.reset(seed=seed)
        game = env.unwrapped
        gap = min(np.linalg.norm(a.pos - e.pos) for a in game.agents.values() for e in game.enemies.values())
        assert gap > 0.75  # more than one marine's width between the closest opposing units
    assert min(ratios) < 0.9 and max(ratios) > 1.1  # both sides get the upper hand


def test_collect_with_sampler_records_settings_and_controls(tmp_path):
    dataset = tmp_path / "sampled"
    collect(
        dataset,
        policy=SwitchingPolicy(),
        description="test",
        scenario="10m_vs_10m",
        step_mul=8,
        max_steps=40,
        n_episodes=3,
        allow_dirty=True,
        sampler=EpisodeSampler(),
    )
    index = storage.load_episode_index(dataset)
    assert {"n_blue", "n_red", "start_health_blue", "start_health_red", "spawn_distance"} <= set(
        index.columns
    )

    snaps = storage.load_snapshots(dataset)
    first = snaps[snaps["step"] == 0].groupby(["episode_idx", "team"]).size().unstack()
    assert (first["blue"].to_numpy() == index["n_blue"].to_numpy()).all()
    assert (first["red"].to_numpy() == index["n_red"].to_numpy()).all()

    controls = storage.load_controls(dataset)
    assert len(controls) == index["n_steps"].sum()
    assert "episode_sampler" in storage.load_manifest(dataset)["config"]["scenario"]


def test_collect_rejects_sampler_with_randomize_spawn(tmp_path):
    with pytest.raises(ValueError):
        collect(
            tmp_path / "x",
            SwitchingPolicy(),
            "test",
            scenario="10m_vs_10m",
            n_episodes=1,
            allow_dirty=True,
            randomize_spawn=True,
            sampler=EpisodeSampler(),
        )
