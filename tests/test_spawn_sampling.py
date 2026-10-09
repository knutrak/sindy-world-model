"""
Are sampled spawn positions inside bounds, at the right distance, and reproducible?
"""

import numpy as np
import pytest

from sindy_wm.envs.spawn_sampling import sample_opposing_spawns

BOUNDS = (0.0, 31.0, 8.0, 23.0)  # 5m_vs_5m's walkable area


def test_distance_and_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        spawn = sample_opposing_spawns(rng, BOUNDS, min_distance=8.0, max_distance=16.0, margin=1.5)
        ally, enemy = np.array(spawn.ally_pos), np.array(spawn.enemy_pos)
        distance = np.linalg.norm(ally - enemy)
        assert 8.0 - 1e-6 <= distance <= 16.0 + 1e-6
        for pos in (ally, enemy):
            assert BOUNDS[0] + 1.5 <= pos[0] <= BOUNDS[1] - 1.5
            assert BOUNDS[2] + 1.5 <= pos[1] <= BOUNDS[3] - 1.5
    assert spawn.attack_point == spawn.ally_pos


def test_reproducible():
    a = sample_opposing_spawns(np.random.default_rng(42), BOUNDS, 8.0, 16.0)
    b = sample_opposing_spawns(np.random.default_rng(42), BOUNDS, 8.0, 16.0)
    assert a == b


def test_too_tight_margin_raises():
    with pytest.raises(ValueError):
        sample_opposing_spawns(np.random.default_rng(0), BOUNDS, 8.0, 16.0, margin=20.0)


def test_min_distance_too_large_for_bounds_raises():
    # BOUNDS is 32x16 after the default margin; min_distance can't exceed twice the short axis
    with pytest.raises(ValueError):
        sample_opposing_spawns(np.random.default_rng(0), BOUNDS, min_distance=13.0, max_distance=16.0)


def test_bearing_is_not_restricted_to_one_side():
    rng = np.random.default_rng(1)
    deltas = []
    for _ in range(200):
        spawn = sample_opposing_spawns(rng, BOUNDS, 8.0, 10.0)
        deltas.append(np.array(spawn.enemy_pos) - np.array(spawn.ally_pos))
    deltas = np.array(deltas)
    assert (deltas[:, 0] > 0).any() and (deltas[:, 0] < 0).any()
    assert (deltas[:, 1] > 0).any() and (deltas[:, 1] < 0).any()
