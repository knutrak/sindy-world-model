"""
Does the same seed give the same game?
"""

import pandas as pd

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.envs.scenarios import make_env


def test_same_seed_same_episode():
    env = make_env("5m_vs_5m")
    a = play_episode(env, AttackNearestPolicy(aggression=0.8), seed=3, max_steps=50)
    b = play_episode(env, AttackNearestPolicy(aggression=0.8), seed=3, max_steps=50)
    pd.testing.assert_frame_equal(a.snapshots, b.snapshots)
    pd.testing.assert_frame_equal(a.actions, b.actions)
