"""
Are dead units kept?
"""

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.envs.scenarios import make_env


def test_dead_units_kept_with_zero_hp():
    env = make_env("5m_vs_5m")
    ep = play_episode(env, AttackNearestPolicy(aggression=1.0), seed=0, max_steps=200)
    snaps = ep.snapshots

    # Every step lists the full roster of 10 units
    assert (snaps.groupby("step").size() == 10).all()
    # Dead units have hp 0, and at least one unit died
    dead = snaps[~snaps["alive"]]
    assert len(dead) > 0
    assert (dead["hp"] == 0).all()
