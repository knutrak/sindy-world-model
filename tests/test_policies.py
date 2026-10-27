"""
Do TargetChoicePolicy and RetreatPolicy pick the actions their rules describe?
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import (
    FIRST_TARGET,
    MOVE_EAST,
    MOVE_WEST,
    TARGET_RULES,
    AttackNearestPolicy,
    RetreatPolicy,
    TargetChoicePolicy,
)
from sindy_wm.envs.scenarios import make_env


def _unit(x, y, hp=45.0):
    return SimpleNamespace(pos=np.array([x, y], dtype=float), hp=hp, shield=0.0, max_hp=45.0, max_shield=0.0)


def _env(agents, enemies, in_range):
    """A stand-in for a SMAClite env: in_range[agent_id] lists the enemies that agent can shoot."""
    n_actions = FIRST_TARGET + len(enemies)

    def avail():
        masks = []
        for agent_id in range(len(agents)):
            mask = np.zeros(n_actions)
            mask[1:FIRST_TARGET] = 1  # stop and all four moves
            mask[[FIRST_TARGET + k for k in in_range.get(agent_id, [])]] = 1
            masks.append(mask)
        return masks

    return SimpleNamespace(unwrapped=SimpleNamespace(agents=agents, enemies=enemies, get_avail_actions=avail))


def _act(policy, env):
    policy.reset(seed=0)
    return policy.act(env)


def test_nearest_rule_reproduces_attack_nearest_policy():
    env = make_env("5m_vs_5m")
    a = play_episode(env, AttackNearestPolicy(aggression=0.8), seed=3, max_steps=60)
    b = play_episode(env, TargetChoicePolicy("nearest", aggression=0.8), seed=3, max_steps=60)
    pd.testing.assert_frame_equal(a.snapshots, b.snapshots)
    pd.testing.assert_frame_equal(a.actions, b.actions)


@pytest.mark.parametrize("rule", TARGET_RULES)
def test_retreat_policy_without_thresholds_equals_target_choice(rule):
    env = make_env("5m_vs_5m")
    a = play_episode(env, TargetChoicePolicy(rule, aggression=0.8), seed=5, max_steps=60)
    b = play_episode(env, RetreatPolicy(rule, aggression=0.8), seed=5, max_steps=60)
    pd.testing.assert_frame_equal(a.actions, b.actions)


def test_weakest_and_nearest_pick_different_targets():
    agents = {0: _unit(0, 0)}
    enemies = {0: _unit(1, 0, hp=45), 1: _unit(3, 0, hp=10)}  # 0 is nearer, 1 is weaker
    env = _env(agents, enemies, in_range={0: [0, 1]})
    assert _act(TargetChoicePolicy("nearest"), env) == [FIRST_TARGET + 0]
    assert _act(TargetChoicePolicy("weakest"), env) == [FIRST_TARGET + 1]


def test_focus_makes_every_unit_that_can_reach_it_shoot_the_same_enemy():
    agents = {0: _unit(0, 0), 1: _unit(0, 5), 2: _unit(0, 10)}
    enemies = {0: _unit(2, 0, hp=40), 1: _unit(2, 5, hp=20), 2: _unit(2, 10, hp=5)}
    # Agent 0 can't reach enemy 2 (the team's focus): it falls back to its own weakest in range
    env = _env(agents, enemies, in_range={0: [0, 1], 1: [1, 2], 2: [2]})
    assert _act(TargetChoicePolicy("focus"), env) == [FIRST_TARGET + 1, FIRST_TARGET + 2, FIRST_TARGET + 2]


def test_weak_unit_retreats_only_when_an_enemy_is_close():
    agents = {0: _unit(5, 0, hp=10), 1: _unit(5, 3)}  # unit 0 is at 22% health
    enemies = {0: _unit(8, 0)}  # east of both
    env = _env(agents, enemies, in_range={0: [0], 1: [0]})
    policy = RetreatPolicy(unit_below=0.3, danger_radius=6.0)
    assert _act(policy, env) == [MOVE_WEST, FIRST_TARGET + 0]  # weak one moves away; healthy one shoots

    enemies[0].pos = np.array([20.0, 0.0])  # far away: no danger, so it rejoins
    env = _env(agents, enemies, in_range={})
    assert _act(policy, env) == [MOVE_EAST, MOVE_EAST]


def test_team_retreats_when_losing():
    agents = {0: _unit(5, 0, hp=20), 1: _unit(5, 3, hp=20)}  # Blue 40 hp
    enemies = {0: _unit(8, 0), 1: _unit(8, 3)}  # Red 90 hp
    env = _env(agents, enemies, in_range={0: [0], 1: [1]})
    assert _act(RetreatPolicy(team_below=0.5), env) == [MOVE_WEST, MOVE_WEST]  # 40 < 0.5 * 90
    assert _act(RetreatPolicy(team_below=0.4), env) == [FIRST_TARGET + 0, FIRST_TARGET + 1]  # 40 > 36


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        TargetChoicePolicy("strongest")
    with pytest.raises(ValueError):
        RetreatPolicy(unit_below=1.5)
    with pytest.raises(ValueError):
        RetreatPolicy(team_below=-1)


def test_config_records_every_parameter():
    assert RetreatPolicy("focus", 0.9, unit_below=0.3, team_below=0.5).config() == {
        "class": "RetreatPolicy",
        "params": {
            "rule": "focus",
            "aggression": 0.9,
            "unit_below": 0.3,
            "team_below": 0.5,
            "danger_radius": 6.0,
        },
    }
