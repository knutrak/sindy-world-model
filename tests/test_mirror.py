"""Mirrored Red (docs/planning/40): masks follow SMAClite's agent rules, a standing Red never moves, and a
Red whose policy does nothing plays exactly as the scripted attack-move."""

import numpy as np

from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentPolicy
from sindy_wm.envs.mirror import MirrorRed, _MirrorEnv
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds


def _env(seed):
    s = CloseBattleSampler().sample(seed, walkable_bounds(load_map_info("10m_vs_10m")))
    env = make_env("10m_vs_10m", step_mul=2, spawn=s.spawn, unit_counts=s.unit_counts)
    env.reset(seed=seed)
    return env


class _Noop:
    def reset(self, seed=None):
        pass

    def act(self, env):
        return [0] * env.unwrapped._game.n_enemies


def _play(env, blue, red, steps=80):
    track = []
    for _ in range(steps):
        if red is not None:
            red.apply(env)
        _, _, term, trunc, _ = env.step(blue.act(env))
        g = env.unwrapped
        track.append(
            [(i, *u.pos, u.hp) for i, u in sorted(g.enemies.items())]
            + [(i, *u.pos, u.hp) for i, u in sorted(g.agents.items())]
        )
        if term or trunc:
            break
    return track


def test_masks_follow_the_agent_rules():
    env = _env(1)
    blue = IntentPolicy("advance", "nearest", withdraw_radius=7.0)
    blue.reset(1)
    for _ in range(30):
        env.step(blue.act(env))
    g = env.unwrapped
    masks = _MirrorEnv(env).unwrapped.get_avail_actions()
    for i, unit in g.enemies.items():
        for k, target in g.agents.items():
            assert masks[i][6 + k] == float(np.linalg.norm(target.pos - unit.pos) <= 6.0 + 1e-9)


def test_noop_mirror_equals_scripted_red():
    blue = IntentPolicy("advance", "focus", withdraw_radius=7.0)
    blue.reset(2)
    a = _play(_env(2), blue, None)
    blue.reset(2)
    b = _play(_env(2), blue, MirrorRed(_Noop()))
    assert a == b


def test_standing_red_never_moves():
    env = _env(3)
    blue = IntentPolicy("advance", "nearest", withdraw_radius=7.0)
    blue.reset(3)
    red = MirrorRed(IntentPolicy("stand", "nearest", withdraw_radius=7.0))
    red.reset(3)
    start = {i: u.pos.copy() for i, u in env.unwrapped.enemies.items()}
    for _ in range(60):
        red.apply(env)
        _, _, term, trunc, _ = env.step(blue.act(env))
        for i, u in env.unwrapped.enemies.items():
            np.testing.assert_allclose(u.pos, start[i], atol=1e-6)
        if term or trunc:
            break
