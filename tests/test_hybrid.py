"""Unit-level hybrid world model (docs/generalisation/16): a rollout starts at the live state, and with
nobody in range and nobody walking, nothing changes."""

import numpy as np

from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.models.hybrid import HybridWorld, state_from_game


def _always(walk):
    return lambda X: np.full(len(X), walk)


def test_rollout_starts_at_the_live_state_and_stopped_units_stay_put():
    s = CloseBattleSampler().sample(2, walkable_bounds(load_map_info("10m_vs_10m")))
    env = make_env("10m_vs_10m", step_mul=2, spawn=s.spawn, unit_counts=s.unit_counts)
    env.reset(seed=2)
    st = state_from_game(env.unwrapped)
    sw = {(side, "MARINE"): (_always(False), ["1"]) for side in ("blue", "red")}
    w = HybridWorld(sw, {("MARINE", "MARINE"): -0.18}, wd=7.0)
    r = w.rollout(st, [("advance", "nearest")] * 8)
    assert np.isclose(r["hp_blue"][0], st["hp"][st["blue"]].sum())
    assert np.isclose(r["hp_red"][0], st["hp"][~st["blue"]].sum())
    assert len(r["reach_blue"]) == 9
    if (
        r["reach_blue"][0] == 0 and r["reach_red"][0] == 0
    ):  # nobody in range and nobody walks: nothing changes
        assert np.allclose(r["hp_blue"], r["hp_blue"][0]) and np.allclose(r["hp_red"], r["hp_red"][0])


def test_separation_removes_overlap():
    from sindy_wm.models.hybrid import separate

    pos = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [5.0, 5.0]])
    out = separate(pos, np.ones(4, bool), np.full(4, 0.375), passes=50)
    d = np.linalg.norm(out[:, None] - out[None], axis=-1) + np.eye(4) * 10
    assert d.min() >= 0.75 - 1e-3
    np.testing.assert_allclose(out[3], pos[3])
