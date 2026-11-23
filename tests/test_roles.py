"""Per-type roles (docs/planning/48): both types advancing with nearest targets plays exactly as
advance/nearest; a priority picks only enemies of that type when one is in reach; the oracle accepts role
candidates."""

import numpy as np

from sindy_wm.envs.episode_sampling import MixedBattleSampler
from sindy_wm.envs.intents import IntentPolicy
from sindy_wm.envs.planner import OraclePlannerPolicy
from sindy_wm.envs.roles import ROLE_NAMES, RolePolicy
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds


def _env(seed):
    s = MixedBattleSampler().sample(seed, walkable_bounds(load_map_info("mixed_m_mrd")))
    env = make_env("mixed_m_mrd", step_mul=2, spawn=s.spawn, unit_types=s.unit_types)
    env.reset(seed=seed)
    return env


def test_advance_nearest_role_equals_the_intent():
    e1, e2 = _env(4), _env(4)
    a, b = (
        RolePolicy("advance", "advance", "nearest"),
        IntentPolicy("advance", "nearest", withdraw_radius=8.0),
    )
    a.reset(4)
    b.reset(4)
    for _ in range(80):
        x, y = a.act(e1), b.act(e2)
        assert x == y
        r1, _ = e1.step(x), e2.step(y)
        if r1[2] or r1[3]:
            break


def test_priority_targets_the_type_when_in_reach():
    env = _env(6)
    pol = RolePolicy("advance", "advance", "marauder")
    pol.reset(6)
    checked = 0
    for _ in range(80):
        g = env.unwrapped
        acts = pol.act(env)
        masks = g.get_avail_actions()
        for i, a in enumerate(acts):
            u = g.agents.get(i)
            if u is None or a < 6:
                continue
            reach_mrd = [
                k
                for k in np.flatnonzero(masks[i][6:])
                if g.enemies[int(k)].type.name.upper() == "MARAUDER"
                and u.has_within_attack_range(g.enemies[int(k)])
            ]
            if reach_mrd:
                assert g.enemies[a - 6].type.name.upper() == "MARAUDER"
                checked += 1
        _, _, term, trunc, _ = env.step(acts)
        if term or trunc:
            break
    assert checked > 0


def test_oracle_accepts_role_candidates():
    env = _env(8)
    o = OraclePlannerPolicy(
        commit_seconds=2.0, horizon_seconds=1.0, candidates=ROLE_NAMES[:3], withdraw_radius=8.0
    )
    o.reset(8)
    o.act(env)
    assert len(o.score_log[0]) == 3
