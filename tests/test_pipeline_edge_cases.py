"""Edge cases found by the stress tests of 25 Nov (docs/triple/04_stress.md): defects that were fixed, and
properties the stress tests confirmed."""

import copy

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode_sampling import TripleMixSampler
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger
from sindy_wm.models.intent_model import DT, stlsq
from sindy_wm.models.pipeline import Layout, Pipeline, check_rows, check_steps, prepare
from sindy_wm.states.typed import typed_states
from sindy_wm.world_model import INTENTS, WorldModel, parse_intent, plan_controls

SIDES = ("blue", "red")


# ---- stlsq
def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 2))
    return rng, np.column_stack([x, np.ones(n)])


def test_stlsq_keeps_a_constant_and_does_not_bias_the_other_terms():
    rng, T = _data()
    y = -0.18 * T[:, 0] - 0.1 + rng.normal(0, 0.01, len(T))
    c = stlsq(T, y, 0.15)
    assert c == pytest.approx([-0.18, 0, -0.1], abs=2e-3)


def test_stlsq_returns_nothing_when_nothing_reaches_the_threshold():
    rng, T = _data()
    assert (stlsq(T, rng.normal(0, 1, len(T)), 0.15) == 0).all()


def test_stlsq_ignores_absent_columns_and_a_zero_target():
    rng, T = _data()
    T = np.column_stack([T, np.zeros(len(T))])
    y = -0.18 * T[:, 0] + rng.normal(0, 0.01, len(T))
    assert stlsq(T, y, 0.15) == pytest.approx([-0.18, 0, 0, 0], abs=2e-3)
    assert (stlsq(T, np.zeros(len(T)), 0.15) == 0).all()


# ---- tables
def _table(n_battles=2, steps=20, dt=DT):
    idx = pd.MultiIndex.from_product([range(n_battles), range(steps)], names=["episode_idx", "step"])
    return pd.DataFrame({"t": idx.get_level_values(1) * dt}, index=idx)


def test_wrong_time_step_is_refused():
    check_steps(_table())
    check_steps(_table().drop(columns="t"))
    with pytest.raises(ValueError, match="0.125 s apart"):
        check_steps(_table(dt=0.25))
    with pytest.raises(ValueError, match="0.125 s apart"):
        prepare(_table(dt=0.25), Layout(("marine",)))


def test_two_battles_under_one_id_and_unsorted_rows_are_refused():
    t = _table().reset_index()
    t["episode_idx"] = 0
    with pytest.raises(ValueError, match="sharing an episode id"):
        check_steps(t.set_index(["episode_idx", "step"]))
    with pytest.raises(ValueError, match="unsorted"):
        check_steps(_table().iloc[::-1])


def test_shards_merged_with_colliding_ids_are_refused():
    """The collision a shard merge makes: battles 0, 1, 0, 1, whose duplicates are not next to each other."""
    merged = pd.concat([_table(), _table()])
    with pytest.raises(ValueError, match="share an id"):
        check_steps(merged)
    with pytest.raises(ValueError, match="share an id"):
        Layout(("marine",)).batch(merged)


def _rows(n=60):
    r = {"movement": ["advance", "hold", "withdraw"] * (n // 3), "distance": 6.0, "d_distance": -1.0}
    for s in SIDES:
        r |= {f"depth_{s}": 1.0, f"width_{s}": 1.0}
        for k in ("health", "alive", "reach", "d_health", "d_alive"):
            r[f"{k}_{s}_marine"] = 0.5
    return pd.DataFrame(r)


def test_fit_refuses_data_it_cannot_use_with_a_clear_message():
    L, rows = Layout(("marine",)), _rows()
    check_rows(rows, L)
    cases = {
        "no rows": rows.iloc[:0],
        "movement 'hold'": rows[rows["movement"] != "hold"],
        "movement 'advance'": rows[rows["movement"] != "advance"],
        "red never has a living marine": rows.assign(alive_red_marine=0.0),
        "ever has an enemy in reach": rows.assign(reach_blue_marine=0.0),
        "infinite values in": rows.assign(distance=np.inf),
        "'d_health_red_marine'": rows.assign(d_health_red_marine=np.nan),
    }
    for message, bad in cases.items():
        with pytest.raises(ValueError, match=message):
            check_rows(bad, L)
    with pytest.raises(ValueError, match="no rows"):
        Pipeline(("MARINE",), alloc="rule", movement_wd="rule", withdraw_radius=7.0).fit(rows.iloc[:0])


def test_unit_types_the_state_does_not_know_are_refused():
    units = pd.DataFrame(
        [
            {"step": 0, "team": s, "unit_id": i, "unit_type": t, "x": x, "y": float(i), "hp": 40.0}
            for s, x in (("blue", 0.0), ("red", 8.0))
            for i, t in enumerate(("MARINE", "STALKER"))
        ]
    ).assign(shield=0.0, alive=True, target_team=None, target_id=-1, target_in_range=False)
    with pytest.raises(ValueError, match="STALKER"):
        typed_states(units, ("MARINE", "MARAUDER"))


# ---- three types
def test_three_types_need_the_proportional_rule():
    with pytest.raises(NotImplementedError):
        Pipeline(("MARINE", "MARAUDER", "STALKER"), alloc="closure")


def test_three_type_rollout_with_an_absent_type_equals_the_two_type_model():
    """The three-type code path, given the mixed model's numbers and no stalkers, is the mixed model."""
    two = WorldModel.load("mixed").pipeline
    m = Pipeline(
        ("MARINE", "MARAUDER", "STALKER"),
        alloc="rule",
        movement_wd="rule",
        withdraw_radius=two.withdraw_radius,
    )
    m.rho = two.rho | {"stalker": 6.0}
    zero = {k: 0.0 for k in two.coef[("blue", "marine")]}
    m.coef = two.coef | {(s, "stalker"): zero for s in SIDES}
    m.p = two.p | {(s, "stalker"): (0.5, 0.0) for s in SIDES}
    m.share = two.share | {(s, "stalker"): np.ones(3) for s in SIDES}
    m.alloc_b, m.P_team = {}, two.P_team
    m.dist_names, m.dist_coef = two.dist_names, two.dist_coef
    rng = np.random.default_rng(0)
    n = 50
    A = rng.uniform(0, 1, (n, 4))
    A[rng.random((n, 4)) < 0.2] = 0
    H, F, reach = A * rng.uniform(0.1, 1, (n, 4)), rng.normal(0, 0.5, (n, 4)), A * rng.uniform(0, 1, (n, 4))

    def three(v):
        z = np.zeros((n, 1))
        return np.column_stack([v[:, :2], z, v[:, 2:], z])

    d, U = rng.uniform(1, 12, n), np.stack([plan_controls(INTENTS[i % 5], 4.0) for i in range(n)])
    common = {"shape0": rng.uniform(0.1, 2, (n, 4)), "U": U}
    a = two.rollout({"X0": np.column_stack([H, A, d]), "front0": F, "reach0": reach} | common)
    b = m.rollout(
        {"X0": np.column_stack([three(H), three(A), d]), "front0": three(F), "reach0": three(reach)} | common
    )
    np.testing.assert_allclose(m.L.team_view(b), two.L.team_view(a), atol=1e-12)


def test_triple_sampler_and_scenario_play():
    s = TripleMixSampler()
    for n in range(1, 25):
        for mix in s.mixes:
            k = s.counts(n, mix)
            assert sum(k) == n and (n < 3 or min(k) >= 1)
    bounds = walkable_bounds(load_map_info("mixed_m_mrd_s"))
    a = s.sample(5, bounds)
    assert a == s.sample(5, bounds) and set(a.unit_types[0]) == set(s.types) == set(a.unit_types[1])
    env = make_env("mixed_m_mrd_s", step_mul=2, spawn=a.spawn, unit_types=a.unit_types)
    env.reset(seed=5)
    snap = SnapshotLogger(env).start_episode()
    assert snap.loc[snap["unit_type"] == "STALKER", "shield"].max() == 80
    row = typed_states(snap, s.types, max_reach=6.0).iloc[0]
    assert row[[f"alive_blue_{t.lower()}" for t in s.types]].sum() * 10 == pytest.approx(a.unit_counts[0])


# ---- rollouts: properties under random and extreme states
@pytest.mark.parametrize("name", WorldModel.available())
@pytest.mark.parametrize("scale, dmax", [(1.5, 16), (50, 16), (1e-6, 16), (1.5, 0.1), (1.5, 1000)])
def test_rollouts_stay_physical(name, scale, dmax):
    m = WorldModel.load(name).pipeline
    k, n, rng = 2 * m.L.n, 200, np.random.default_rng(1)
    A = rng.uniform(0, scale, (n, k))
    A[rng.random((n, k)) < 0.2] = 0
    H = A * rng.uniform(0.02, 1, (n, k))
    B = {
        "X0": np.column_stack([H, A, rng.uniform(0, dmax, n)]),
        "shape0": rng.uniform(0, 3, (n, 4)),
        "front0": rng.uniform(-2, 2, (n, k)),
        "reach0": A * rng.uniform(0, 1, (n, k)),
        "U": np.stack([plan_controls(INTENTS[i % 5], 8.0) for i in range(n)]),
    }
    before = copy.deepcopy(B)
    with np.errstate(all="ignore"):
        Y = m.rollout(B)
    assert np.isfinite(Y).all() and (Y >= 0).all()
    assert (np.diff(Y[..., : 2 * k], axis=1) <= 1e-12 * max(scale, 1)).all()  # no healing, no new units
    assert all(np.array_equal(B[c], before[c]) for c in B)
    one = m.rollout({c: v[7:8] for c, v in B.items()})
    np.testing.assert_array_equal(one[0], Y[7])


# ---- interface
def _units():
    return [
        {"team": t, "unit_type": "MARINE", "x": x + 0.8 * (i % 3), "y": float(i), "hp": 45.0}
        for t, x in (("blue", 0.0), ("red", 8.0))
        for i in range(5)
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"hp": float("nan")},
        {"hp": float("inf")},
        {"hp": 4500.0},
        {"hp": "many"},
        {"x": float("inf")},
        {"x": None},
        {"shield": -1.0},
        {"shield": 100.0},
        {"team": "Blue"},
        {"unit_type": None},
        {"alive": None},
    ],
)
def test_interface_refuses_bad_units(change):
    units = _units()
    units[0] = units[0] | change
    if "alive" in change:
        units = [u | {"alive": u.get("alive", True)} for u in units]
    with pytest.raises(ValueError):
        WorldModel.load("marines").state_from_units(units)


def test_interface_fills_missing_shields_and_reads_number_text():
    wm, units = WorldModel.load("marines"), _units()
    units[0] = units[0] | {"shield": 0.0}
    assert wm.state_from_units(units).hp("blue") == pytest.approx(225)
    assert wm.state_from_units([u | {"hp": "45"} for u in _units()]).hp("red") == pytest.approx(225)


@pytest.mark.parametrize(
    "plan, horizon",
    [
        (None, 2),
        (5, 2),
        ([("hold", "focus")], None),
        (("hold", 2.0), 2),
        ([("hold", float("nan"))], None),
        ("hold", float("inf")),
        ("hold", float("nan")),
        ("hold", "2"),
        ("hold", -1),
        ([("hold", 1.0, "x")], None),
        ([], 2),
    ],
)
def test_interface_refuses_bad_plans(plan, horizon):
    wm = WorldModel.load("marines")
    with pytest.raises(ValueError):
        wm.predict(wm.state_from_units(_units()), plan, horizon)


def test_interface_refuses_other_misuse():
    wm = WorldModel.load("marines")
    s = wm.state_from_units(_units())
    for call in (
        lambda: wm.compare(s, []),
        lambda: wm.compare(s, commit=0),
        lambda: wm.predict("state", "hold", 1),
        lambda: parse_intent(3),
        lambda: wm.state_from_units([]),
    ):
        with pytest.raises(ValueError):
            call()
