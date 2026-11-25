"""The world-model interface (`sindy_wm.world_model`) gives exactly the frozen pipeline's numbers,
in game units, and refuses input the models were not fitted for."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode_sampling import CloseBattleSampler, MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.planner import CANDIDATES
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger
from sindy_wm.models.intent_model import DT, control_vector
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT
from sindy_wm.states.typed import typed_states
from sindy_wm.world_model import INTENTS, WorldModel, parse_intent, plan_controls

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
CASES = {
    "marines": ("10m_vs_10m", CloseBattleSampler()),
    "marauders": ("10mrd_vs_10mrd", CloseBattleSampler(unit_size=1.125)),
    "mixed": ("mixed_m_mrd", MixedBattleSampler()),
}
MARINES = [
    {"team": team, "unit_type": "MARINE", "x": x, "y": 10.0 + i, "hp": 45.0}
    for team, x in (("blue", 10.0), ("red", 18.0))
    for i in range(5)
]
BLOCKS = [u | {"x": u["x"] + 1.2 * (i % 3)} for i, u in enumerate(MARINES)]  # not a single flat line


def _battle(name, seed=3):
    mapname, sampler = CASES[name]
    s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        mapname, step_mul=2, spawn=s.spawn, unit_counts=None if types else s.unit_counts, unit_types=types
    )
    env.reset(seed=seed)
    return env


def _played(name, seed=3, steps=400):
    """Steps of a played battle with both sides alive: (env, the snapshot record so far)."""
    env = _battle(name, seed)
    record = SnapshotLogger(env)
    snaps = [record.start_episode()]
    driver = IntentSchedule(withdraw_radius=7.0)
    driver.reset(seed)
    for k in range(1, steps):
        _, _, term, trunc, _ = env.step(driver.act(env))
        snaps.append(record.snapshot(step=k))
        if term or trunc:
            return
        if k % 8 == 0 and env.unwrapped.agents and env.unwrapped.enemies:
            yield env, k, snaps


def test_available_and_repr():
    assert WorldModel.available() == ["marauders", "marines", "mixed"]
    assert "marines" in repr(WorldModel.load("marines"))
    with pytest.raises(FileNotFoundError, match="available"):
        WorldModel.load("zealots")


@pytest.mark.parametrize("name", WorldModel.available())
def test_equations_are_the_documented_ones(name):
    doc = (PROJECT_ROOT / "docs" / "generalisation" / "06_frozen_equations.md").read_text()
    assert WorldModel.load(name).equations() in doc


@pytest.mark.parametrize("name", WorldModel.available())
def test_save_load_round_trip_is_exact(name, tmp_path):
    wm = WorldModel.load(name)
    wm.save(tmp_path / "copy.json")
    again = WorldModel.load(tmp_path / "copy.json")
    again.save(tmp_path / "copy2.json")
    assert (tmp_path / "copy.json").read_text() == (tmp_path / "copy2.json").read_text()


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen models not built")
@pytest.mark.parametrize("name", list(CASES))
def test_exported_model_rolls_out_exactly_as_the_frozen_one(name):
    """Same state, same controls: the JSON model and the original pickle give bitwise-equal rollouts."""
    frozen, wm = pickle.load(open(FROZEN, "rb"))[name], WorldModel.load(name)
    checked = 0
    for env, _, _ in _played(name):
        state = wm.observe(env)
        for intent in INTENTS:
            U = plan_controls(intent, 4.0)
            L = frozen.L
            B = {
                k: np.nan_to_num(state.row[list(c)].to_numpy(float))[None]
                for k, c in (
                    ("X0", L.x_cols),
                    ("shape0", ["depth_blue", "depth_red", "width_blue", "width_red"]),
                    ("front0", L.front_cols),
                    ("reach0", L.r_cols),
                )
            }
            Y = L.team_view(frozen.rollout(B | {"U": U[None]}))[0]
            tr = wm.predict(state, intent, horizon=4.0).trajectory
            scale = 10 * L.hp[L.types[0]]
            np.testing.assert_allclose(tr["hp_blue"], Y[:, 0] * scale, rtol=1e-12)
            np.testing.assert_allclose(tr["hp_red"], Y[:, 1] * scale, rtol=1e-12)
            np.testing.assert_allclose(tr["units_blue"], Y[:, 2] * 10, rtol=1e-12)
            np.testing.assert_allclose(tr["units_red"], Y[:, 3] * 10, rtol=1e-12)
            np.testing.assert_array_equal(tr["distance"], Y[:, 4])
            checked += 1
    assert checked >= 25


@pytest.mark.parametrize("name", list(CASES))
def test_observed_state_is_the_recorded_table_to_the_end_of_the_battle(name):
    """Also after units (or a whole type) have died, although `observe` never sees the dead units."""
    wm = WorldModel.load(name)
    L, checked = wm.pipeline.L, 0
    cols = [*L.x_cols, *L.r_cols, *L.front_cols, "depth_blue", "depth_red", "width_blue", "width_red"]
    deaths = False
    for env, k, snaps in _played(name):
        table = typed_states(
            pd.concat(snaps[-1:]), tuple(t.upper() for t in L.types), max_reach=wm.pipeline.max_reach
        )
        got = np.nan_to_num(wm.observe(env).row[cols].to_numpy(float))
        np.testing.assert_allclose(got, np.nan_to_num(table.loc[k, cols].to_numpy(float)), atol=1e-12)
        deaths |= bool((~snaps[-1]["alive"]).any())
        checked += 1
    assert checked > 5 and deaths


@pytest.mark.parametrize("name", list(CASES))
def test_compare_ranks_as_the_planner(name):
    """compare(horizon=3, commit=2) scores the candidates exactly as `PipelinePlannerPolicy` does."""
    wm = WorldModel.load(name)
    planner = wm.planner()
    planner.reset(3)
    assert planner.withdraw_radius == wm.pipeline.withdraw_radius
    assert tuple("/".join(c) for c in CANDIDATES) == INTENTS
    scale = 10 * wm.hp_per_unit[wm.types[0]]
    checked = 0
    for env, _, _ in _played(name):
        table = wm.compare(wm.observe(env), horizon=3.0, commit=2.0).set_index("plan")
        scores = planner.predict(env) * scale
        np.testing.assert_allclose(table.loc[list(INTENTS), "margin"], scores, rtol=1e-9, atol=1e-9)
        assert table["margin"].is_monotonic_decreasing and table["behind_best"].iloc[0] == 0
        checked += 1
    assert checked > 5


def test_state_is_in_game_units():
    s = WorldModel.load("marines").state_from_units(MARINES)
    assert s.hp("blue") == pytest.approx(225) and s.units("red") == pytest.approx(5)
    assert s.distance == pytest.approx(8) and s.in_reach("blue") == 0
    assert "5 marine; 225 hp" in s.to_text() and s.to_dict()["blue"]["units"] == {"marine": pytest.approx(5)}


def test_mixed_model_accepts_a_side_with_one_type():
    s = WorldModel.load("mixed").state_from_units(MARINES)
    p = WorldModel.load("mixed").predict(s, "advance", 2.0)
    assert np.isfinite(p.trajectory.to_numpy()).all() and p.end["units_blue_marauder"] == 0


def test_dead_units_are_ignored_and_order_does_not_matter():
    wm = WorldModel.load("marines")
    dead = {"team": "red", "unit_type": "marine", "x": float("nan"), "y": float("nan"), "hp": 0.0}
    a, b = wm.state_from_units(MARINES), wm.state_from_units([dead, *reversed(MARINES)])
    pd.testing.assert_series_equal(a.row, b.row)


def test_predictions_start_at_the_state_and_behave():
    wm = WorldModel.load("marines")
    s = wm.state_from_units(MARINES)
    for intent in INTENTS:
        p = wm.predict(s, intent, horizon=2.0)
        assert p.horizon == 2.0 and len(p.trajectory) == 17
        assert p.start["hp_blue"] == pytest.approx(s.hp("blue"))
        assert (p.trajectory.drop(columns=["t"]) >= 0).all().all()
        assert (np.diff(p.trajectory["hp_red"]) <= 1e-9).all()
    assert wm.predict(s, "advance/focus", 2.0).end["distance"] < s.distance


def test_stand_off_is_flagged():
    """Two flat lines: the closing gate stops them just outside reach (a limit of the frozen model)."""
    wm = WorldModel.load("marines")
    flat, blocks = wm.state_from_units(MARINES), wm.state_from_units(BLOCKS)
    p = wm.predict(flat, "advance", 2.0)
    assert p.end["hp_red"] == pytest.approx(225) and "stand-off" in p.to_text()
    assert not wm.predict(blocks, "advance", 2.0).warnings
    assert wm.predict(blocks, "advance", 2.0).end["hp_red"] < 225


# margin after withdraw 1 s + hold/focus 2 s, Red hp after 3 s of advance/nearest; needs no data/
PINNED = {
    "marauders": (-17.118515934515415, 113.66572740854787),
    "marines": (6.675216497068703, 142.97535262529865),
    "mixed": (5.801939554216062, 147.93714646236697),
}


@pytest.mark.parametrize("name", list(PINNED))
def test_pinned_predictions(name):
    wm = WorldModel.load(name)
    t = "MARAUDER" if name == "marauders" else "MARINE"
    s = wm.state_from_units([u | {"unit_type": t} for u in BLOCKS])
    a = wm.predict(s, [("withdraw", 1.0), ("hold/focus", 2.0)]).margin
    b = wm.predict(s, "advance/nearest", 3.0).end["hp_red"]
    assert (a, b) == pytest.approx(PINNED[name], rel=1e-9)


def test_plans():
    U = plan_controls([("withdraw/nearest", 1.0), ("advance/focus", 0.5)], None)
    assert len(U) == round(1.5 / DT)
    np.testing.assert_array_equal(U[0], control_vector("withdraw", "nearest"))
    np.testing.assert_array_equal(U[-1], control_vector("advance", "focus"))
    longer = plan_controls([("withdraw", 1.0), ("advance/focus", 0.5)], 3.0)
    assert len(longer) == 24 and (longer[-1] == control_vector("advance", "focus")).all()
    assert len(plan_controls([("withdraw", 1.0), ("advance", 2.0)], 1.0)) == 8
    wm = WorldModel.load("marines")
    s = wm.state_from_units(MARINES)
    same = wm.predict(s, [("hold/focus", 1.0), ("hold/focus", 1.0)]).trajectory
    pd.testing.assert_frame_equal(same, wm.predict(s, ("hold", "focus"), 2.0).trajectory)
    mixed = wm.predict_many(s, [[("hold", 1.0)], [("hold", 2.0)]])
    assert [p.horizon for p in mixed] == [1.0, 2.0]


def test_horizon_warning_and_typical_error():
    wm = WorldModel.load("marines")
    s = wm.state_from_units(BLOCKS)
    ok, far = wm.predict(s, "hold", 2.0), wm.predict(s, "hold", 6.0)
    assert not ok.warnings and "typical error" in ok.to_text()
    assert ok.typical_error["hp"] == pytest.approx(0.027464153113809658 * 450)
    assert "past the 2 s" in far.to_text() and far.typical_error is None


@pytest.mark.parametrize("bad", ["kite/nearest", "hold/weakest", "hold/focus/now", ("hold",) * 3, ""])
def test_unknown_intents_are_refused(bad):
    with pytest.raises(ValueError, match="unknown intent"):
        parse_intent(bad)


def test_bad_states_and_plans_are_refused():
    wm = WorldModel.load("marines")
    s = wm.state_from_units(MARINES)
    with pytest.raises(ValueError, match="battle is over"):
        wm.state_from_units([u for u in MARINES if u["team"] == "blue"])
    with pytest.raises(ValueError, match="not \\['marauder'\\]"):
        wm.state_from_units([*MARINES, MARINES[0] | {"unit_type": "MARAUDER"}])
    with pytest.raises(ValueError, match="missing the columns \\['hp'\\]"):
        wm.state_from_units(pd.DataFrame(MARINES).drop(columns="hp"))
    with pytest.raises(ValueError, match="team must be"):
        wm.state_from_units([*MARINES, MARINES[0] | {"team": "green"}])
    with pytest.raises(ValueError, match="must be finite numbers"):
        wm.state_from_units([*MARINES, MARINES[0] | {"x": float("nan")}])
    with pytest.raises(ValueError, match="needs a horizon"):
        wm.predict(s, "hold/focus")
    with pytest.raises(ValueError, match="shorter than one model step"):
        wm.predict(s, "hold/focus", 0.01)
    with pytest.raises(ValueError, match="empty plan"):
        wm.predict(s, [])
    with pytest.raises(ValueError, match="the state has types"):
        WorldModel.load("mixed").predict(s, "hold", 1.0)
    assert wm.predict_many(s, []) == []
