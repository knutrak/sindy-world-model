"""Per-type states (states.typed): consistency with the single-type pipeline, and the fire allocation."""

import numpy as np
import pytest

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.scenarios import make_env
from sindy_wm.states.aggregate import level_a
from sindy_wm.states.geometry import MARINE_HP, REACH, REF_UNITS, geometry
from sindy_wm.states.typed import pair_reach, typed_states


@pytest.fixture(scope="module")
def marine_snapshots():
    env = make_env("10m_vs_10m", step_mul=2)
    return play_episode(env, IntentSchedule(), seed=3, max_steps=160).snapshots


@pytest.fixture(scope="module")
def mixed_snapshots():
    types = (("MARINE", "MARAUDER") * 3, ("MARAUDER", "MARINE", "MARINE") * 2)
    env = make_env("mixed_m_mrd", step_mul=2, unit_types=types)
    return play_episode(env, IntentSchedule(withdraw_radius=8.0), seed=1, max_steps=200).snapshots


def test_pair_reach_matches_the_game_rule():
    assert pair_reach("MARINE", "MARINE") == pytest.approx(REACH)
    assert pair_reach("MARINE", "MARAUDER") == pytest.approx(5 + 0.375 + 0.5625)
    assert pair_reach("MARAUDER", "MARAUDER") == pytest.approx(7.125)


def test_one_type_reduces_to_the_team_pipeline(marine_snapshots):
    typed = typed_states(marine_snapshots, ("MARINE",))
    team = level_a(marine_snapshots).join(geometry(marine_snapshots))
    for side in ("blue", "red"):
        np.testing.assert_allclose(
            typed[f"health_{side}_marine"], team[f"health_{side}"] / (REF_UNITS * MARINE_HP)
        )
        np.testing.assert_allclose(typed[f"alive_{side}_marine"], team[f"alive_{side}"] / REF_UNITS)
        np.testing.assert_allclose(typed[f"reach_{side}_marine"], team[f"reach_{side}"] / REF_UNITS)
        np.testing.assert_allclose(typed[f"depth_{side}"], team[f"depth_{side}"], atol=1e-9)
    np.testing.assert_allclose(typed["distance"], team["distance"], atol=1e-9)
    np.testing.assert_allclose(typed["front_distance"], team["front_distance"], atol=1e-9)


def test_fire_allocation_counts_units_firing_and_never_exceeds_reach(mixed_snapshots):
    typed = typed_states(mixed_snapshots, ("MARINE", "MARAUDER"))
    for side in ("blue", "red"):
        for a in ("marine", "marauder"):
            fire = typed[f"fire_{side}_{a}_marine"] + typed[f"fire_{side}_{a}_marauder"]
            assert (fire <= typed[f"alive_{side}_{a}"] + 1e-12).all()
        total = sum(
            typed[f"fire_{side}_{a}_{t}"] for a in ("marine", "marauder") for t in ("marine", "marauder")
        )
        assert total.max() > 0  # the battle reaches contact
    # step 0: full health of the given composition
    first = typed.iloc[0]
    assert first["alive_blue_marauder"] == pytest.approx(0.3) and first["alive_red_marine"] == pytest.approx(
        0.4
    )
    assert first["health_blue_marauder"] == pytest.approx(0.3)


def test_team_view_sums_types_in_marine_hp():
    from sindy_wm.models.typed_model import idx, team_view

    X = np.zeros(9)
    X[idx("health", "blue", "marine")], X[idx("health", "blue", "marauder")] = 0.5, 0.2
    X[idx("alive", "red", "marine")], X[idx("alive", "red", "marauder")] = 0.3, 0.1
    X[8] = 6.0
    T = team_view(X)
    assert T[0] == pytest.approx(0.5 + 0.2 * 125 / 45)  # ten marines' hp
    assert T[3] == pytest.approx(0.4) and T[4] == 6.0
