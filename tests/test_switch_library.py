"""
Do the switch candidates behave as switches (0 far away, full force up close), and is
unit-type scaling consistent: scaling a model must not change its dimensionless form?
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sindy_wm.models.switch_library import (
    MOVES,
    UNITS,
    dimensionless,
    fire_switch,
    move_switches,
    scaled,
)


def _state(distance, depth=0.0, width=0.0, alive=1.0, **intents):
    n = len(np.atleast_1d(distance))
    d = {
        "distance": np.atleast_1d(distance).astype(float),
        "depth_blue": np.full(n, depth),
        "depth_red": np.full(n, depth),
        "width_blue": np.full(n, width),
        "width_red": np.full(n, width),
        "alive_blue": np.full(n, alive),
        "alive_red": np.full(n, alive),
    }
    for k in ("advance", "hold", "withdraw", "flee", "focus"):
        d[k] = np.full(n, intents.get(k, 0.0))
    return pd.DataFrame(d)


# --- fire_switch ---


def test_fire_switch_is_off_far_and_on_close():
    d = _state([0.5, 100.0], alive=0.7)
    s = fire_switch(d, "blue", c=6.0)[:, 0]
    assert s[0] == pytest.approx(0.7)  # all alive units in reach
    assert s[1] == pytest.approx(0.0, abs=1e-12)


def test_fire_switch_monotone_in_distance_and_c():
    d = _state(np.linspace(0, 20, 50), depth=1.0)
    s = fire_switch(d, "blue", c=6.0)[:, 0]
    assert np.all(np.diff(s) <= 1e-12)
    assert np.all(fire_switch(d, "blue", c=8.0)[:, 0] >= s - 1e-12)


def test_fire_switch_columns_are_products_with_intents():
    d = _state([3.0, 6.0, 9.0], depth=0.5, withdraw=1.0, hold=0.0, focus=0.5)
    F = fire_switch(d, "red", c=6.0)
    np.testing.assert_allclose(F[:, 1], F[:, 0])
    np.testing.assert_allclose(F[:, 2], 0.0)
    np.testing.assert_allclose(F[:, 3], 0.5 * F[:, 0])


def test_fire_switch_symmetric_sides():
    d = _state([4.0, 7.0], depth=0.8, width=1.0)
    np.testing.assert_allclose(fire_switch(d, "blue", 6.0), fire_switch(d, "red", 6.0))


# --- move_switches ---


def test_move_switch_closes_with_the_gap():
    cs, ws = dict.fromkeys(MOVES, 7.0), dict.fromkeys(MOVES, 0.5)
    d = _state([2.0, 20.0], advance=1.0)
    F = move_switches(d, cs, ws)
    assert F.shape == (2, 4)
    assert F[0, 0] == pytest.approx(0.0, abs=1e-6)  # fronts closer than c: advance has stopped
    assert F[1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(F[:, 1:], 0.0)  # other intents off


def test_flee_switch_on_when_close():
    cs, ws = dict.fromkeys(MOVES, 7.0), dict.fromkeys(MOVES, 0.5)
    F = move_switches(_state([2.0, 20.0], withdraw=1.0), cs, ws)
    assert F[0, 3] == pytest.approx(1.0)
    assert F[1, 3] == pytest.approx(0.0, abs=1e-6)


# --- scaling between unit types ---


@pytest.fixture
def model():
    names = ["1", "alive_blue", "fire"]
    return SimpleNamespace(
        names=names,
        coef={
            "health_blue": np.array([0.0, -0.2, -1.3]),
            "health_red": np.array([0.01, 0.0, -0.9]),
            "distance": np.array([-2.5, 0.4, 0.0]),
        },
        P={
            "share_blue": np.array([6.4, 0, 1, 0.05, 1, 1.0]),
            "share_red": np.array([5.8, 0, 1, 0.05, 1, 1.0]),
            "gates": {
                "advance": np.array([0, 7.2, 1, 1, 0, 0, 0.5, 0]),
                "hold": np.array([0, 6.0, 1, 1, 0, 0, 1.0, 0]),
                "withdraw": np.array([0, 8.0, 1, 1, 0, 0, 0.5, 0, 0, 3.4, 1.5, 1]),
            },
        },
        p={"blue": {"health": 0.5, "alive": 0.0}, "red": {"health": 0.7, "alive": 0.3}},
    )


def test_scaling_to_same_unit_is_identity(model):
    m = scaled(model, "marine", "marine")
    for eq in model.coef:
        np.testing.assert_allclose(m.coef[eq], model.coef[eq])
    for g in model.P["gates"]:
        np.testing.assert_allclose(m.P["gates"][g], model.P["gates"][g])
    np.testing.assert_allclose(m.P["share_blue"], model.P["share_blue"])


def test_scaling_does_not_mutate_original(model):
    before = model.P["gates"]["advance"].copy(), model.coef["distance"].copy()
    scaled(model, "marine", "stalker")
    np.testing.assert_array_equal(model.P["gates"]["advance"], before[0])
    np.testing.assert_array_equal(model.coef["distance"], before[1])


def test_round_trip(model):
    back = scaled(scaled(model, "marine", "stalker"), "stalker", "marine")
    for eq in model.coef:
        np.testing.assert_allclose(back.coef[eq], model.coef[eq])
    for g in model.P["gates"]:
        np.testing.assert_allclose(back.P["gates"][g], model.P["gates"][g])
    for side in ("share_blue", "share_red"):
        np.testing.assert_allclose(back.P[side], model.P[side])


def test_dimensionless_form_unchanged_by_scaling(model):
    """The point of the dimensionless form: a scaled model is the same model in other units."""
    a = dimensionless(model, "marine")
    b = dimensionless(scaled(model, "marine", "stalker"), "stalker")
    pd.testing.assert_series_equal(a, b)


def test_switch_distances_scale_with_reach(model):
    m = scaled(model, "marine", "stalker")
    s = UNITS["stalker"]["reach"] / UNITS["marine"]["reach"]
    assert m.P["gates"]["advance"][1] == pytest.approx(7.2 * s)
    assert m.P["gates"]["withdraw"][9] == pytest.approx(3.4 * s)  # flee distance
    assert m.P["share_blue"][0] == pytest.approx(6.4 * s)
    assert m.P["share_blue"][2] == 1  # unit-free entries untouched


def test_units_tau():
    for u in UNITS.values():
        assert u["tau"] == pytest.approx(u["health"] / u["dps"])
        assert u["tau"] > 0
