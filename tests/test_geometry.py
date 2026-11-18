"""
Do the geometry functions give the hand-computed values, respect the symmetries of the
plane, and agree with each other (single moment vs. vectorised over an episode)?
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sindy_wm.states.geometry import (
    GEOMETRY,
    MARINE_HP,
    REACH,
    REF_UNITS,
    geometry,
    geometry_from_positions,
    state_from_game,
)


def _rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, -s], [s, c]]).T


def _snapshots(frames):
    """Snapshot table from a list of (blue, red) position arrays per step; NaN rows = dead unit."""
    rows = []
    for step, (blue, red) in enumerate(frames):
        for team, pos in (("blue", blue), ("red", red)):
            for uid, (x, y) in enumerate(pos):
                alive = not np.isnan(x)
                rows.append(
                    {
                        "step": step,
                        "team": team,
                        "unit_id": uid,
                        "x": x if alive else 0.0,
                        "y": y if alive else 0.0,
                        "alive": alive,
                    }
                )
    return pd.DataFrame(rows)


# --- geometry_from_positions: hand-computed cases ---


def test_one_vs_one_on_x_axis():
    g = geometry_from_positions(np.array([[0.0, 0.0]]), np.array([[4.0, 0.0]]))
    assert g["distance"] == pytest.approx(4.0)
    assert g["front_distance"] == pytest.approx(4.0)
    assert g["reach_blue"] == g["reach_red"] == 1.0  # 4 < REACH
    for k in ("depth_blue", "depth_red", "width_blue", "width_red"):
        assert g[k] == 0.0


def test_line_abreast_is_width_not_depth():
    # Blue in a line across the axis to Red: all spread is width
    blue = np.array([[0.0, -1.0], [0.0, 1.0]])
    red = np.array([[10.0, 0.0]])
    g = geometry_from_positions(blue, red)
    assert g["depth_blue"] == pytest.approx(0.0)
    assert g["width_blue"] == pytest.approx(1.0)  # population std of (-1, 1)


def test_column_is_depth_not_width():
    blue = np.array([[-1.0, 0.0], [1.0, 0.0]])
    red = np.array([[10.0, 0.0]])
    g = geometry_from_positions(blue, red)
    assert g["depth_blue"] == pytest.approx(1.0)
    assert g["width_blue"] == pytest.approx(0.0)
    assert g["front_distance"] == pytest.approx(9.0)  # closest pair, not centroid distance
    assert g["distance"] == pytest.approx(10.0)


def test_reach_counts_units_not_pairs():
    # Two blue units both near the same red unit; one far red unit
    blue = np.array([[0.0, 0.0], [0.0, 1.0]])
    red = np.array([[3.0, 0.0], [100.0, 0.0]])
    g = geometry_from_positions(blue, red)
    assert g["reach_blue"] == 2.0  # each blue unit has an enemy in reach
    assert g["reach_red"] == 1.0  # only the near red unit


def test_reach_boundary_is_inclusive():
    blue, red = np.array([[0.0, 0.0]]), np.array([[REACH, 0.0]])
    assert geometry_from_positions(blue, red)["reach_blue"] == 1.0
    red = np.array([[REACH + 1e-9, 0.0]])
    assert geometry_from_positions(blue, red)["reach_blue"] == 0.0


def test_wiped_out_side():
    g = geometry_from_positions(np.zeros((0, 2)), np.array([[1.0, 1.0]]))
    assert np.isnan(g["distance"]) and np.isnan(g["front_distance"])
    assert all(g[k] == 0.0 for k in GEOMETRY if k != "front_distance")


def test_coincident_centroids_do_not_crash():
    # Formations centred on the same point: the axis falls back to x
    blue = np.array([[-1.0, 0.0], [1.0, 0.0]])
    red = np.array([[0.0, -1.0], [0.0, 1.0]])
    g = geometry_from_positions(blue, red)
    assert g["distance"] == 0.0
    assert all(np.isfinite(g[k]) for k in g)


# --- geometry_from_positions: symmetries ---


@pytest.fixture
def random_battle():
    rng = np.random.default_rng(0)
    blue = rng.normal([0, 0], [1.5, 2.0], size=(7, 2))
    red = rng.normal([6, 1], [2.0, 1.0], size=(5, 2))
    return blue, red


@pytest.mark.parametrize("angle", [0.3, np.pi / 2, 2.0, np.pi])
def test_invariant_to_rotation_and_translation(random_battle, angle):
    blue, red = random_battle
    shift = np.array([17.0, -4.0])
    g0 = geometry_from_positions(blue, red)
    g1 = geometry_from_positions(_rotate(blue, angle) + shift, _rotate(red, angle) + shift)
    for k in g0:
        assert g1[k] == pytest.approx(g0[k], abs=1e-9), k


def test_swapping_sides_swaps_columns(random_battle):
    blue, red = random_battle
    g, s = geometry_from_positions(blue, red), geometry_from_positions(red, blue)
    assert s["distance"] == pytest.approx(g["distance"])
    assert s["front_distance"] == pytest.approx(g["front_distance"])
    for k in ("reach", "depth", "width"):
        assert s[f"{k}_blue"] == pytest.approx(g[f"{k}_red"])
        assert s[f"{k}_red"] == pytest.approx(g[f"{k}_blue"])


def test_unit_order_does_not_matter(random_battle):
    blue, red = random_battle
    g0 = geometry_from_positions(blue, red)
    g1 = geometry_from_positions(blue[::-1], np.roll(red, 2, axis=0))
    for k in g0:
        assert g1[k] == pytest.approx(g0[k]), k


def test_reach_bounded_by_unit_counts(random_battle):
    blue, red = random_battle
    for reach in (0.0, 2.0, REACH, 50.0):
        g = geometry_from_positions(blue, red, reach=reach)
        assert 0 <= g["reach_blue"] <= len(blue)
        assert 0 <= g["reach_red"] <= len(red)
    g = geometry_from_positions(blue, red, reach=1e6)
    assert (g["reach_blue"], g["reach_red"]) == (len(blue), len(red))


# --- geometry (vectorised) agrees with geometry_from_positions ---


def test_vectorised_matches_single_moment():
    rng = np.random.default_rng(1)
    frames = []
    for step in range(6):
        blue = rng.normal([step, 0], 1.5, size=(4, 2))
        red = rng.normal([8 - step, 0], 1.5, size=(3, 2))
        if step >= 3:
            blue[0] = np.nan  # a unit dies at step 3
        if step >= 4:
            red[[1, 2]] = np.nan
        frames.append((blue, red))
    out = geometry(_snapshots(frames))
    for step, (blue, red) in enumerate(frames):
        expected = geometry_from_positions(blue[~np.isnan(blue[:, 0])], red[~np.isnan(red[:, 0])])
        for k in GEOMETRY:
            assert out.loc[step, k] == pytest.approx(expected[k], abs=1e-9), (step, k)


def test_vectorised_carries_forward_after_wipe_out():
    frames = [
        (np.array([[0.0, 0.0], [0.0, 2.0]]), np.array([[4.0, 0.0]])),
        (np.array([[1.0, 0.0], [1.0, 2.0]]), np.array([[np.nan, np.nan]])),  # red wiped out
    ]
    out = geometry(_snapshots(frames))
    assert out.loc[1, "front_distance"] == out.loc[0, "front_distance"]
    assert out.loc[1, "width_blue"] == out.loc[0, "width_blue"]
    assert out.loc[1, "reach_blue"] == 0.0  # counts are not carried forward


def test_vectorised_matches_single_moment_with_coincident_centroids():
    blue = np.array([[-1.0, 0.0], [1.0, 0.0]])
    red = np.array([[0.0, -1.0], [0.0, 1.0]])
    out = geometry(_snapshots([(blue, red)]))
    expected = geometry_from_positions(blue, red)
    for k in GEOMETRY:
        assert out.loc[0, k] == pytest.approx(expected[k]), k


def test_dead_units_never_in_reach():
    # The dead red unit's recorded position (0, 0) sits on top of blue; it must not count
    frames = [(np.array([[0.0, 0.0]]), np.array([[np.nan, np.nan], [50.0, 0.0]]))]
    out = geometry(_snapshots(frames))
    assert out.loc[0, "reach_blue"] == 0.0
    assert out.loc[0, "front_distance"] == pytest.approx(50.0)


# --- state_from_game: the live state is on the common scale ---


def test_state_from_game_scaling():
    def unit(x, y, hp, shield=0.0):
        return SimpleNamespace(pos=np.array([x, y]), hp=hp, shield=shield)

    game = SimpleNamespace(
        agents={0: unit(0, 0, 45), 1: unit(0, 2, 20, 5)},
        enemies={0: unit(4, 1, 45)},
    )
    s = state_from_game(game)
    scale = REF_UNITS * MARINE_HP
    assert s["health_blue"] == pytest.approx(70 / scale)
    assert s["health_red"] == pytest.approx(45 / scale)
    assert s["alive_blue"] == pytest.approx(2 / REF_UNITS)
    assert s["reach_blue"] == pytest.approx(2 / REF_UNITS)
    assert s["distance"] == pytest.approx(4.0)
    assert set(GEOMETRY) <= set(s)


def test_state_from_game_with_no_enemies():
    game = SimpleNamespace(agents={0: SimpleNamespace(pos=np.zeros(2), hp=10.0, shield=0.0)}, enemies={})
    s = state_from_game(game)
    assert s["alive_red"] == 0.0 and s["health_red"] == 0.0
    assert np.isnan(s["distance"])
