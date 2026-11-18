"""
Does level_a sum health over the roster, place centroids on alive units only, and
carry distance forward after a wipe-out? Does engagement count what it says?
"""

import pandas as pd
import pytest

from sindy_wm.states.aggregate import engagement, level_a


def _row(step, team, uid, x, y, hp, shield=0.0, alive=None, **extra):
    return {
        "step": step,
        "team": team,
        "unit_id": uid,
        "x": float(x),
        "y": float(y),
        "hp": float(hp),
        "shield": float(shield),
        "alive": hp > 0 if alive is None else alive,
        **extra,
    }


@pytest.fixture
def battle():
    """Two steps; Blue loses unit 1 at step 1, Red is wiped out at step 2."""
    return pd.DataFrame(
        [
            _row(0, "blue", 0, 0, 0, 40, 5),
            _row(0, "blue", 1, 0, 2, 30),
            _row(0, "red", 0, 6, 1, 45),
            _row(1, "blue", 0, 1, 0, 35, 5),
            _row(1, "blue", 1, 50, 50, 0),  # dead, at a far-away position
            _row(1, "red", 0, 5, 0, 10),
            _row(2, "blue", 0, 2, 0, 35, 5),
            _row(2, "blue", 1, 50, 50, 0),
            _row(2, "red", 0, 5, 0, 0),
        ]
    )


def test_health_and_alive(battle):
    s = level_a(battle)
    assert s["health_blue"].tolist() == [75.0, 40.0, 40.0]
    assert s["health_red"].tolist() == [45.0, 10.0, 0.0]
    assert s["alive_blue"].tolist() == [2.0, 1.0, 1.0]
    assert s["alive_red"].tolist() == [1.0, 1.0, 0.0]


def test_shields_optional(battle):
    s = level_a(battle, include_shields=False)
    assert s["health_blue"].tolist() == [70.0, 35.0, 35.0]


def test_centroid_ignores_dead_units(battle):
    s = level_a(battle)
    assert s.loc[0, "distance"] == pytest.approx(6.0)  # blue centroid (0, 1), red (6, 1)
    assert s.loc[1, "distance"] == pytest.approx(4.0)  # dead unit at (50, 50) left out


def test_distance_carried_forward_after_wipe_out(battle):
    s = level_a(battle)
    assert s.loc[2, "distance"] == s.loc[1, "distance"]


def test_normalize(battle):
    s = level_a(battle, normalize=True)
    for c in ("health_blue", "health_red", "alive_blue", "alive_red"):
        assert s.loc[0, c] == 1.0
    assert s.loc[1, "health_blue"] == pytest.approx(40 / 75)
    assert s.loc[0, "distance"] == pytest.approx(6.0)  # distance is not normalised


def test_health_never_increases_without_healing(battle):
    s = level_a(battle)
    assert (s[["health_blue", "health_red", "alive_blue", "alive_red"]].diff().dropna() <= 0).all().all()


def test_rejects_several_episodes(battle):
    with pytest.raises(ValueError, match="single episode"):
        level_a(pd.concat([battle, battle]))


# --- engagement ---


def test_engagement_counts():
    extra = dict(target_team="red", vx=3.0, vy=4.0)
    red = dict(target_team="blue", vx=0.0, vy=0.0)
    snaps = pd.DataFrame(
        [
            _row(0, "blue", 0, 0, 0, 45, target_id=0, target_in_range=True, **extra),
            _row(0, "blue", 1, 0, 1, 45, target_id=0, target_in_range=True, **extra),
            _row(0, "blue", 2, 0, 2, 45, target_id=1, target_in_range=False, **extra),
            _row(0, "blue", 3, 0, 3, 0, target_id=1, target_in_range=True, **extra),  # dead
            _row(0, "red", 0, 5, 0, 45, target_id=0, target_in_range=True, **red),
            _row(0, "red", 1, 5, 1, 45, target_id=2, target_in_range=True, **red),
        ]
    )
    e = engagement(snaps).loc[0]
    assert e["engaged_blue"] == 2  # out of range and dead units excluded
    assert e["targets_blue"] == 1  # focused fire
    assert e["engaged_red"] == 2
    assert e["targets_red"] == 2  # spread fire
    assert e["speed_blue"] == pytest.approx(5.0)
    assert e["speed_red"] == 0.0


def test_engagement_zero_when_nobody_in_range():
    extra = dict(target_team="red", target_id=0, target_in_range=False, vx=0.0, vy=0.0)
    snaps = pd.DataFrame([_row(0, "blue", 0, 0, 0, 45, **extra), _row(0, "red", 0, 9, 0, 45, **extra)])
    e = engagement(snaps).loc[0]
    assert e["engaged_blue"] == e["engaged_red"] == 0
    assert e["targets_blue"] == e["targets_red"] == 0


def test_engagement_needs_target_columns(battle):
    with pytest.raises(ValueError, match="lack"):
        engagement(battle)
