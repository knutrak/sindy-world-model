"""Tests for the final-exploration harness: splits, end-of-battle logic, metrics, the test-set guard."""

import numpy as np
import pandas as pd
import pytest

from sindy_wm.evaluation import harness as H


def _traj(hb, hr, ab=None, ar=None, d=None):
    """One rollout (1, T, 5) from health sequences; alive follows health unless given."""
    hb, hr = np.asarray(hb, float), np.asarray(hr, float)
    ab = hb if ab is None else np.asarray(ab, float)
    ar = hr if ar is None else np.asarray(ar, float)
    d = np.zeros_like(hb) if d is None else np.asarray(d, float)
    return np.stack([hb, hr, ab, ar, d], -1)[None]


def test_folds_are_disjoint_and_cover_everything():
    eps = np.arange(103)
    f = H.folds(eps, k=5, seed=1)
    allf = np.concatenate(f)
    assert len(f) == 5
    assert sorted(allf) == list(eps)
    assert len(set(allf)) == len(allf)
    assert all(len(x) in (20, 21) for x in f)
    # fixed seed: same folds again
    assert all((a == b).all() for a, b in zip(f, H.folds(eps, k=5, seed=1), strict=True))


def test_rollout_end_red_eliminated_first():
    Y = _traj([1, 0.8, 0.6, 0.5], [1, 0.5, 0.0, 0.0])
    t, won = H.rollout_end(Y)
    assert t[0] == 2 and won[0]


def test_rollout_end_blue_eliminated_first():
    Y = _traj([1, 0.3, 0.0, 0.0], [1, 0.9, 0.8, 0.8])
    t, won = H.rollout_end(Y)
    assert t[0] == 2 and not won[0]


def test_rollout_end_without_elimination_uses_last_margin():
    Y = _traj([1, 0.9, 0.8], [1, 0.95, 0.9])
    t, won = H.rollout_end(Y)
    assert t[0] == -1 and not won[0]


def test_rollout_end_by_alive_below_half_a_unit():
    Y = _traj([1, 0.5, 0.2], [1, 0.9, 0.8], ar=[1, 0.5, 0.04])
    t, won = H.rollout_end(Y)
    assert t[0] == 2 and won[0]


def test_battle_metrics_perfect_prediction():
    Yt = _traj([1, 0.5, 0.2, np.nan], [1, 0.4, 0.0, np.nan])
    B = {"Y": Yt, "end": np.array([2]), "won": np.array([True]), "X0": Yt[:, 0]}
    Y = np.nan_to_num(Yt.copy())
    Y[:, 3] = Y[:, 2]
    m = H.battle_metrics(Y, B)
    assert m["battle_health_nrmse"] == pytest.approx(0)
    assert m["winner_acc"] == 1 and m["brier"] == 0
    assert m["final_health_err"] == pytest.approx(0)
    assert m["duration_err_s"] == pytest.approx(0)
    assert m["no_end_rate"] == 0 and m["blowup_rate"] == 0


def test_battle_metrics_flags_blowup_and_healing():
    Yt = _traj([1, 0.5, 0.0], [1, 0.8, 0.6])
    B = {"Y": Yt, "end": np.array([2]), "won": np.array([False]), "X0": Yt[:, 0]}
    healing = _traj([1, 1.2, 0.0], [1, 0.8, 0.6])
    assert H.battle_metrics(healing, B)["blowup_rate"] == 1
    nan = _traj([1, np.nan, 0.0], [1, 0.8, 0.6])
    assert H.battle_metrics(nan, B)["blowup_rate"] == 1


def test_horizon_metrics_perfect_prediction():
    rng = np.random.default_rng(0)
    T = 8 * 8 + 1
    Y = np.cumsum(rng.normal(size=(50, T, 5)), axis=1)
    m = H.horizon_metrics(Y, {"Y": Y})
    assert m["health_4s"] == 0 and m["margin_r2_4s"] == pytest.approx(1)
    assert m["step_r2_health_blue"] == pytest.approx(1)


def test_test2_sets_are_guarded(monkeypatch):
    monkeypatch.delenv(H.PHASE3_FLAG, raising=False)
    with pytest.raises(PermissionError):
        H.load_states("test2")
    with pytest.raises(PermissionError):
        H.rows_with_rates("test2_cf")


def test_append_adds_new_columns_without_touching_old_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "OUT", tmp_path)
    path = tmp_path / "r.csv"
    H._append(path, pd.DataFrame([{"a": 1, "b": 2}]))
    H._append(path, pd.DataFrame([{"a": 3, "b": 4}]))
    H._append(path, pd.DataFrame([{"a": 5, "c": 6}]))
    r = pd.read_csv(path)
    assert r["a"].tolist() == [1, 3, 5]
    assert r["b"].tolist()[:2] == [2, 4] and np.isnan(r["b"].iloc[2])
    assert np.isnan(r["c"].iloc[0]) and r["c"].iloc[2] == 6
