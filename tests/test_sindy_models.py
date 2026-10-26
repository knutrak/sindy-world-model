"""
Does to_trajectory trim/downsample correctly, and does the fit/eval/save pipeline round-trip?
"""

import json

import numpy as np
import pandas as pd
import pytest

from sindy_wm.models.gated import GatedModel, fit_gated
from sindy_wm.models.lanchester import LanchesterModel, fit_lanchester
from sindy_wm.models.metrics import evaluate, evaluate_horizons, healing_share, summarize_eval
from sindy_wm.models.runs import save_run
from sindy_wm.models.simulation import simulate, simulate_to_elimination
from sindy_wm.models.sindy import fit_sindy
from sindy_wm.models.trajectories import derivative_data, to_trajectory

STATE_COLUMNS = ["health_blue", "health_red"]


def test_to_trajectory_trims_to_one_step_before_first_damage():
    ep = pd.DataFrame(
        {
            "t": [0.0, 0.25, 0.5, 0.75, 1.0],
            "health_blue": [100.0, 100.0, 100.0, 90.0, 80.0],
            "health_red": [100.0, 100.0, 100.0, 100.0, 95.0],
        }
    )
    trimmed = to_trajectory(ep, STATE_COLUMNS, trim_to_engagement=True, downsample=1)
    # First damage is at index 3 (health_blue drops); trim starts one step before, at index 2
    assert len(trimmed) == 3
    assert trimmed["health_blue"].tolist() == [100.0, 90.0, 80.0]
    assert trimmed["t"].iloc[0] == 0.0  # time reset to start at 0


def test_to_trajectory_downsamples():
    ep = pd.DataFrame({"t": np.arange(6) * 0.1, "health_blue": range(6), "health_red": range(6)})
    trimmed = to_trajectory(ep, STATE_COLUMNS, trim_to_engagement=False, downsample=2)
    assert trimmed["health_blue"].tolist() == [0, 2, 4]


def _synthetic_episode(seed, n=40, dt=0.25):
    # A simple linear decay, like a toy Lanchester law, with a short no-damage lead-in
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    blue = np.clip(100 - 2 * t + rng.normal(0, 0.5, n), 0, None)
    red = np.clip(100 - 3 * t + rng.normal(0, 0.5, n), 0, None)
    blue[:3], red[:3] = 100.0, 100.0
    return pd.DataFrame({"t": t, "health_blue": blue, "health_red": red})


@pytest.fixture
def fitted():
    states = {i: _synthetic_episode(i) for i in range(10)}
    battle_won = pd.Series(
        {i: states[i]["health_blue"].iloc[-1] > states[i]["health_red"].iloc[-1] for i in range(10)}
    )
    trajectories = {
        i: to_trajectory(states[i], STATE_COLUMNS, trim_to_engagement=True, downsample=1) for i in range(10)
    }
    train_ids, test_ids = list(range(8)), list(range(8, 10))
    model = fit_sindy(
        trajectories, train_ids, STATE_COLUMNS, dt=0.25, smooth_window=5, degree=1, threshold=0.01
    )
    return model, trajectories, test_ids, battle_won


def test_simulate_matches_trajectory_length(fitted):
    model, trajectories, test_ids, _ = fitted
    traj = trajectories[test_ids[0]]
    sim = simulate(model, traj, STATE_COLUMNS)
    assert sim is not None
    assert sim.shape == (len(traj), len(STATE_COLUMNS))
    assert np.all(np.isfinite(sim))


def test_evaluate_and_summarize_eval(fitted):
    model, trajectories, test_ids, battle_won = fitted
    per_episode = evaluate(model, test_ids, trajectories, STATE_COLUMNS, battle_won)
    assert list(per_episode.index) == test_ids
    assert {"rmse_start", "rmse_midway", "winner_correct_start", "winner_correct_midway"} <= set(
        per_episode.columns
    )

    metrics = summarize_eval(model, test_ids, trajectories, STATE_COLUMNS, battle_won)
    assert 0.0 <= metrics["majority_baseline"] <= 1.0
    assert metrics["n_terms"] > 0
    assert metrics["failed_simulations"] == 0


def test_derivative_data_is_exactly_what_fit_sindy_regresses_on(fitted):
    import pysindy as ps

    model, trajectories, _, _ = fitted
    x, dx = derivative_data(trajectories, list(range(8)), STATE_COLUMNS, smooth_window=5)
    theta = np.column_stack([np.ones(len(x)), x])  # the degree-1 library: 1, health_blue, health_red
    coef = ps.STLSQ(threshold=0.01).fit(theta, dx).coef_
    np.testing.assert_allclose(coef, model.coefficients(), atol=1e-10)


def test_save_run_writes_expected_files_and_never_overwrites(tmp_path, fitted):
    model, trajectories, test_ids, battle_won = fitted
    metrics = summarize_eval(model, test_ids, trajectories, STATE_COLUMNS, battle_won)
    per_episode = evaluate(model, test_ids, trajectories, STATE_COLUMNS, battle_won)

    run_dir = save_run(
        tmp_path, "unit_test", {"description": "test"}, model, STATE_COLUMNS, metrics, per_episode, {}
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "model.json").exists()
    assert (run_dir / "equations.txt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "per_episode.csv").exists()
    assert (run_dir / "figures").is_dir()

    model_json = json.loads((run_dir / "model.json").read_text())
    assert model_json["state_names"] == STATE_COLUMNS
    assert len(model_json["coefficients"]) == len(STATE_COLUMNS)

    config = json.loads((run_dir / "config.json").read_text())
    assert "code" in config  # code_info() was merged in

    with pytest.raises(FileExistsError):
        save_run(tmp_path, "unit_test", {}, model, STATE_COLUMNS, metrics, per_episode, {})


def test_fit_lanchester_recovers_known_coefficients():
    from scipy.integrate import solve_ivp

    # A trajectory actually generated by the square law dX1/dt = -0.5*X2, dX2/dt = -0.3*X1
    # (two independent exponentials would NOT satisfy this -- the coupling matters).
    t = np.arange(60) * 0.1
    sol = solve_ivp(lambda _t, x: [-0.5 * x[1], -0.3 * x[0]], (t[0], t[-1]), [100.0, 80.0], t_eval=t)
    traj = pd.DataFrame({"t": t, "x1": sol.y[0], "x2": sol.y[1]})
    m = fit_lanchester({0: traj}, [0], ["x1", "x2"], dt=0.1, smooth_window=5)
    assert m.a == pytest.approx(0.5, abs=0.02)
    assert m.b == pytest.approx(0.3, abs=0.02)


def test_lanchester_model_interface_matches_sindy_shape():
    t = np.arange(10) * 0.1
    traj = pd.DataFrame({"t": t, "x1": 1.0 - 0.05 * t, "x2": 1.0 - 0.03 * t})
    m = fit_lanchester({0: traj}, [0], ["x1", "x2"], dt=0.1, smooth_window=5)

    assert m.get_feature_names() == ["x1", "x2"]
    assert m.coefficients().shape == (2, 2)
    assert np.count_nonzero(m.coefficients()) == 2  # pure cross-coupling: no self-term, no constant
    eqns = m.equations(precision=3)
    assert len(eqns) == 2 and "x2" in eqns[0] and "x1" in eqns[1]
    sim = m.simulate(np.array([1.0, 1.0]), np.array([0.0, 0.1, 0.2]))
    assert sim.shape == (3, 2)


def test_simulate_to_elimination_matches_square_law_solution():
    from scipy.optimize import brentq

    # With a = b = k, the square law has Red(t) = R0*cosh(kt) - B0*sinh(kt) in closed form
    k, b0, r0, threshold = 0.5, 1.0, 0.6, 0.02
    model = LanchesterModel(["x1", "x2"], a=k, b=k)
    expected_t = brentq(lambda t: r0 * np.cosh(k * t) - b0 * np.sinh(k * t) - threshold, 0, 20)

    blue_won, t_elim = simulate_to_elimination(model, np.array([b0, r0]), t_max=60, threshold=threshold)
    assert blue_won is True
    assert t_elim == pytest.approx(expected_t, rel=1e-4)

    red_won, _ = simulate_to_elimination(model, np.array([r0, b0]), t_max=60, threshold=threshold)
    assert red_won is False


def test_simulate_to_elimination_gives_up_without_elimination_or_on_blow_up():
    still = LanchesterModel(["x1", "x2"], a=0.0, b=0.0)
    assert simulate_to_elimination(still, np.array([1.0, 1.0]), t_max=10) == (None, None)

    growing = LanchesterModel(["x1", "x2"], a=-1.0, b=-1.0)  # both sides grow without bound
    assert simulate_to_elimination(growing, np.array([1.0, 1.0]), t_max=1e6) == (None, None)


def _gated_trajectories(true_model, starts, t):
    """Trajectories of a known model from several starting states (off the diagonal, so it's identifiable)."""
    return {
        k: pd.DataFrame({"t": t, "b": sim[:, 0], "r": sim[:, 1]})
        for k, x0 in enumerate(starts)
        for sim in [true_model.simulate(np.array(x0), t, integrator_kws={"method": "LSODA", "rtol": 1e-9})]
    }


def test_fit_gated_recovers_known_coefficients():
    # dB/dt = -R*(0.4 - 0.2*B), dR/dt = -B*0.3; degree-1 terms in order: 1, own, opp
    true = GatedModel(["b", "r"], degree=1, coef=[[0.4, -0.2, 0.0], [0.3, 0.0, 0.0]])
    t = np.arange(40) * 0.1
    trajectories = _gated_trajectories(true, [(1.0, 1.0), (1.0, 0.6), (0.6, 1.0), (0.8, 0.9)], t)
    m = fit_gated(trajectories, list(trajectories), ["b", "r"], smooth_window=5, degree=1, threshold=0.05)
    np.testing.assert_allclose(m.coefficients(), true.coefficients(), atol=0.03)


def test_gated_model_never_changes_a_side_once_the_opponent_is_gone():
    m = GatedModel(["b", "r"], degree=2, coef=np.full((2, 6), 0.3))
    assert m.get_feature_names() == ["-opp", "-opp own", "-opp^2", "-opp own^2", "-opp^2 own", "-opp^3"]
    assert m.predict(np.array([0.7, 0.0]))[0, 0] == 0.0  # Red gone: Blue stays at 0.7
    assert m.predict(np.array([0.0, 0.4]))[0, 1] == 0.0  # Blue gone: Red stays at 0.4
    eqs = m.equations(precision=1)
    assert eqs[0].startswith("-0.3 r + -0.3 r b")


def test_healing_share():
    assert healing_share(LanchesterModel(["b", "r"], a=0.2, b=0.3)) == 0.0
    # A positive own term makes a side gain health whenever it outweighs the opponent's damage
    healer = GatedModel(["b", "r"], degree=1, coef=[[0.2, -1.0, 0.0], [0.3, 0.0, 0.0]])
    assert 0.0 < healing_share(healer) < 1.0


def test_evaluate_horizons_exact_model_beats_baselines():
    true = GatedModel(["b", "r"], degree=0, coef=[[0.1], [0.15]])  # square law; both sides stay > 0
    t = np.arange(40) * 0.125
    trajectories = _gated_trajectories(true, [(1.0, 1.0), (1.0, 0.8)], t)
    table = evaluate_horizons(true, list(trajectories), trajectories, ["b", "r"], horizons=(0.5, 2.0))
    assert list(table.index) == [0.5, 2.0]
    assert (table["n"] > 0).all() and (table["failed"] == 0).all()
    assert (table["model"] < 1e-4).all()
    assert (table["model"] < table["last_slope"]).all() and (table["last_slope"] < table["persistence"]).all()
