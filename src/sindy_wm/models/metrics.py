"""Scoring fitted models on test episodes.

summarize_eval(...)      winner and battle-length predictions, from the start and from midway
evaluate_horizons(...)   error a fixed time ahead, as a world model is used
healing_share(model)     does the model ever make a side's quantity increase?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pysindy as ps

from sindy_wm.models.simulation import simulate, simulate_to_elimination


def evaluate(
    model: ps.SINDy,
    ids,
    trajectories: dict[int, pd.DataFrame],
    state_columns: list[str],
    battle_won: pd.Series,
    threshold: float = 0.02,
    horizon_factor: float = 3.0,
) -> pd.DataFrame:
    """Metrics per episode, simulating from the start and from midway.

    Two kinds of winner prediction are computed:

    - `winner_correct_*`: simulate over the real trajectory's time points and
      call the side with more health at the real end time the winner (uses the
      real battle length).
    - `winner_correct_*_elim`: simulate until one side drops to `threshold`
      (see `simulate_to_elimination`), so the model must also predict when the
      battle ends. `t_elim_*` is that predicted remaining time and
      `duration_actual_*` the real one; `no_elim_*` is True if neither side was
      eliminated within `horizon_factor` times the longest trajectory in `ids`
      (counted as a wrong prediction).

    Args:
        model: A fitted SINDy model.
        ids: Episode indices to evaluate on.
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        state_columns: The two SINDy state columns, Blue's quantity first and
            Red's second (e.g. `["health_blue", "health_red"]`, or
            `["alive_blue", "alive_red"]`) -- order matters, name doesn't.
        battle_won: Whether Blue won, indexed by episode_idx (e.g. `index["battle_won"]`).
    """
    blue, red = 0, 1  # by convention: state_columns is always [blue_quantity, red_quantity]
    t_max = horizon_factor * max(float(trajectories[i]["t"].iloc[-1]) for i in ids)
    rows = []
    for i in ids:
        traj = trajectories[i]
        t_all = traj["t"].to_numpy()
        actual = traj[state_columns].to_numpy()
        blue_won = bool(battle_won.loc[i])
        mid = len(traj) // 2
        row = {
            "episode_idx": i,
            "blue_won": blue_won,
            # Naive baseline: whoever has more health midway wins
            "leader_correct_midway": bool((actual[mid, blue] > actual[mid, red]) == blue_won),
        }
        for label, start in [("start", 0), ("midway", mid)]:
            won_elim, t_elim = simulate_to_elimination(model, actual[start], t_max, threshold)
            row |= {
                f"winner_correct_{label}_elim": won_elim is not None and won_elim == blue_won,
                f"no_elim_{label}": won_elim is None,
                f"t_elim_{label}": np.nan if t_elim is None else t_elim,
                f"duration_actual_{label}": float(t_all[-1] - t_all[start]),
            }
            sim = simulate(model, traj, state_columns, start)
            if sim is None:
                row |= {f"rmse_{label}": np.nan, f"winner_correct_{label}": False, f"failed_{label}": True}
                continue
            row |= {
                f"rmse_{label}": float(np.sqrt(np.mean((sim - actual[start:]) ** 2))),
                f"winner_correct_{label}": bool((sim[-1, blue] > sim[-1, red]) == blue_won),
                f"failed_{label}": False,
            }
        rows.append(row)
    return pd.DataFrame(rows).set_index("episode_idx")


def summarize_eval(
    model: ps.SINDy,
    ids,
    trajectories: dict[int, pd.DataFrame],
    state_columns: list[str],
    battle_won: pd.Series,
    threshold: float = 0.02,
    horizon_factor: float = 3.0,
) -> dict:
    """RMSE and winner accuracy (vs. their naive baselines), averaged over `ids`.

    `*_elim` metrics simulate until a side is eliminated (see `evaluate`).
    `duration_mae_*` is the mean absolute error of the predicted remaining
    battle time in seconds, over episodes where the model eliminated a side.
    """
    ev = evaluate(model, ids, trajectories, state_columns, battle_won, threshold, horizon_factor)
    blue_share = ev["blue_won"].mean()
    return {
        "rmse_start": float(ev["rmse_start"].mean()),
        "rmse_midway": float(ev["rmse_midway"].mean()),
        "winner_acc_start": float(ev["winner_correct_start"].mean()),
        "winner_acc_midway": float(ev["winner_correct_midway"].mean()),
        "majority_baseline": float(max(blue_share, 1 - blue_share)),
        "leader_baseline_midway": float(ev["leader_correct_midway"].mean()),
        "winner_acc_start_elim": float(ev["winner_correct_start_elim"].mean()),
        "winner_acc_midway_elim": float(ev["winner_correct_midway_elim"].mean()),
        "no_elimination": int(ev["no_elim_start"].sum() + ev["no_elim_midway"].sum()),
        "duration_mae_start": float((ev["t_elim_start"] - ev["duration_actual_start"]).abs().mean()),
        "duration_mae_midway": float((ev["t_elim_midway"] - ev["duration_actual_midway"]).abs().mean()),
        "failed_simulations": int(ev["failed_start"].sum() + ev["failed_midway"].sum()),
        "n_terms": int(np.count_nonzero(model.coefficients())),
    }


def evaluate_horizons(
    model: ps.SINDy,
    ids,
    trajectories: dict[int, pd.DataFrame],
    state_columns: list[str],
    horizons=(0.5, 1.0, 2.0, 4.0),
    every: float = 0.5,
    history: float = 1.0,
) -> pd.DataFrame:
    """Error of predictions a fixed time ahead, the way a world model is used.

    From many start points in every episode (every `every` seconds, once
    `history` seconds have passed), the model is integrated from the real state
    to each horizon and compared with the real state then. Predictions are
    clipped at 0, since a quantity like health cannot go below it (a model can).

    Two baselines, scored on the same start points:

    - `persistence`: nothing changes.
    - `last_slope`: the average rate of change over the last `history` seconds
      continues (also clipped at 0).

    A start point is skipped for all three if the model's simulation fails;
    `failed` counts those.

    Args:
        model: A fitted model (`ps.SINDy`, `LanchesterModel` or `GatedModel`).
        ids: Episode indices to evaluate on.
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        state_columns: The state columns the model was fitted on.
        horizons: Prediction horizons in seconds.
        every: Seconds between start points.
        history: Seconds of history before the first start point; also the
            window of the `last_slope` baseline.

    Returns:
        One row per horizon: mean absolute error (averaged over the state
        columns) of `model`, `persistence` and `last_slope`, the number of
        start points `n`, and `failed`.
    """
    horizons = sorted(horizons)
    integrator_kws = {"method": "LSODA", "rtol": 1e-6, "atol": 1e-8}
    errors = {h: {"model": [], "persistence": [], "last_slope": []} for h in horizons}
    failed = dict.fromkeys(horizons, 0)
    for i in ids:
        traj = trajectories[i]
        t = traj["t"].to_numpy()
        x = traj[state_columns].to_numpy()
        dt = float(np.median(np.diff(t)))
        k_history = max(round(history / dt), 1)
        steps = {h: max(round(h / dt), 1) for h in horizons}
        for k0 in range(k_history, len(x) - steps[horizons[0]], max(round(every / dt), 1)):
            ahead = [h for h in horizons if k0 + steps[h] < len(x)]
            t_eval = np.array([0.0] + [steps[h] * dt for h in ahead])
            try:
                sim = np.asarray(model.simulate(x[k0], t_eval, integrator_kws=integrator_kws))
            except Exception:  # integration can fail if the model blows up
                sim = None
            if sim is None or len(sim) != len(t_eval) or not np.all(np.isfinite(sim)):
                for h in ahead:
                    failed[h] += 1
                continue
            slope = (x[k0] - x[k0 - k_history]) / (k_history * dt)
            for n, h in enumerate(ahead, start=1):
                actual = x[k0 + steps[h]]
                predictions = {
                    "model": sim[n],
                    "persistence": x[k0],
                    "last_slope": x[k0] + slope * t_eval[n],
                }
                for name, pred in predictions.items():
                    errors[h][name].append(float(np.mean(np.abs(np.clip(pred, 0, None) - actual))))
    rows = [
        {
            "horizon_s": h,
            **{name: float(np.mean(e)) if e else np.nan for name, e in errors[h].items()},
            "n": len(errors[h]["model"]),
            "failed": failed[h],
        }
        for h in horizons
    ]
    return pd.DataFrame(rows).set_index("horizon_s")


def healing_share(model, n: int = 30, low: float = 0.02) -> float:
    """Share of states in [low, 1]^2 where the model makes either side's quantity increase.

    A combat model should never do that (no healing or reinforcements here).
    The grid covers states a battle may never reach, so read it as a check on
    the model's form, not on its accuracy. `low` keeps eliminated sides out.
    """
    g = np.linspace(low, 1.0, n)
    grid = np.array([[a, b] for a in g for b in g])
    return float((model.predict(grid) > 1e-9).any(axis=1).mean())
