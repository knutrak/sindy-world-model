"""Fitting and evaluating a SINDy model on Level A trajectories.

trajectories = {i: to_trajectory(states.loc[i], ...) for i in index.index}
model = fit_sindy(trajectories, train_ids, STATE_COLUMNS, dt, SMOOTH_WINDOW, degree, threshold)
metrics = summarize_eval(model, test_ids, trajectories, STATE_COLUMNS, index["battle_won"])
save_run(RUNS_DIR, name, config, model, STATE_COLUMNS, metrics, per_episode, figures)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps

from sindy_wm.evaluation.plots import TEAM_COLORS
from sindy_wm.provenance import code_info


def to_trajectory(
    ep: pd.DataFrame, state_columns: list[str], trim_to_engagement: bool, downsample: int
) -> pd.DataFrame:
    """Preprocess one episode's Level A table into the rows used for fitting.

    Args:
        ep: Level A table of one episode (as returned by `states.loc[i]`), with
            a "t" column (time in seconds).
        state_columns: Level A columns used as the SINDy state, e.g.
            `["health_blue", "health_red"]`.
        trim_to_engagement: If True, start one step before the first point
            where `state_columns` sum to less than at step 0 (the first
            damage dealt), instead of at the start of the episode.
        downsample: Keep every k-th row, to reduce the staircase of
            individual shots.

    Returns:
        The trimmed, downsampled table, with "t" reset to start at 0.
    """
    ep = ep.reset_index(drop=True)
    if trim_to_engagement:
        total = ep[state_columns].sum(axis=1)
        damaged = np.flatnonzero(total.to_numpy() < total.iloc[0] - 1e-9)
        start = max(int(damaged[0]) - 1, 0) if len(damaged) else 0
        ep = ep.iloc[start:]
    ep = ep.iloc[::downsample]
    return ep.assign(t=ep["t"] - ep["t"].iloc[0])


def fit_sindy(
    trajectories: dict[int, pd.DataFrame],
    train_ids,
    state_columns: list[str],
    dt: float,
    smooth_window: int,
    degree: int,
    threshold: float,
) -> ps.SINDy:
    """Fit a polynomial SINDy model with an STLSQ optimizer on training episodes.

    Args:
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        train_ids: Episode indices to fit on.
        state_columns: Level A columns used as the SINDy state.
        dt: Time step between rows of a trajectory, in seconds (after downsampling).
        smooth_window: Savitzky-Golay window length for the derivative, in points.
        degree: Polynomial feature library degree.
        threshold: STLSQ sparsity threshold.
    """
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=degree),
        differentiation_method=ps.SmoothedFiniteDifference(
            smoother_kws={"window_length": smooth_window, "polyorder": 2}
        ),
    )
    x = [trajectories[i][state_columns].to_numpy() for i in train_ids]
    model.fit(x, t=dt, feature_names=state_columns)
    return model


def simulate(
    model: ps.SINDy, traj: pd.DataFrame, state_columns: list[str], start: int = 0
) -> np.ndarray | None:
    """Integrate the model from row `start` of a trajectory over the remaining time points.

    Returns None if integration fails or blows up (doesn't reach every time
    point, or produces non-finite values), rather than raising.
    """
    t = traj["t"].to_numpy()[start:]
    x0 = traj[state_columns].iloc[start].to_numpy()
    try:
        sim = model.simulate(x0, t - t[0], integrator_kws={"method": "LSODA", "rtol": 1e-6, "atol": 1e-8})
    except Exception:  # integration can fail if the model blows up
        return None
    return sim if len(sim) == len(t) and np.all(np.isfinite(sim)) else None


def evaluate(
    model: ps.SINDy,
    ids,
    trajectories: dict[int, pd.DataFrame],
    state_columns: list[str],
    battle_won: pd.Series,
) -> pd.DataFrame:
    """Metrics per episode, simulating from the start and from midway.

    Args:
        model: A fitted SINDy model.
        ids: Episode indices to evaluate on.
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        state_columns: Level A columns used as the SINDy state.
        battle_won: Whether Blue won, indexed by episode_idx (e.g. `index["battle_won"]`).
    """
    blue, red = state_columns.index("health_blue"), state_columns.index("health_red")
    rows = []
    for i in ids:
        traj = trajectories[i]
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
) -> dict:
    """RMSE and winner accuracy (vs. their naive baselines), averaged over `ids`."""
    ev = evaluate(model, ids, trajectories, state_columns, battle_won)
    blue_share = ev["blue_won"].mean()
    return {
        "rmse_start": float(ev["rmse_start"].mean()),
        "rmse_midway": float(ev["rmse_midway"].mean()),
        "winner_acc_start": float(ev["winner_correct_start"].mean()),
        "winner_acc_midway": float(ev["winner_correct_midway"].mean()),
        "majority_baseline": float(max(blue_share, 1 - blue_share)),
        "leader_baseline_midway": float(ev["leader_correct_midway"].mean()),
        "failed_simulations": int(ev["failed_start"].sum() + ev["failed_midway"].sum()),
        "n_terms": int(np.count_nonzero(model.coefficients())),
    }


def plot_predictions(
    model: ps.SINDy,
    ids,
    trajectories: dict[int, pd.DataFrame],
    state_columns: list[str],
    battle_won: pd.Series,
    title: str = "",
) -> plt.Figure:
    """Plot a few episodes: actual (solid), simulated from the start (dashed) and midway (dotted)."""
    fig, axes = plt.subplots(1, len(ids), figsize=(4 * len(ids), 3.5), sharey=True, squeeze=False)
    for ax, i in zip(axes[0], ids, strict=True):
        traj = trajectories[i]
        mid = len(traj) // 2
        sim_start = simulate(model, traj, state_columns)
        sim_mid = simulate(model, traj, state_columns, mid)
        for k, col in enumerate(state_columns):
            color = TEAM_COLORS.get(col.split("_")[-1], "k")
            ax.plot(traj["t"], traj[col], color=color, label=f"{col} actual")
            if sim_start is not None:
                ax.plot(traj["t"], sim_start[:, k], "--", color=color, label="model from start")
            if sim_mid is not None:
                ax.plot(
                    traj["t"].iloc[mid:],
                    sim_mid[:, k],
                    ":",
                    color=color,
                    linewidth=2,
                    label="model from midway",
                )
        outcome = "Blue won" if bool(battle_won.loc[i]) else "Blue lost"
        ax.set(title=f"episode {i} ({outcome})", xlabel="time since engagement (s)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("health (normalized)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_run(
    runs_dir: Path,
    name: str,
    config: dict,
    model: ps.SINDy,
    state_columns: list[str],
    metrics: dict,
    per_episode: pd.DataFrame,
    figures: dict[str, plt.Figure],
) -> Path:
    """Save a run: config, model, equations, metrics, per-episode results, and figures.

    An existing run folder is never overwritten (see the README's Runs section).

    Args:
        runs_dir: Folder to create the new run folder in, e.g. `paths.PROJECT_ROOT / "runs"`.
        name: Short name included in the run folder's name.
        config: Dataset, split, and preprocessing/model choices (whatever isn't
            already implied by `model`); `code_info()` is added automatically.
        model: The fitted SINDy model.
        state_columns: Level A columns used as the SINDy state (recorded in model.json).
        metrics: Summary metrics on the test episodes, e.g. from `summarize_eval`.
        per_episode: Metrics per test episode, e.g. from `evaluate`.
        figures: filename -> figure, saved under the run's figures/ folder.

    Returns:
        The new run folder's path.
    """
    run_dir = Path(runs_dir) / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    (run_dir / "figures").mkdir()

    def write_json(filename, obj):
        (run_dir / filename).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    write_json("config.json", {**config, "code": code_info()})
    write_json(
        "model.json",
        {
            "state_names": state_columns,
            "feature_names": model.get_feature_names(),
            "coefficients": model.coefficients().tolist(),  # one row per state variable
            "equations": model.equations(precision=5),
        },
    )
    (run_dir / "equations.txt").write_text(
        "\n".join(f"({s})' = {eq}" for s, eq in zip(state_columns, model.equations(precision=5), strict=True))
        + "\n",
        encoding="utf-8",
    )
    write_json("metrics.json", metrics)
    per_episode.to_csv(run_dir / "per_episode.csv")
    for filename, fig in figures.items():
        fig.savefig(run_dir / "figures" / filename, dpi=150, bbox_inches="tight")
    return run_dir
