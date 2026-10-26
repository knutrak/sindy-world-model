"""From a Level A table to the trajectories models are fitted on, and their derivatives.

trajectories = {i: to_trajectory(states.loc[i], STATE_COLUMNS, True, DOWNSAMPLE) for i in index.index}
x, dx = derivative_data(trajectories, train_ids, STATE_COLUMNS, SMOOTH_WINDOW)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pysindy as ps


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


def derivative_data(
    trajectories: dict[int, pd.DataFrame], ids, state_columns: list[str], smooth_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed states and their Savitzky-Golay derivatives, stacked over episodes.

    Exactly the data `fit_sindy` regresses on: `ps.SINDy` evaluates its library
    on the states smoothed by the same filter that gives the derivatives, not
    on the raw states, so `fit_lanchester` and `fit_gated` do the same. Each
    episode is smoothed and differentiated on its own, so nothing spans two episodes.
    """
    diff = ps.SmoothedFiniteDifference(smoother_kws={"window_length": smooth_window, "polyorder": 2})
    xs, dxs = [], []
    for i in ids:
        traj = trajectories[i]
        dxs.append(diff(traj[state_columns].to_numpy(), t=traj["t"].to_numpy()))
        xs.append(np.asarray(diff.smoothed_x_).copy())  # set by the call above
    return np.concatenate(xs), np.concatenate(dxs)
