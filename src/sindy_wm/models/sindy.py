"""Fitting a polynomial SINDy model (pysindy's own model class)."""

from __future__ import annotations

import pandas as pd
import pysindy as ps


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
