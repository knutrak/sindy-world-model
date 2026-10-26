"""Integrating a fitted model forward in time, over a trajectory or until a side is eliminated.

Works for any model with `simulate` (and, for `simulate_to_elimination`,
`predict`): `ps.SINDy`, `LanchesterModel`, `GatedModel`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pysindy as ps


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


def simulate_to_elimination(
    model: ps.SINDy,
    x0: np.ndarray,
    t_max: float,
    threshold: float = 0.02,
    blow_up: float = 1e3,
) -> tuple[bool | None, float | None]:
    """Integrate the model from `x0` until one side is eliminated, without using the real end time.

    A side counts as eliminated once its quantity drops to `threshold` (a
    smooth model may approach 0 only slowly). Blue is `x0[0]`, Red is `x0[1]`.
    Integration stops at that moment (a solver event), so the elimination time
    is exact and a model that blows up is not integrated all the way to `t_max`.

    Args:
        model: A fitted SINDy model (or anything with the same `predict`).
        x0: Starting state, e.g. the real state at engagement or at midway.
        t_max: Give up after this many seconds.
        threshold: Value at which a side counts as eliminated.
        blow_up: Give up once any state's magnitude exceeds this.

    Returns:
        `(blue_won, t_elimination)`, or `(None, None)` if neither side is
        eliminated within `t_max` or the integration fails or blows up.
    """
    from scipy.integrate import solve_ivp

    x0 = np.asarray(x0, dtype=float)
    if min(x0[0], x0[1]) <= threshold:  # already eliminated: no crossing for the event to find
        return bool(x0[1] < x0[0]), 0.0

    def rhs(_t, x):
        return model.predict(x[np.newaxis, :])[0]

    def eliminated(_t, x):
        return min(x[0], x[1]) - threshold

    def blown_up(_t, x):
        return blow_up - np.max(np.abs(x))

    eliminated.terminal = blown_up.terminal = True
    try:
        sol = solve_ivp(
            rhs,
            (0.0, t_max),
            x0,
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
            events=[eliminated, blown_up],
        )
    except Exception:  # integration can fail if the model blows up
        return None, None
    if sol.status != 1 or len(sol.t_events[0]) == 0:  # no event, or blew up first
        return None, None
    blue, red = sol.y_events[0][0][:2]
    return bool(red < blue), float(sol.t_events[0][0])
