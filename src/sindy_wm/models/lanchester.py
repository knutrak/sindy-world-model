"""The classical Lanchester square law, as an explicit 2-parameter baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sindy_wm.models.trajectories import derivative_data


class LanchesterModel:
    """A classical Lanchester square-law model: dX1/dt = -a*X2, dX2/dt = -b*X1.

    An explicit, 2-parameter alternative to a general SINDy fit: no constant
    term, no self-coupling, only the opposing-side cross term the square law
    predicts. Exposes just enough of `ps.SINDy`'s interface (`simulate`,
    `coefficients`, `get_feature_names`, `equations`) to drop into the same
    pipeline as a SINDy model -- `simulate`, `evaluate`, `summarize_eval`,
    `plot_predictions`, `save_run` all work on either unchanged.
    """

    def __init__(self, state_columns: list[str], a: float, b: float):
        self.state_columns = state_columns
        self.a = a
        self.b = b

    def coefficients(self) -> np.ndarray:
        # one row per state variable, one column per feature (the state variables themselves)
        return np.array([[0.0, -self.a], [-self.b, 0.0]])

    def get_feature_names(self) -> list[str]:
        return list(self.state_columns)

    def equations(self, precision: int = 3) -> list[str]:
        x1, x2 = self.state_columns
        return [f"{-self.a:.{precision}f} {x2}", f"{-self.b:.{precision}f} {x1}"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Time derivatives at the rows of `x`, like `ps.SINDy.predict`."""
        x = np.atleast_2d(x)
        return np.column_stack([-self.a * x[:, 1], -self.b * x[:, 0]])

    def simulate(self, x0, t, integrator_kws: dict | None = None) -> np.ndarray:
        from scipy.integrate import solve_ivp

        def rhs(_t, x):
            x1, x2 = x
            return [-self.a * x2, -self.b * x1]

        sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, **(integrator_kws or {}))
        return sol.y.T


def fit_lanchester(
    trajectories: dict[int, pd.DataFrame],
    train_ids,
    state_columns: list[str],
    dt: float,
    smooth_window: int,
) -> LanchesterModel:
    """Fit the classical Lanchester square law: dX1/dt = -a*X2, dX2/dt = -b*X1.

    Uses the same Savitzky-Golay derivative estimate as `fit_sindy` (so the two
    are comparable), but regresses it onto a fixed model form instead of
    searching a feature library: `a`/`b` are each a one-variable linear
    regression through the origin (no intercept, no self-term), the exact
    textbook square law -- not a SINDy fit that happened to sparsify to it.

    Args:
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        train_ids: Episode indices to fit on.
        state_columns: The two state columns, Blue's quantity first and Red's
            second, e.g. `["health_blue", "health_red"]`, `["alive_blue",
            "alive_red"]`, or `["engaged_blue", "engaged_red"]`.
        dt: Time step between rows of a trajectory, in seconds (after downsampling).
        smooth_window: Savitzky-Golay window length for the derivative, in points.
    """
    x, dx = derivative_data(trajectories, train_ids, state_columns, smooth_window)
    x1, x2, dx1, dx2 = x[:, 0], x[:, 1], dx[:, 0], dx[:, 1]

    # dX1/dt = -a*X2, through the origin: a = -<X2, dX1/dt> / <X2, X2>, and symmetrically for b
    a = -np.dot(x2, dx1) / np.dot(x2, x2)
    b = -np.dot(x1, dx2) / np.dot(x1, x1)
    return LanchesterModel(state_columns, float(a), float(b))
