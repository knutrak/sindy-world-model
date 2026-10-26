"""SINDy on a library where every term is gated by the opponent: no healing once a side is gone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pysindy as ps

from sindy_wm.models.trajectories import derivative_data


class GatedModel:
    """Each side's losses are the opponent's strength times a polynomial "firepower" factor.

        dX1/dt = -X2 * p1(X1, X2),    dX2/dt = -X1 * p2(X2, X1)

    p_i is a polynomial up to `degree` in the side's *own* quantity and the
    opponent's: degree 0 is the Lanchester square law (p_i constant), degree 1
    adds `own` and `opp`, degree 2 adds `own^2`, `own*opp` and `opp^2`. Because
    every term carries the opponent's quantity, a side's quantity never changes
    once the opponent is eliminated: no "healing" after a battle is over, by
    construction. (Inside the battle p_i could still turn negative; check with
    `healing_share`.)

    Same interface as `LanchesterModel`, so it drops into `simulate`,
    `evaluate`, `summarize_eval`, `evaluate_horizons`, `plot_predictions` and
    `save_run`.

    Attributes:
        state_columns: Blue's quantity first, Red's second.
        degree: Degree of the firepower polynomials.
        coef: Array (2, n_features). Row i holds p_i's coefficients, in the
            order of `exponents`; a positive coefficient increases losses.
    """

    def __init__(self, state_columns: list[str], degree: int, coef: np.ndarray):
        self.state_columns = state_columns
        self.degree = degree
        self.coef = np.asarray(coef, dtype=float)

    @property
    def exponents(self) -> list[tuple[int, int]]:
        """(power of own, power of opp) of each term of p_i: 1, own, opp, own^2, own*opp, opp^2, ..."""
        return [(total - j, j) for total in range(self.degree + 1) for j in range(total + 1)]

    def features(self, x: np.ndarray, side: int) -> np.ndarray:
        """Library of side `side`'s equation at the rows of `x`: -opp * own^i * opp^j for each term."""
        x = np.atleast_2d(x)
        own, opp = x[:, side], x[:, 1 - side]
        return np.column_stack([-opp * own**i * opp**j for i, j in self.exponents])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Time derivatives at the rows of `x`, like `ps.SINDy.predict`."""
        return np.column_stack([self.features(x, side) @ self.coef[side] for side in (0, 1)])

    def simulate(self, x0, t, integrator_kws: dict | None = None) -> np.ndarray:
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            lambda _t, x: self.predict(x)[0], (t[0], t[-1]), x0, t_eval=t, **(integrator_kws or {})
        )
        return sol.y.T

    def coefficients(self) -> np.ndarray:
        return self.coef

    def get_feature_names(self) -> list[str]:
        # Generic names: the same feature means "-opp * ..." in both equations
        return [_monomial({"opp": 1 + j, "own": i}, sign="-") for i, j in self.exponents]

    def equations(self, precision: int = 3) -> list[str]:
        eqs = []
        for side in (0, 1):
            own, opp = self.state_columns[side], self.state_columns[1 - side]
            terms = [
                f"{-c:.{precision}f} {_monomial({opp: 1 + j, own: i})}"
                for c, (i, j) in zip(self.coef[side], self.exponents, strict=True)
                if c != 0
            ]
            eqs.append(" + ".join(terms) if terms else f"{0:.{precision}f}")
        return eqs


def _monomial(powers: dict[str, int], sign: str = "") -> str:
    """E.g. {"health_red": 2, "health_blue": 1} -> "health_red^2 health_blue"."""
    parts = [name if p == 1 else f"{name}^{p}" for name, p in powers.items() if p > 0]
    return sign + " ".join(parts)


def fit_gated(
    trajectories: dict[int, pd.DataFrame],
    train_ids,
    state_columns: list[str],
    smooth_window: int,
    degree: int,
    threshold: float,
) -> GatedModel:
    """Fit a `GatedModel`: SINDy's STLSQ, but on a library where every term is gated by the opponent.

    Uses the same Savitzky-Golay derivative estimate and STLSQ optimizer as
    `fit_sindy`, so the two are comparable; only the library differs.

    Args:
        trajectories: episode_idx -> trajectory, as returned by `to_trajectory`.
        train_ids: Episode indices to fit on.
        state_columns: The two state columns, Blue's quantity first and Red's second.
        smooth_window: Savitzky-Golay window length for the derivative, in points.
        degree: Degree of the firepower polynomial (see `GatedModel`).
        threshold: STLSQ sparsity threshold.
    """
    x, dx = derivative_data(trajectories, train_ids, state_columns, smooth_window)
    model = GatedModel(state_columns, degree, np.zeros((2, 0)))
    coef = []
    for side in (0, 1):
        optimizer = ps.STLSQ(threshold=threshold).fit(model.features(x, side), dx[:, [side]])
        coef.append(optimizer.coef_[0])
    model.coef = np.array(coef)
    return model
