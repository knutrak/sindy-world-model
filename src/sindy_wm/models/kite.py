"""The pipeline with kite as a fifth control (docs/planning/32_kite_refit_plan.md).

Everything as `pipeline.Pipeline`, plus:

    health      H_st' = sum_a k_{a,t} R_oa s_{oa,t} + sum_a c_{a,t} R_oa s_{oa,t} kite + ...
                (the kite terms are controls: pruned at half the health threshold; state terms keep it)
    distance    a kite closing gate, close*kite, fitted as the advance gate (same form and starting values)

U has the columns CTRL + ["kite"]; a kite step is (0, 0, 0, focus, 1).
Batches need `L.batch(..., ctrl=CTRL_K)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import ndtr

from sindy_wm.models.intent_model import CTRL, SHAPE, _close_gate
from sindy_wm.models.pipeline import Pipeline

CTRL_K = [*CTRL, "kite"]


def control_vector_k(movement: str, target: str) -> np.ndarray:
    """u for one intent, five controls."""
    return np.array(
        [
            movement == "advance",
            movement == "hold",
            movement == "withdraw",
            target == "focus",
            movement == "kite",
        ],
        float,
    )


def add_kite(states: pd.DataFrame) -> pd.DataFrame:
    states = states.copy()
    states["kite"] = (states["movement"] == "kite").astype(float)
    return states


def stlsq_per_column(theta, y, thresholds, alpha: float = 1e-6, max_iter: int = 25) -> np.ndarray:
    """`intent_model.stlsq` with one threshold per column (standardised effect sizes)."""
    scale = theta.std(axis=0) + 1e-12
    T, yscale = theta / scale, y.std()
    thresholds = np.asarray(thresholds, float)
    keep = np.ones(T.shape[1], bool)
    coef = np.zeros(T.shape[1])
    for _ in range(max_iter):
        if not keep.any():
            break
        A = T[:, keep]
        c = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ y)
        coef[:] = 0
        coef[keep] = c
        new = np.abs(coef) >= thresholds * yscale
        if (new == keep).all():
            break
        keep = new
    return coef / scale


class KitePipeline(Pipeline):
    ctrl_cols = CTRL_K
    control_vector = staticmethod(control_vector_k)

    def fit(self, rows: pd.DataFrame) -> KitePipeline:
        """`rows`: from `pipeline.prepare`, with a "kite" column (`add_kite`)."""
        self._u = rows[CTRL_K].to_numpy(float)
        super().fit(rows)
        self._u = None
        return self

    def _terms(self, R_o, sig, A_o, wd, o, t) -> dict:
        L = super()._terms(R_o, sig, A_o, wd, o, t)
        kite = self._u[:, 4]
        for a in self.L.types:
            for t2 in self.L.types:
                L[f"R[{a}]*s[{t2}]*kite"] = L[f"R[{a}]*s[{t2}]"] * kite
        return L

    def _health_stlsq(self, theta, y, names) -> np.ndarray:
        thr = [self.thr_health / 2 if n.endswith("*kite") else self.thr_health for n in names]
        return stlsq_per_column(theta, y, thr)

    def _fit_extra_gates(self, T: pd.DataFrame) -> None:
        a = T[T["movement"] == "kite"]
        a = a.sample(min(100_000, len(a)), random_state=0)
        s = self.gate_scale
        p0 = (6, 7.6 * s, 0, 0, 0, 0, 0.5 * s, 0.5)  # advance's starting values
        X = tuple(a[c].to_numpy() for c in ["distance", *SHAPE])
        self.P_team["gates"]["kite"], _ = curve_fit(_close_gate, X, a["d_distance"], p0=p0, maxfev=80000)

    def _dist_library(self, X, U, S, reach) -> dict:
        F = super()._dist_library(X, U[:, :4], S, reach)
        kite = U[:, 4]
        d = X[:, -1]
        db, dr, wb, wr = S.T
        _, c, bb, br, eb, er, w, a = self.P_team["gates"]["kite"][:8]
        g = ndtr(
            (d - c - bb * db - br * dr - eb * wb - er * wr) / np.sqrt(w**2 + (a * (db + dr)) ** 2 + 1e-4)
        )
        F["close*kite"] = kite * g
        return F

    def rollout(self, B: dict) -> np.ndarray:
        try:
            return super().rollout(B)
        finally:
            self._u = None
