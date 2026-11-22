"""Reach as a state, discovered from a generic library (docs/generalisation/07_share_weak_plan.md).

For each side s and attacker type a the share in reach s_sa = R_sa / A_sa is a state:

    s_sa' = sum_g sum_c xi_{g,c} * theta_g(blocks) * u_c

- state groups theta_g: monomials of degree <= D in the building blocks (s, d_a, depth_own,
  depth_opp, width_own, width_opp, A_own, A_opp [, clo]), distance gates Phi((d_a - c)/0.5) and
  s * gate at c = 4..8, and the rate groups (ln A_own)', s (ln A_own)' [, clo'];
- controls u_c: 1, adv, hold, wd, focus [, kite];
- two-level STLSQ (sparse on the state groups, the control interactions pruned at half the
  threshold), columns and target scaled by their root mean square, solved from Gram matrices (so
  bootstrap refits are weighted Grams).

Fitted in weak form (test functions on windows, no derivative of s) or on one-step SG-9 rates.
Rolled out inside the frozen pipeline through `Pipeline._reach`; everything else stays frozen.
"""

from __future__ import annotations

import copy
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.special import ndtr

from sindy_wm.models.intent_model import CTRL, DT, SHAPE
from sindy_wm.models.pipeline import OTHER, SIDES, Pipeline

GATES = (4.0, 5.0, 6.0, 7.0, 8.0)
GATE_W = 0.5
CONTROLS = ("1", "adv", "hold", "wd", "focus", "kite")
BLOW = 0.05  # a share outside [-0.05, 1.05] before clipping counts as a blow-up (plan 07, R2)
BLOCKS = ("s", "d", "depth_own", "depth_opp", "width_own", "width_opp", "A_own", "A_opp")


# ------------------------------------------------------------------ the library
def block_arrays(s, d_a, shape, A_own, A_opp, side: str, clo=None) -> dict:
    own, opp = (0, 1) if side == "blue" else (1, 0)
    b = {
        "s": s,
        "d": d_a,
        "depth_own": shape[:, own],
        "depth_opp": shape[:, opp],
        "width_own": shape[:, 2 + own],
        "width_opp": shape[:, 2 + opp],
        "A_own": A_own,
        "A_opp": A_opp,
    }
    if clo is not None:
        b["clo"] = clo
    return b


def group_names(degree: int, with_clo: bool) -> list[str]:
    blocks = [*BLOCKS, "clo"] if with_clo else list(BLOCKS)
    names = ["1"]
    for k in range(1, degree + 1):
        names += ["*".join(c) for c in combinations_with_replacement(blocks, k)]
    names += [f"gate{c:g}" for c in GATES] + [f"s*gate{c:g}" for c in GATES]
    names += ["dlnA", "s*dlnA"] + (["dclo"] if with_clo else [])
    return names


def state_groups(b: dict, rates: dict, names: list[str]) -> np.ndarray:
    """(n, groups): every group of `names` from the block arrays `b` and the rate arrays `rates`
    (dlnA, dclo)."""
    n = len(b["s"])
    out = np.empty((n, len(names)), np.float64)
    for j, nm in enumerate(names):
        if nm == "1":
            out[:, j] = 1.0
        elif nm.startswith("gate"):
            out[:, j] = ndtr((b["d"] - float(nm[4:])) / GATE_W)
        elif nm.startswith("s*gate"):
            out[:, j] = b["s"] * ndtr((b["d"] - float(nm[6:])) / GATE_W)
        elif nm == "dlnA":
            out[:, j] = rates["dlnA"]
        elif nm == "s*dlnA":
            out[:, j] = b["s"] * rates["dlnA"]
        elif nm == "dclo":
            out[:, j] = rates["dclo"]
        else:
            v = np.ones(n)
            for f in nm.split("*"):
                v = v * b[f]
            out[:, j] = v
    return out


def control_arrays(U: np.ndarray, controls) -> np.ndarray:
    """U: (n, 4 or 5) in CTRL (+ kite) order -> (n, len(controls))."""
    col = {"adv": 0, "hold": 1, "wd": 2, "focus": 3, "kite": 4}
    return np.column_stack([np.ones(len(U)) if c == "1" else U[:, col[c]] for c in controls])


def design(G: np.ndarray, C: np.ndarray) -> np.ndarray:
    """State groups (n, g) x controls (n, c) -> (n, g * c), group-major."""
    return (G[:, :, None] * C[:, None, :]).reshape(len(G), -1)


# ------------------------------------------------------------------ rows from data
def _share_series(g: pd.DataFrame, L, side: str, a: str, clo_fn=None):
    A = g[f"alive_{side}_{a}"].to_numpy(float)
    R = g[f"reach_{side}_{a}"].to_numpy(float)
    s = np.divide(R, A, out=np.zeros_like(A), where=A > 0)
    d_a = g["distance"].to_numpy(float) - g[f"front_{side}_{a}"].fillna(0).to_numpy(float)
    shape = g[SHAPE].fillna(0).to_numpy(float)
    A_opp = sum(g[f"alive_{OTHER[side]}_{t}"].to_numpy(float) for t in L.types)
    clo = clo_fn(g, side, a) if clo_fn is not None else None
    return s, d_a, shape, A, A_opp, clo


def closure_share_fn(model: Pipeline):
    """clo per step from the frozen pipeline's closure, on measured states."""
    L = model.L

    def f(g, side, a):
        X = g[L.x_cols].to_numpy(float)
        S = g[SHAPE].fillna(0).to_numpy(float)
        F = g[L.front_cols].fillna(0).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(model._shares(X, S, F)[(side, a)])

    return f


def bump(radius: int, power: int = 4):
    s = np.linspace(-1, 1, 2 * radius + 1)
    half = radius * DT
    phi = (1 - s**2) ** power
    dphi = power * (1 - s**2) ** (power - 1) * (-2 * s) / half
    w = np.full(len(s), DT)
    w[[0, -1]] = DT / 2
    norm = (w * phi).sum()
    return phi / norm, dphi / norm, w


def rows_for(
    states: pd.DataFrame,
    L,
    side: str,
    a: str,
    degree: int,
    controls,
    method: str,
    radius: int = 8,
    stride: int = 8,
    clo_fn=None,
    sg: int = 9,
):
    """Regression rows for one equation. method "weak": one row per window (G = ∫φΘ, b = −∫φ′s);
    "step": one row per `stride`-th step (Θ, SG rate of s). Returns (Theta float32, b, battle id)."""
    names = group_names(degree, clo_fn is not None)
    ctrl_cols = [*CTRL, "kite"] if "kite" in controls else CTRL
    phi, dphi, w = bump(radius)
    Th, bs, ids = [], [], []
    for ep, g in states.groupby(level=0, sort=False):
        s, d_a, shape, A, A_opp, clo = _share_series(g, L, side, a, clo_fn)
        alive = (A > 0) & (A_opp > 0) & (sum(g[f"alive_{side}_{t}"].to_numpy(float) for t in L.types) > 0)
        n = len(s)
        if n < 2 * radius + 3:
            continue
        lnA = np.log(np.maximum(A, 1e-9))
        rates = {"dlnA": np.r_[np.diff(lnA) / DT, 0.0]}
        if clo is not None:
            rates["dclo"] = (
                np.r_[np.diff(clo) / DT, 0.0]
                if method == "weak"
                else savgol_filter(clo, sg, 2, deriv=1, delta=DT)
            )
        b = block_arrays(s, d_a, shape, A, A_opp, side, clo)
        Theta = design(state_groups(b, rates, names), control_arrays(g[ctrl_cols].to_numpy(float), controls))
        if method == "weak":
            for c in range(radius, n - radius - 1, stride):
                sl = slice(c - radius, c + radius + 1)
                if not alive[c - radius : c + radius + 2].all():
                    continue
                Th.append(((w * phi) @ Theta[sl]).astype(np.float32))
                bs.append(-(w * dphi) @ s[sl])
                ids.append(ep)
        else:
            ds = savgol_filter(s, sg, 2, deriv=1, delta=DT)
            for k in range(sg // 2, n - sg // 2 - 1, stride):
                if alive[k - sg // 2 : k + sg // 2 + 2].all():
                    Th.append(Theta[k].astype(np.float32))
                    bs.append(ds[k])
                    ids.append(ep)
    return np.array(Th, np.float32), np.array(bs), np.array(ids), names


# ------------------------------------------------------------------ sparse fit from Gram matrices
def gram(Theta: np.ndarray, b: np.ndarray, weights=None, chunk: int = 20_000):
    """Weighted Gram matrix and moments in float64, in row chunks (memory)."""
    W = np.ones(len(b)) if weights is None else np.asarray(weights, float)
    p = Theta.shape[1]
    GtG, Gty = np.zeros((p, p)), np.zeros(p)
    for i in range(0, len(b), chunk):
        T = Theta[i : i + chunk].astype(np.float64)
        w = W[i : i + chunk]
        TW = T * w[:, None]
        GtG += TW.T @ T
        Gty += TW.T @ b[i : i + chunk]
    return GtG, Gty, float((W * b * b).sum()), float(W.sum())


def two_level_gram(GtG, Gty, yy, n, groups, base, threshold: float, max_iter: int = 50) -> np.ndarray:
    """`pipeline.two_level_stlsq` on Gram matrices, with root-mean-square scaling. Columns with zero RMS are
    never kept."""
    groups, base = np.asarray(groups), np.asarray(base, bool)
    scale = np.sqrt(np.maximum(np.diag(GtG), 0) / n)
    yr = np.sqrt(yy / n) + 1e-300
    live = scale > 1e-12
    sc = np.where(live, scale, 1.0)
    A_all = GtG / np.outer(sc, sc)
    r_all = Gty / sc / yr
    keep = live.copy()
    coef = np.zeros(len(scale))
    for _ in range(max_iter):
        coef[:] = 0
        if keep.any():
            idx = np.flatnonzero(keep)
            M = A_all[np.ix_(idx, idx)] / n
            coef[idx] = np.linalg.solve(M + 1e-6 * np.eye(len(idx)), r_all[idx] / n)
        new = keep.copy()
        for g in dict.fromkeys(groups):
            cols = (groups == g) & keep
            if not cols.any():
                continue
            if np.linalg.norm(coef[cols]) < threshold:
                new[groups == g] = False
            else:
                new[cols & ~base & (np.abs(coef) < threshold / 2)] = False
        if (new == keep).all():
            break
        keep = new
    coef[~keep] = 0
    return coef / sc * yr


def group_index(names, controls):
    groups = np.repeat(np.arange(len(names)), len(controls))
    base = np.tile(np.array([c == "1" for c in controls]), len(names))
    return groups, base


# ------------------------------------------------------------------ the model
class ReachStatePipeline(Pipeline):
    """The frozen pipeline with reach as a state. `eqs[(side, type)] = (names, controls, degree, coef)`;
    `eqs=None` keeps the frozen reach (closure + decaying offset), for comparisons with the same logging.
    `self.R_log` keeps the units in reach of the last rollout, (N, steps, 2) per side (team); `self.blown`
    flags start points whose share left [-0.05, 1.05] before clipping (or went NaN)."""

    @classmethod
    def from_frozen(cls, model: Pipeline, eqs: dict | None, with_clo: bool = False):
        m = copy.deepcopy(model)
        m.__class__ = cls
        m.eqs, m.with_clo = eqs, with_clo
        return m

    def _dist_library(self, X, U, S, reach) -> dict:
        return super()._dist_library(X, U[:, :4], S, reach)

    def rollout(self, B: dict) -> np.ndarray:
        self._state, self._prev, self._log = {}, None, []
        self.blown = np.zeros(len(B["X0"]), bool)
        try:
            Y = super().rollout(B)
        finally:
            log, self._log = self._log, None
        self.R_log = np.stack(log, axis=1)
        return Y

    def _reach(self, n, X, S, F, u, sh, offset, B) -> dict:
        L = self.L
        if self.eqs is None:
            R = super()._reach(n, X, S, F, u, sh, offset, B)
        else:
            if n == 0:
                for k in sh:
                    A0 = X[:, L.idx("alive", *k)]
                    self._state[k] = np.clip(
                        np.divide(B["reach0"][:, L.pos(*k)], A0, out=np.zeros_like(A0), where=A0 > 1e-9), 0, 1
                    )
            else:
                Xp, Sp, Fp, up, shp = self._prev
                for k, (names, controls, _deg, coef) in self.eqs.items():
                    side, a = k
                    A_p, A_n = Xp[:, L.idx("alive", *k)], X[:, L.idx("alive", *k)]
                    rates = {"dlnA": (np.log(np.maximum(A_n, 1e-9)) - np.log(np.maximum(A_p, 1e-9))) / DT}
                    clo = shp[k] if self.with_clo else None
                    if self.with_clo:
                        rates["dclo"] = (sh[k] - shp[k]) / DT
                    A_opp = sum(Xp[:, L.idx("alive", OTHER[side], t)] for t in L.types)
                    b = block_arrays(self._state[k], Xp[:, -1] - Fp[:, L.pos(*k)], Sp, A_p, A_opp, side, clo)
                    Th = design(state_groups(b, rates, names), control_arrays(up, controls))
                    new = self._state[k] + DT * (Th @ coef)
                    alive = A_n > 1e-9
                    self.blown |= alive & (~np.isfinite(new) | (new < -BLOW) | (new > 1 + BLOW))
                    self._state[k] = np.clip(np.nan_to_num(new), 0, 1)
            R = {k: X[:, L.idx("alive", *k)] * self._state[k] for k in sh}
        self._prev = (X.copy(), S.copy(), F.copy(), u.copy(), {k: v.copy() for k, v in sh.items()})
        self._log.append(np.column_stack([sum(R[(s, a)] for a in L.types) for s in SIDES]))
        return R
