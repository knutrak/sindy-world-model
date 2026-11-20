"""Weak-form (integral) fitting of the IntentModel equations, for notebooks/sindyc_07.

Instead of regressing smoothed derivatives on the library, each equation x' = Θ(x, u) ξ is
multiplied by a smooth test function φ that is zero at both ends of a window and integrated:

    −∫ φ'(t) x(t) dt = ∫ φ(t) Θ(t) dt · ξ          (integration by parts, no boundary terms)

so no derivative of the data is needed (Messenger & Bortz 2021, WSINDy; Schaeffer & McCalla 2017).
One row per window; the same library, families, exclusions and STLSQ thresholds as
`IntentModel.fit`. The alive rule (ln A)' = p(u) (ln H)' is integrated the same way on windows
where the intent is constant, so p multiplies −∫ φ' ln H dt.
"""

import copy

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import CTRL, DT, REACH_COLS, SHAPE, STATE, _family_cols, library, r2, stlsq


def bump(radius: int, power: int = 4):
    """φ and φ' on 2·radius + 1 grid points (time in seconds), φ = (1 − s²)^power, normalised to ∫φ = 1."""
    s = np.linspace(-1, 1, 2 * radius + 1)
    half = radius * DT
    phi = (1 - s**2) ** power
    dphi = power * (1 - s**2) ** (power - 1) * (-2 * s) / half
    w = np.full(len(s), DT)
    w[[0, -1]] = DT / 2  # trapezoid weights
    norm = (w * phi).sum()
    return phi / norm, dphi / norm, w


def weak_rows(rows: pd.DataFrame, P: dict, names: list, radius: int = 8, stride: int = 2, ctrl: bool = True):
    """Weak-form regression rows from whole battles (`rows`: one battle's steps in order per episode).

    `names`: the library columns to use (a fitted model's `names`, so the columns line up).
    Returns (G, b) for health and distance, and (Ga, ba) for the alive rule, as dicts of arrays.
    """
    phi, dphi, w = bump(radius)
    out = {k: [] for k in ("G", "b_hb", "b_hr", "b_d", "ga_b", "ga_r", "ba_b", "ba_r", "ctrl_b", "ctrl_r")}
    for _, g in rows.groupby(level=0, sort=False):
        X, U, S = g[STATE].to_numpy(float), g[CTRL].to_numpy(float), g[SHAPE].to_numpy(float)
        R = g[REACH_COLS].to_numpy(float)
        both = (X[:, 2] > 0) & (X[:, 3] > 0)
        if both.sum() < 2 * radius + 1:
            continue
        F = library(X, U, S, P, R, ctrl)
        T = np.column_stack([F[n] for n in names])
        lnA = np.log(np.maximum(X[:, 2:4], 1e-9))
        lnH = np.log(np.maximum(X[:, :2], 1e-9))
        n = len(X)
        for c in range(radius, n - radius, stride):
            sl = slice(c - radius, c + radius + 1)
            if not both[sl].all():
                continue
            out["G"].append((w * phi) @ T[sl])
            out["b_hb"].append(-(w * dphi) @ X[sl, 0])
            out["b_hr"].append(-(w * dphi) @ X[sl, 1])
            out["b_d"].append(-(w * dphi) @ X[sl, 4])
            constant = (U[sl] == U[c]).all()
            for j, t in enumerate("br"):
                ok = constant and (X[sl, j] > 0.03).all()
                out[f"ga_{t}"].append(-(w * dphi) @ lnH[sl, j] if ok else np.nan)
                out[f"ba_{t}"].append(-(w * dphi) @ lnA[sl, j] if ok else np.nan)
                out[f"ctrl_{t}"].append(U[c])
    out = {k: np.array(v) for k, v in out.items()}
    out["names"] = names
    return out


def fit_weak(base, W: dict, thr_health=0.15, thr_distance=0.05, thr_alive=0.02):
    """A copy of IntentModel `base` with every coefficient refitted on weak-form rows W."""
    m = copy.deepcopy(base)
    assert W["names"] == m.names
    G = W["G"]
    m.coef, m.fit_r2 = {}, {}
    for eq, key, family, thr in (
        ("health_blue", "b_hb", m.fire_family, thr_health),
        ("health_red", "b_hr", m.fire_family, thr_health),
        ("distance", "b_d", m.move_family, thr_distance),
    ):
        cols = [j for j in _family_cols(m.names, family) if m.names[j] not in m.exclude]
        c = np.zeros(len(m.names))
        c[cols] = stlsq(G[:, cols], W[key], thr)
        m.coef[eq] = c
        m.fit_r2[eq] = r2(W[key], G @ c)
    m.p = {}
    for side, t in (("blue", "b"), ("red", "r")):
        ok = ~np.isnan(W[f"ga_{t}"])
        g, y, U = W[f"ga_{t}"][ok], W[f"ba_{t}"][ok], W[f"ctrl_{t}"][ok]
        feats = {"1": g, "hold": g * U[:, 1], "wd": g * U[:, 2], "focus": g * U[:, 3]}
        c = stlsq(np.column_stack(list(feats.values())), y, thr_alive)
        m.p[side] = dict(zip(feats, c, strict=True))
        m.fit_r2[f"alive_{side}"] = r2(y, np.column_stack(list(feats.values())) @ c)
    return m


def _gate_windows(rows: pd.DataFrame, movement: str, radius: int, stride: int = 2):
    """Sparse window matrices for one movement: Φ (∫φ ·) and D (−∫φ' ·) over stacked rows.

    Windows lie where both sides are alive and the movement is `movement` throughout, like the
    rows `fit_movement_gates` uses.
    """
    from scipy import sparse

    phi, dphi, w = bump(radius)
    keep, ri, ci, vp, vd = [], [], [], [], []
    offset, k = 0, 0
    for _, g in rows.groupby(level=0, sort=False):
        n = len(g)
        ok = ((g["alive_blue"] > 0) & (g["alive_red"] > 0) & (g["movement"] == movement)).to_numpy()
        for c in range(radius, n - radius, stride):
            sl = slice(c - radius, c + radius + 1)
            if ok[sl].all():
                cols = offset + np.arange(c - radius, c + radius + 1)
                ri.append(np.full(len(cols), k))
                ci.append(cols)
                vp.append(w * phi)
                vd.append(-w * dphi)
                k += 1
        keep.append(g)
        offset += n
    stacked = pd.concat(keep)
    ri, ci = np.concatenate(ri), np.concatenate(ci)
    shape = (k, offset)
    Phi = sparse.csr_matrix((np.concatenate(vp), (ri, ci)), shape=shape)
    D = sparse.csr_matrix((np.concatenate(vd), (ri, ci)), shape=shape)
    return stacked, Phi, D


def fit_gates_weak(rows: pd.DataFrame, P: dict, radius: int = 16):
    """The movement switch shapes of `fit_movement_gates`, refitted in weak form.

    Same functions and starting values (the current shapes); the residual per window is
    −∫φ' d dt − ∫φ · gate(d, shape; θ) dt instead of d_distance − gate at each step.
    Returns a copy of P with new "gates".
    """
    from scipy.optimize import least_squares

    from sindy_wm.models.intent_model import _close_flee_gate, _close_gate

    gates = {}
    for m, f in (("advance", _close_gate), ("hold", _close_gate), ("withdraw", _close_flee_gate)):
        stacked, Phi, D = _gate_windows(rows, m, radius)
        X = tuple(stacked[c].to_numpy() for c in ["distance"] + SHAPE)
        b = D @ stacked["distance"].to_numpy()
        p0 = np.asarray(P["gates"][m], float)

        def resid(q, Phi=Phi, f=f, X=X, b=b):
            return Phi @ f(X, *q) - b

        sol = least_squares(resid, p0, x_scale="jac", max_nfev=400)
        gates[m] = sol.x
    return dict(P, gates=gates)
