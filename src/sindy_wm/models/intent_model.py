"""A team-level world model with control: SINDYc on a physics-informed library, for Blue's intents.

State (common scale, see `states.geometry`):

    x = [health_blue, health_red, alive_blue, alive_red, distance]
    shape = [depth_blue, depth_red, width_blue, width_red]     (held, or with equations of its own)

Control: u = [advance, hold, withdraw, focus], the one-hot movement and the focus-fire flag
of `envs.intents`.

Structure:

    reach    = alive * share(distance, shape)               algebraic closure (a normal CDF)
    health'  = sparse combination of {reach, alive} x {1, hold, withdraw, focus}
    distance'= sparse combination of movement gates x movement
    alive    : (log alive)' = p(u) (log health)'            units die as health is lost
    shape'   = sparse drift / relaxation terms per zone (far, contact) and movement

Every sparse fit is sequentially thresholded least squares (STLSQ, the SINDy optimizer), one
equation at a time on standardised features, so each equation has its own threshold.
"""

import numpy as np
import pandas as pd
import pysindy as ps
from scipy.optimize import curve_fit
from scipy.special import ndtr

STATE = ["health_blue", "health_red", "alive_blue", "alive_red", "distance"]
SHAPE = ["depth_blue", "depth_red", "width_blue", "width_red"]
REACH_COLS = ["reach_blue", "reach_red"]
CTRL = ["advance", "hold", "withdraw", "focus"]
MOVEMENTS = ("advance", "hold", "withdraw")
DT = 0.125
FAR_GATE = (7.6, 0.45)  # centre and width of the far / contact zone split, in centroid distance


def control_vector(movement: str, target: str) -> np.ndarray:
    """u for one intent."""
    return np.array(
        [movement == "advance", movement == "hold", movement == "withdraw", target == "focus"], float
    )


def add_controls(states: pd.DataFrame) -> pd.DataFrame:
    """Add the CTRL columns from `movement` / `target`, and fill the shape columns' gaps."""
    states = states.copy()
    for m in MOVEMENTS:
        states[m] = (states["movement"] == m).astype(float)
    states["focus"] = (states["target"] == "focus").astype(float)
    states[SHAPE] = states[SHAPE].fillna(0)
    states["distance"] = states.groupby(level=0)["distance"].ffill().bfill()
    return states


def add_rates(states: pd.DataFrame, cols=tuple(STATE + SHAPE), window: int = 9) -> pd.DataFrame:
    """Add d_<col>: smoothed derivatives per episode (Savitzky-Golay, as in the notebooks)."""
    smooth = ps.SmoothedFiniteDifference(smoother_kws={"window_length": window, "polyorder": 2})
    out = []
    for _, g in states.groupby(level=0, sort=False):
        if len(g) <= window:
            continue
        g = g.copy()
        t = g["t"].to_numpy()
        for c in cols:
            g["d_" + c] = np.asarray(smooth(g[[c]].to_numpy(), t=t))[:, 0]
        out.append(g)
    return pd.concat(out)


def r2(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def stlsq(theta: np.ndarray, y: np.ndarray, threshold: float, alpha: float = 1e-6, max_iter: int = 25):
    """Sequentially thresholded least squares for one equation.

    Features and target are standardised first, so `threshold` is an effect size (a
    coefficient in standard-deviation units) and one value suits equations of different
    scale. Returns coefficients in the original units.
    """
    scale = theta.std(axis=0) + 1e-12
    T, yscale = theta / scale, y.std()
    keep = np.ones(T.shape[1], bool)
    coef = np.zeros(T.shape[1])
    for _ in range(max_iter):
        if not keep.any():
            break
        A = T[:, keep]
        c = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ y)
        coef[:] = 0
        coef[keep] = c
        new = np.abs(coef) >= threshold * yscale
        if (new == keep).all():
            break
        keep = new
    return coef / scale


def equation(names, coef, lhs: str, digits: int = 3) -> str:
    terms = [f"{c:+.{digits}f} {n}" for n, c in zip(names, coef, strict=True) if c != 0]
    return f"({lhs})' = " + (" ".join(terms) if terms else "0")


# ---------- closures ----------
def share(d, own, opp, w_own, w_opp, c, b_own, b_opp, w0, a, k):
    """Share of a side's alive units with an enemy within reach, from the team geometry.

    A normal CDF in the centroid distance d. The enemy's depth shifts the switch (its front
    is ahead of its centroid), own depth widens it (own units cross the boundary at
    different moments), and a difference in formation width lengthens the effective
    distance (flank units are further from the enemy).
    """
    d_eff = np.sqrt(d**2 + k * (w_own - w_opp) ** 2)
    return ndtr((c + b_own * own + b_opp * opp - d_eff) / np.sqrt(w0**2 + (a * own) ** 2 + 1e-4))


RHO = 5 + 2 * 0.375  # marine reach: attack range plus two unit radii


def derived_share(d, own, opp, w_own, w_opp, n_opp, lam, beta, g, rho=RHO):
    """Share of a side's units in reach, derived from the geometry (alternative to `share`).

    A unit is in reach when the enemy's front unit is within `rho`. Own units spread around
    their centre with std `own` (depth); the enemy front stands about beta * opp * m(n_opp)
    ahead of its centre, with m(n) = sqrt(2 ln n) the typical maximum of n spread positions;
    a width mismatch lengthens the distance (lam). Fitted: lam, beta, g; `rho` is the game's.
    """
    m = np.sqrt(2 * np.log(np.maximum(n_opp, 1.0)))
    d_eff = np.sqrt(d**2 + lam * (w_own - w_opp) ** 2)
    return ndtr((rho - d_eff + beta * opp * m) / np.sqrt(0.05**2 + (g * own) ** 2 + 1e-4))


def team_shares(X, S, P):
    """(share of Blue in reach, share of Red in reach) from state X (N, 5) and shape S (N, 4)."""
    d, db, dr, wb, wr = X[:, 4], *S.T
    if P.get("form") == "derived":
        return (
            derived_share(d, db, dr, wb, wr, 10 * X[:, 3], *P["share_blue"]),
            derived_share(d, dr, db, wr, wb, 10 * X[:, 2], *P["share_red"]),
        )
    return share(d, db, dr, wb, wr, *P["share_blue"]), share(d, dr, db, wr, wb, *P["share_red"])


def fit_derived_share(rows: pd.DataFrame, side: str, rho: float = RHO, n: int = 200_000):
    """Fit lam, beta, g of `derived_share` for one side. Returns (parameters incl. rho, R²)."""
    a = rows[rows[f"alive_{side}"] > 0]
    a = a.sample(min(n, len(a)), random_state=0)
    opp = "red" if side == "blue" else "blue"
    X = (*_share_inputs(a, side), 10 * a[f"alive_{opp}"].to_numpy())
    y = (a[f"reach_{side}"] / a[f"alive_{side}"]).to_numpy()
    p, _ = curve_fit(lambda X, *p: derived_share(*X, *p, rho=rho), X, y, p0=(1.0, 1.0, 1.0), maxfev=20000)
    p = np.append(p, rho)
    return p, r2(a[f"reach_{side}"], a[f"alive_{side}"] * derived_share(*X, *p))


def _share_inputs(rows, side):
    opp = "red" if side == "blue" else "blue"
    cols = ["distance", f"depth_{side}", f"depth_{opp}", f"width_{side}", f"width_{opp}"]
    return tuple(rows[c].to_numpy() for c in cols)


def fit_share(rows: pd.DataFrame, side: str, n: int = 200_000):
    """Fit `share` for one side. Returns (parameters, R² of the predicted count of units in reach)."""
    a = rows[(rows["alive_blue"] > 0) & (rows["alive_red"] > 0)]
    a = a.sample(min(n, len(a)), random_state=0)
    X = _share_inputs(a, side)
    y = (a[f"reach_{side}"] / a[f"alive_{side}"]).to_numpy()
    p, _ = curve_fit(lambda X, *p: share(*X, *p), X, y, p0=(6, 0, 1, 0.3, 1, 1), maxfev=60000)
    return p, r2(a[f"reach_{side}"], a[f"alive_{side}"] * share(*X, *p))


def _close_gate(X, v, c, bb, br, eb, er, w, a):
    d, db, dr, wb, wr = X
    return -v * ndtr(
        (d - c - bb * db - br * dr - eb * wb - er * wr) / np.sqrt(w**2 + (a * (db + dr)) ** 2 + 1e-4)
    )


def _close_flee_gate(X, v, c, bb, br, eb, er, w, a, v2, c2, w2, b2):
    d, db = X[0], X[1]
    return _close_gate(X, v, c, bb, br, eb, er, w, a) + v2 * ndtr((c2 + b2 * db - d) / w2)


def fit_movement_gates(rows: pd.DataFrame, n: int = 100_000) -> dict:
    """Shape of the closing-speed gate for each movement (the nonlinear part of the distance law).

    advance / hold:  d' = -v gate(d, shape)            closing until in contact
    withdraw:        the same, plus + v2 flee(d)        Blue walks away from enemies that get close
    Returns {movement: parameters}; the speeds v are refitted by the sparse regression.
    """
    X = lambda a: tuple(a[c].to_numpy() for c in ["distance"] + SHAPE)  # noqa: E731
    gates = {}
    for m, f, p0 in [
        ("advance", _close_gate, (6, 7.6, 0, 0, 0, 0, 0.5, 0.5)),
        ("hold", _close_gate, (3, 8, 0, 0, 0, 0, 0.5, 0.5)),
        ("withdraw", _close_flee_gate, (3, 8.5, 0, 0, 0, 0, 0.3, 0.5, 2, 6, 0.5, 0)),
    ]:
        a = rows[rows["movement"] == m]
        a = a.sample(min(n, len(a)), random_state=0)
        gates[m], _ = curve_fit(f, X(a), a["d_distance"], p0=p0, maxfev=80000)
    return gates


def fit_closures(rows: pd.DataFrame) -> dict:
    """All nonlinear closure parameters: reach share per side and the movement gates."""
    P = {f"share_{side}": fit_share(rows, side)[0] for side in ["blue", "red"]}
    P["gates"] = fit_movement_gates(rows)
    return P


# ---------- the candidate library ----------
def library(X, U, shape, P, reach=None, ctrl=True) -> dict:
    """Candidate features, name -> array, from state X (N, 5), control U (N, 4) and shape (N, 4).

    Fire family:     Rb, Rr (units in reach), Ab, Ar (alive), each x {1, hold, wd, focus}
    Movement family: close*<movement> (closing gates), flee*wd,
                     and generic alternatives: gate, near, walk_r, walk_b, danger x movement
    With `reach` (N, 2) given, it replaces the closure (an oracle, or for fitting on real reach).
    """
    Hb, Hr, Ab, Ar, d = X.T
    adv, hold, wd, focus = U.T
    db, dr, wb, wr = shape.T
    sb, sr = team_shares(X, shape, P)
    if reach is not None:
        sb = np.divide(reach[:, 0], Ab, out=np.zeros_like(Ab), where=Ab > 0)
        sr = np.divide(reach[:, 1], Ar, out=np.zeros_like(Ar), where=Ar > 0)
    Rb, Rr = Ab * sb, Ar * sr
    one = np.ones_like(d)
    fire_ctrl = {"": one, "*hold": hold, "*wd": wd, "*focus": focus} if ctrl else {"": one}
    move_ctrl = {"*adv": adv, "*hold": hold, "*wd": wd} if ctrl else {"": one}
    F = {}
    for cn, c in fire_ctrl.items():
        for name, v in {"Rb": Rb, "Rr": Rr, "Ab": Ab, "Ar": Ar}.items():
            F[name + cn] = v * c
    p_danger = np.array(P["share_blue"], float)
    p_danger[-1 if P.get("form") == "derived" else 0] += 1.25  # the withdraw radius (7), not the reach
    generic = {
        "gate": ndtr((d - FAR_GATE[0]) / FAR_GATE[1]),
        "near": 1 - ndtr((d - 5.4) / 0.2),
        "walk_r": 1 - sr,  # Red units with nobody in reach: walking
        "walk_b": 1 - sb,
        # Blue units with an enemy within the withdraw radius
        "danger": team_shares(X, shape, P | {"share_blue": p_danger})[0],
    }
    for cn, c in {"": one, **move_ctrl}.items():
        for name, v in generic.items():
            F[name + cn] = v * c
    if ctrl:
        gates = {}
        for short, u in (("adv", adv), ("hold", hold), ("wd", wd)):
            g = P["gates"][{"adv": "advance", "hold": "hold", "wd": "withdraw"}[short]]
            _, c, bb, br, eb, er, w, a = g[:8]
            gates[short] = ndtr(
                (d - c - bb * db - br * dr - eb * wb - er * wr) / np.sqrt(w**2 + (a * (db + dr)) ** 2 + 1e-4)
            )
            F[f"close*{short}"] = u * gates[short]
        _, c2, w2, b2 = P["gates"]["withdraw"][8:]
        F["flee*wd"] = wd * ndtr((c2 + b2 * db - d) / w2)
        # Hold the line, split by contact: before contact only Red walks (the hold gate); once any
        # Blue unit has an enemy in reach, every Blue unit closes in (the advance gate). Fit with
        # exclude=("close*hold",) to let SINDy choose between the two.
        contact = 1 - (1 - np.clip(sb, 0, 1)) ** (10 * Ab)  # chance that at least one Blue unit is in reach
        F["close*hold*far"] = hold * gates["hold"] * (1 - contact)
        F["close*hold*contact"] = hold * gates["adv"] * contact
    F["1"] = one
    return F


FIRE_FAMILY = ("Rb", "Rr", "Ab", "Ar")
MOVE_FAMILY = ("close", "flee")
GENERIC_MOVE_FAMILY = ("gate", "near", "walk_r", "walk_b", "danger")
# control x alive candidates: excluding them says "an intent acts on the units in reach"
ALIVE_CONTROL_TERMS = tuple(f"{a}*{c}" for a in ("Ab", "Ar") for c in ("hold", "wd", "focus"))


def _family_cols(names, family):
    return [j for j, n in enumerate(names) if n.split("*")[0] in family]


def _shape_library(d, a, U) -> dict:
    """Features for one shape variable a: a drift and a relaxation term per zone and movement."""
    far = ndtr((d - FAR_GATE[0]) / FAR_GATE[1])
    F = {}
    for zone, z in (("far", far), ("contact", 1 - far)):
        for j, m in enumerate(("adv", "hold", "wd")):
            F[f"{zone}*{m}"] = z * U[:, j]
            F[f"a*{zone}*{m}"] = a * z * U[:, j]
    return F


class IntentModel:
    """The world model: fit on team-state rows with rates, roll out from any state under any intents."""

    def __init__(
        self, P: dict, ctrl: bool = True, fire_family=FIRE_FAMILY, move_family=MOVE_FAMILY, exclude=()
    ):
        """`exclude` removes named candidates from the library, e.g. `ALIVE_CONTROL_TERMS`."""
        self.P, self.ctrl = P, ctrl
        self.fire_family = fire_family
        self.move_family = move_family if ctrl else GENERIC_MOVE_FAMILY
        self.exclude = tuple(exclude)
        self.shape_coef = None

    # ---- fitting ----
    def fit(
        self,
        rows: pd.DataFrame,
        thr_health: float = 0.05,
        thr_distance: float = 0.05,
        thr_alive: float = 0.02,
        thr_shape: float | None = 0.03,
        reach_true: bool = False,
        verbose: bool = False,
        rate_prefix: str = "d_",
    ) -> "IntentModel":
        """Fit every equation. `rows`: steps with both sides alive, with CTRL and d_<col> columns.

        reach_true fits the health equations on the measured reach instead of the closure
        (cleaner firepower coefficients; the closure is still used when rolling out).
        rate_prefix names the derivative columns (default d_<col>, from `add_rates`).
        """
        rate = lambda col: rows[rate_prefix + col].to_numpy()  # noqa: E731
        X, U, shape = rows[STATE].to_numpy(), rows[CTRL].to_numpy(), rows[SHAPE].to_numpy()
        F = library(X, U, shape, self.P, rows[REACH_COLS].to_numpy() if reach_true else None, self.ctrl)
        self.names = list(F)
        theta = np.column_stack([F[n] for n in self.names])
        self.coef, self.fit_r2 = {}, {}
        for eq, family, thr in [
            ("health_blue", self.fire_family, thr_health),
            ("health_red", self.fire_family, thr_health),
            ("distance", self.move_family, thr_distance),
        ]:
            cols = [j for j in _family_cols(self.names, family) if self.names[j] not in self.exclude]
            c = np.zeros(len(self.names))
            c[cols] = stlsq(theta[:, cols], rate(eq), thr)
            self.coef[eq] = c
            self.fit_r2[eq] = r2(rate(eq), theta @ c)
        self.p = {}
        for side in ["blue", "red"]:
            ok = ((rows[f"health_{side}"] > 0.03) & (rows[f"alive_{side}"] > 0)).to_numpy()
            g = (rate(f"health_{side}") / rows[f"health_{side}"].to_numpy())[ok]
            y = (rate(f"alive_{side}") / rows[f"alive_{side}"].to_numpy())[ok]
            feats = {"1": g}
            if self.ctrl:
                feats |= {"hold": g * U[ok, 1], "wd": g * U[ok, 2], "focus": g * U[ok, 3]}
            T = np.column_stack(list(feats.values()))
            c = stlsq(T, y, thr_alive)
            self.p[side] = dict(zip(feats, c, strict=True))
            self.fit_r2[f"alive_{side}"] = r2(y, T @ c)
        if thr_shape is not None and self.ctrl:
            d = rows["distance"].to_numpy()
            self.shape_coef = {}
            for name in SHAPE:
                Fs = _shape_library(d, rows[name].to_numpy(), U)
                T = np.column_stack(list(Fs.values()))
                self.shape_coef[name] = stlsq(T, rate(name), thr_shape)
                self.shape_names = list(Fs)
                self.fit_r2[name] = r2(rate(name), T @ self.shape_coef[name])
        if verbose:
            self.print()
        return self

    def print(self) -> None:
        for eq in ["health_blue", "health_red", "distance"]:
            print(f"  {equation(self.names, self.coef[eq], eq)}      [R² {self.fit_r2[eq]:.2f}]")
        for side in ["blue", "red"]:
            terms = " ".join(f"{v:+.3f} {k}" for k, v in self.p[side].items() if v != 0)
            fit = self.fit_r2[f"alive_{side}"]
            print(f"  (log alive_{side})' = [{terms}] (log health_{side})'      [R² {fit:.2f}]")
        if self.shape_coef is not None:
            for name in SHAPE:
                names = [n.replace("a*", name + "*") for n in self.shape_names]
                print(f"  {equation(names, self.shape_coef[name], name)}      [R² {self.fit_r2[name]:.2f}]")

    def n_terms(self) -> int:
        n = sum(int((c != 0).sum()) for c in self.coef.values())
        return n + sum(int(v != 0) for p in self.p.values() for v in p.values())

    # ---- simulation ----
    def _p_of(self, side, U):
        p = self.p[side]
        out = np.full(len(U), p["1"])
        if self.ctrl:
            out = out + p["hold"] * U[:, 1] + p["wd"] * U[:, 2] + p["focus"] * U[:, 3]
        return out

    def _shape_step(self, d, shape, U):
        rates = [
            np.column_stack(list(_shape_library(d, shape[:, j], U).values())) @ self.shape_coef[name]
            for j, name in enumerate(SHAPE)
        ]
        return np.maximum(shape + DT * np.column_stack(rates), 0)

    def simulate(
        self,
        X0: np.ndarray,
        shape0: np.ndarray,
        U: np.ndarray,
        reach0: np.ndarray | None = None,
        shape: str = "hold",
        tau: float = 1.0,
        truth: dict | None = None,
        return_shape: bool = False,
    ):
        """Roll N start points forward under control sequences U (N, steps, 4), in Euler steps of DT.

        shape:  "hold" keeps depth and width at their start values, "model" integrates their equations.
        reach0: the measured reach at the start. If given, its offset from the closure decays
                with time constant tau (reach as a relaxing state) instead of being ignored.
        truth:  oracle inputs for diagnostics: {"reach": (N, steps+1, 2), "shape": ..., "state": ...}
                with keys in `use`, e.g. truth={"reach": YR}. NaNs (after a battle's end) are
                carried forward.
        Returns the states, (N, steps + 1, 5).
        """
        truth = truth or {}
        X, S = X0.astype(float).copy(), shape0.astype(float).copy()
        steps = U.shape[1]
        out, shapes = [X.copy()], [S.copy()]
        Ch = np.stack([self.coef["health_blue"], self.coef["health_red"]])
        offset = None
        if reach0 is not None:
            s0 = np.column_stack(team_shares(X, S, self.P))
            offset = reach0 / np.maximum(X[:, 2:4], 1e-9) - s0
        for n in range(steps):
            u = U[:, n]
            if "shape" in truth:
                S = _carry(truth["shape"], n)
            elif shape == "model" and n > 0:
                S = self._shape_step(X[:, 4], S, U[:, n - 1])
            if "distance" in truth:
                X[:, 4] = _carry(truth["distance"], n)
            if "alive" in truth:
                X[:, 2:4] = _carry(truth["alive"], n)
            reach = _carry(truth["reach"], n) if "reach" in truth else None
            if reach is None and offset is not None:
                s = np.column_stack(team_shares(X, S, self.P))
                reach = X[:, 2:4] * np.clip(s + offset * np.exp(-n * DT / tau), 0, 1)
            F = library(X, u, S, self.P, reach, self.ctrl)
            theta = np.column_stack([F[k] for k in self.names])
            H = np.maximum(X[:, :2] + DT * theta @ Ch.T, 0)
            ratio = np.divide(H, X[:, :2], out=np.zeros_like(H), where=X[:, :2] > 1e-9)
            A = X[:, 2:4] * np.column_stack(
                [ratio[:, 0] ** self._p_of("blue", u), ratio[:, 1] ** self._p_of("red", u)]
            )
            X = np.column_stack([H, A, np.maximum(X[:, 4] + DT * theta @ self.coef["distance"], 0)])
            out.append(X.copy())
            shapes.append(S.copy())
        if return_shape:
            return np.stack(out, axis=1), np.stack(shapes, axis=1)
        return np.stack(out, axis=1)


def _carry(M: np.ndarray, n: int) -> np.ndarray:
    """M[:, n], with NaNs (after a battle's end) replaced by the last real value."""
    v = M[:, n].copy()
    k = n
    while np.isnan(v).any() and k > 0:
        k -= 1
        bad = np.isnan(v).any(axis=1) if v.ndim > 1 else np.isnan(v)
        v[bad] = M[bad, k]
    return v


# ---------- evaluation data ----------
def make_batch(
    states: pd.DataFrame, horizon: int = 64, every: int = 8, history: int = 8, min_future: int = 8
) -> dict:
    """Start points (every `every` steps after `history`) with the controls and truth over the horizon.

    Returns arrays: X0 (N, 5), shape0 (N, 4), reach0 (N, 2), U (N, horizon, 4), and the truth
    Y (N, horizon+1, 5), Yshape, Yreach (NaN after a battle's end), plus slope (the state's
    rate over the last `history` steps, for the last-slope baseline), ep and k0.
    """
    keys = ["X0", "U", "Y", "Yshape", "Yreach", "slope", "ep", "k0"]
    out = {k: [] for k in keys}

    def future(M, k0):
        y = np.full((horizon + 1, M.shape[1]), np.nan)
        m = min(horizon + 1, len(M) - k0)
        y[:m] = M[k0 : k0 + m]
        return y, m

    for i, g in states.groupby(level=0, sort=False):
        S, C, A, R = (g[c].to_numpy() for c in (STATE, CTRL, SHAPE, REACH_COLS))
        for k0 in range(history, len(S) - min_future, every):
            y, m = future(S, k0)
            u = np.repeat(C[k0 + m - 2][None], horizon, 0)  # after the battle's end: the last intent
            u[: m - 1] = C[k0 : k0 + m - 1]
            for key, value in zip(
                keys,
                [
                    S[k0],
                    u,
                    y,
                    future(A, k0)[0],
                    future(R, k0)[0],
                    (S[k0] - S[k0 - history]) / (history * DT),
                    i,
                    k0,
                ],
                strict=True,
            ):
                out[key].append(value)
    B = {k: np.array(v) for k, v in out.items()}
    B["shape0"], B["reach0"] = B["Yshape"][:, 0], B["Yreach"][:, 0]
    return B


def horizon_errors(pred: np.ndarray, B: dict, horizons=(1, 2, 4, 8)) -> pd.Series:
    """Mean absolute error of health (both sides) and of distance at each horizon (seconds)."""
    rows = {}
    for h in horizons:
        n = round(h / DT)
        ok = ~np.isnan(B["Y"][:, n, 0])
        rows[f"health {h} s"] = np.mean(np.abs(pred[ok, n, :2] - B["Y"][ok, n, :2]))
        rows[f"distance {h} s"] = np.mean(np.abs(pred[ok, n, 4] - B["Y"][ok, n, 4]))
    return pd.Series(rows)
