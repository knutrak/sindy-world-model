"""One equation-discovery pipeline for any scenario (docs/generalisation/00_plan.md, WP1).

The only inputs are a scenario's per-type state tables (`scripts/build_typed_states.py`) and its unit
types; everything else is a fixed rule:

    states      per side and unit type: health (ten full units of the type = 1), units alive (/10), units
                in reach (/10); centroid distance; depth / width per side; each type's front offset
    reach       per attacker / target pair: min(attack range + both radii, 6), 6 = SMAClite's
                AGENT_TARGET_RANGE; the closure's rho per attacker type = the mean over the scenario's
                target types
    health      H_st' = sum_a k_{a,t} R_oa sigma_{oa,t}, STLSQ (threshold 0.15) on measured reach x measured
                share of fire; candidates also the cross terms, alive counts and a constant
    withdraw    "rule": Blue's reach counts R (1 - wd) (Blue units don't fire while withdrawing, as the
                scripts do), in fitting and rollouts; "fitted": a candidate R sigma wd for Blue attackers
    alive       A_st' = p_st Hhat_st' A_st / H_st (+ focus term for Red), least squares
    distance    A's movement library with the team's reach, STLSQ (0.05); movement gates by curve_fit,
                started from the marine values scaled by the scenario's mean attack reach
    closures    reach per type: derived share at rho, shifted by the type's front offset, measured start
                offset decaying in 1 s; target choice (2 types only): logistic in the logit of the target
                side's second-type share and the front offset between the types (sigma = 1 with one type)

With one unit type the health equation is A*-2k's (tested in WP2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit, logit, ndtr

from sindy_wm.models.intent_model import (
    ALIVE_CONTROL_TERMS,
    CTRL,
    DT,
    FAR_GATE,
    MOVE_FAMILY,
    SHAPE,
    _close_flee_gate,
    _close_gate,
    _family_cols,
    _shape_library,
    add_controls,
    add_rates,
    derived_share,
    library,
    r2,
    stlsq,
    team_shares,
)
from sindy_wm.states.typed import pair_reach, unit_stats

SIDES = ("blue", "red")
OTHER = {"blue": "red", "red": "blue"}
AGENT_TARGET_RANGE = 6.0
MARINE_REACH = 5.75
DIST_EXCLUDE = (*ALIVE_CONTROL_TERMS, "close*hold")  # as A (SPLIT_EXCLUDE)


@dataclass(frozen=True)
class Layout:
    """Column names and the typed state vector for a scenario's unit types (lower case, e.g. ("marine",))."""

    types: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.types)

    @cached_property
    def x_cols(self) -> list[str]:
        h = [f"health_{s}_{t}" for s in SIDES for t in self.types]
        a = [f"alive_{s}_{t}" for s in SIDES for t in self.types]
        return [*h, *a, "distance"]

    @cached_property
    def r_cols(self) -> list[str]:
        return [f"reach_{s}_{t}" for s in SIDES for t in self.types]

    @cached_property
    def front_cols(self) -> list[str]:
        return [f"front_{s}_{t}" for s in SIDES for t in self.types]

    def idx(self, kind: str, side: str, t: str) -> int:
        base = {"health": 0, "alive": 2 * self.n}[kind]
        return base + self.n * SIDES.index(side) + self.types.index(t)

    def pos(self, side: str, t: str) -> int:
        """Column of (side, type) in the reach / front blocks."""
        return self.n * SIDES.index(side) + self.types.index(t)

    @cached_property
    def hp(self) -> dict:
        return {t: unit_stats(t.upper())["hp"] + unit_stats(t.upper())["shield"] for t in self.types}

    def team_view(self, X: np.ndarray) -> np.ndarray:
        """Typed state (..., 4n+1) -> team state (..., 5): health in ten units of the first type, alive, d."""
        ref = self.hp[self.types[0]]
        out = [sum(X[..., self.idx("health", s, t)] * self.hp[t] / ref for t in self.types) for s in SIDES]
        out += [sum(X[..., self.idx("alive", s, t)] for t in self.types) for s in SIDES]
        out.append(X[..., -1])
        return np.stack(out, axis=-1)

    # ---- batches ----
    def batch(self, states: pd.DataFrame, horizon=64, every=8, history=8, min_future=8, ctrl=None) -> dict:
        """Start points (as `make_batch`): X0, U, Y (typed), shape0, front0, reach0, plus the team view.
        `ctrl`: the control columns of U (default CTRL; the kite variant adds "kite", docs/planning/32)."""
        keys = ["X0", "U", "Y", "shape0", "front0", "reach0", "ep", "k0"]
        out = {k: [] for k in keys}
        check_steps(states)
        for ep, g in states.groupby(level=0, sort=False):
            S, C = g[self.x_cols].to_numpy(float), g[ctrl or CTRL].to_numpy(float)
            Sh, Fr, R = (
                g[SHAPE].to_numpy(float),
                g[self.front_cols].to_numpy(float),
                g[self.r_cols].to_numpy(float),
            )
            for k0 in range(history, len(S) - min_future, every):
                m = min(horizon + 1, len(S) - k0)
                y = np.full((horizon + 1, S.shape[1]), np.nan)
                y[:m] = S[k0 : k0 + m]
                u = np.repeat(C[k0 + m - 2][None], horizon, 0)
                u[: m - 1] = C[k0 : k0 + m - 1]
                for key, v in zip(keys, [S[k0], u, y, Sh[k0], Fr[k0], R[k0], ep, k0], strict=True):
                    out[key].append(v)
        if not out["X0"]:
            raise ValueError("no battle is long enough for a start point")
        B = {k: np.array(v) for k, v in out.items()}
        B["front0"] = np.nan_to_num(B["front0"])
        n = self.n
        B["team"] = {
            "X0": self.team_view(B["X0"]),
            "Y": self.team_view(B["Y"]),
            "U": B["U"],
            "shape0": B["shape0"],
            "reach0": np.column_stack([B["reach0"][:, :n].sum(1), B["reach0"][:, n:].sum(1)]),
            "ep": B["ep"],
            "group": B["ep"] % 10_000,
        }
        return B

    def battle_batch(self, states: pd.DataFrame, steps: int = 720) -> dict:
        """One start per battle at step 0, to its end; B["team"] has the harness's battle fields."""
        with np.errstate(invalid="ignore", divide="ignore"):
            B = self.batch(states, horizon=steps, every=10**9, history=0, min_future=1)
        won = states.groupby(level=0)["battle_won"].first()
        B["team"]["won"] = won.loc[B["ep"]].to_numpy().astype(bool)
        B["team"]["end"] = (~np.isnan(B["Y"][:, :, 0])).sum(1) - 1
        return B


# ------------------------------------------------------------------------------------------- data
def check_steps(states: pd.DataFrame) -> None:
    """Refuse tables the models would silently misread: another time step than DT, or two battles under one
    id (docs/triple/04_stress.md). Needs the `t` column (seconds); without it nothing is checked."""
    if "t" not in states or len(states) < 2:
        return
    ids = states.index.get_level_values(0)
    same = ids[1:] == ids[:-1]
    if (~same).sum() + 1 != ids.nunique():
        raise ValueError(
            "rows of one episode id are not together: two battles share an id, or rows are unsorted"
        )
    dt = np.diff(states["t"].to_numpy(float))[same]
    if len(dt) and not np.allclose(dt, DT, rtol=0, atol=1e-6):
        raise ValueError(
            f"consecutive steps of a battle must be {DT} s apart, found {dt.min():g} to {dt.max():g} s "
            "(another step_mul, dropped steps, unsorted rows, or two battles sharing an episode id)"
        )


def check_rows(rows: pd.DataFrame, layout: Layout, movements=("advance", "hold")) -> None:
    """Clear errors for data `Pipeline.fit` cannot use (they used to fail deep inside curve_fit or lstsq)."""
    if rows.empty:
        raise ValueError("no rows to fit on")
    for m in movements:
        if not (rows["movement"] == m).any():
            raise ValueError(
                f"no steps with movement {m!r}: the distance equation needs advance and hold steps"
            )
    for s in SIDES:
        for t in layout.types:
            if not (rows[f"alive_{s}_{t}"] > 0).any():
                raise ValueError(f"{s} never has a living {t}: fit without that type")
            if not (rows[f"reach_{s}_{t}"] > 0).any():
                raise ValueError(f"no {s} {t} ever has an enemy in reach: nothing to learn firepower from")
    cols = [
        *layout.x_cols,
        *layout.r_cols,
        *SHAPE,
        "d_distance",
        *(f"d_{k}_{s}_{t}" for k in ("health", "alive") for s in SIDES for t in layout.types),
    ]
    bad = [c for c in cols if not np.isfinite(rows[c].to_numpy(float)).all()]
    if bad:
        raise ValueError(f"missing or infinite values in {bad}")


def prepare(states: pd.DataFrame, layout: Layout, window: int = 9) -> pd.DataFrame:
    """Controls and SG rates (per battle) for the typed columns and distance; steps with both sides alive."""
    check_steps(states)
    s = add_controls(states)
    cols = [*layout.x_cols]
    r = add_rates(s, cols=tuple(cols), window=window)
    alive = {side: sum(r[f"alive_{side}_{t}"] for t in layout.types) for side in SIDES}
    return r[(alive["blue"] > 0) & (alive["red"] > 0)]


def team_rows(rows: pd.DataFrame, layout: Layout) -> pd.DataFrame:
    """Team columns for the distance equation (alive and reach summed over types)."""
    T = rows.copy()
    ref = layout.hp[layout.types[0]]
    for s in SIDES:
        T[f"health_{s}"] = sum(T[f"health_{s}_{t}"] * layout.hp[t] / ref for t in layout.types)
        T[f"alive_{s}"] = sum(T[f"alive_{s}_{t}"] for t in layout.types)
        T[f"reach_{s}"] = sum(T[f"reach_{s}_{t}"] for t in layout.types)
    return T


def fit_gates(rows: pd.DataFrame, scale: float, skip=()) -> dict:
    """`intent_model.fit_movement_gates` with distance-like starting values scaled by `scale`."""
    X = lambda a: tuple(a[c].to_numpy() for c in ["distance", *SHAPE])  # noqa: E731
    gates = {}
    for m, f, p0 in [
        ("advance", _close_gate, (6, 7.6 * scale, 0, 0, 0, 0, 0.5 * scale, 0.5)),
        ("hold", _close_gate, (3, 8 * scale, 0, 0, 0, 0, 0.5 * scale, 0.5)),
        (
            "withdraw",
            _close_flee_gate,
            (3, 8.5 * scale, 0, 0, 0, 0, 0.3 * scale, 0.5, 2, 6 * scale, 0.5 * scale, 0),
        ),
    ]:
        a = rows[rows["movement"] == m]
        if a.empty or m in skip:
            continue
        a = a.sample(min(100_000, len(a)), random_state=0)
        gates[m], _ = curve_fit(f, X(a), a["d_distance"], p0=p0, maxfev=80000)
    return gates


# ------------------------------------------------------------------------------------------- model
def _alloc(pi, dfront, b):
    z = b[0] + b[1] * logit(np.clip(pi, 1e-4, 1 - 1e-4)) + b[2] * dfront
    return np.where(pi <= 0, 0.0, np.where(pi >= 1, 1.0, expit(z)))


def intent_features(Fd: dict, TX: np.ndarray, U: np.ndarray) -> dict:
    """Extra focus candidates for the distance equation (exploratory variant): focus split by contact, where
    contact is the chance that at least one Blue unit has an enemy in reach (as in the movement library)."""
    sb = np.clip(1 - Fd["walk_b"], 0, 1)
    contact = 1 - (1 - sb) ** (10 * TX[:, 2])
    f = U[:, 3] * (1 - U[:, 2])
    return {"focus*contact": f * contact, "focus*far": f * (1 - contact)}


def shape_step(shape_eq: dict, d: np.ndarray, S: np.ndarray, U: np.ndarray) -> np.ndarray:
    """One step of the shape equations (exploratory variant): `shape_eq[j] = (names, coef)` for SHAPE
    column j, on `_shape_library` (drift and relaxation per zone and movement)."""
    S = S.copy()
    for j, (names, c) in shape_eq.items():
        F = _shape_library(d, S[:, j], U)
        S[:, j] = np.maximum(S[:, j] + DT * np.column_stack([F[k] for k in names]) @ c, 0)
    return S


SPREAD_ALIVE_GROUPS = ("1", "(1-s)", "(1-s)^2")
SPREAD_ALIVE_CTRL = ("1", "focus", "hold", "wd")
SPREAD_RATE_GROUPS = ("1", "s", "r", "r*s")
SPREAD_RATE_CTRL = ("1", "focus", "hold", "wd", "focus*far", "hold*far")


def _ctrl(U: np.ndarray, d: np.ndarray, names) -> dict:
    far = ndtr((d - FAR_GATE[0]) / FAR_GATE[1])
    c = {"1": np.ones(len(U)), "focus": U[:, 3], "hold": U[:, 1], "wd": U[:, 2]}
    c["focus*far"], c["hold*far"] = U[:, 3] * far, U[:, 1] * far
    return {k: c[k] for k in names}


def spread_alive_features(s: np.ndarray, U: np.ndarray, d: np.ndarray) -> dict:
    """The spread variant's alive exponent p(s, u) = sum of coef * feature (docs/planning/09): state groups
    1, (1 - s), (1 - s)^2, each times the controls 1, focus, hold, wd. Keys "group|control"."""
    phi = {"1": np.ones_like(s), "(1-s)": 1 - s, "(1-s)^2": (1 - s) ** 2}
    psi = _ctrl(U, d, SPREAD_ALIVE_CTRL)
    return {f"{g}|{c}": phi[g] * psi[c] for g in SPREAD_ALIVE_GROUPS for c in SPREAD_ALIVE_CTRL}


def spread_rate_features(s: np.ndarray, r: np.ndarray, U: np.ndarray, d: np.ndarray) -> dict:
    """The spread variant's s' library: state groups 1, s, r, r*s (r = relative rate of health loss), each
    times the controls 1, focus, hold, wd, focus*far, hold*far. Keys "group|control"."""
    phi = {"1": np.ones_like(s), "s": s, "r": r, "r*s": r * s}
    psi = _ctrl(U, d, SPREAD_RATE_CTRL)
    return {f"{g}|{c}": phi[g] * psi[c] for g in SPREAD_RATE_GROUPS for c in SPREAD_RATE_CTRL}


def group_stlsq(
    theta: np.ndarray, y: np.ndarray, groups: list[str], threshold: float, always=()
) -> np.ndarray:
    """Sequentially thresholded least squares over groups of columns (sparse on the state groups, the control
    interactions inside a kept group dense): standardised features and target; a group stays while the L2 norm
    of its standardised coefficients is at least `threshold`. Returns coefficients in the original units."""
    scale = theta.std(axis=0) + 1e-12
    T, ys = theta / scale, y / (y.std() + 1e-12)
    names = list(dict.fromkeys(groups))
    keep = {g: True for g in names}
    coef = np.zeros(T.shape[1])
    for _ in range(25):
        cols = np.array([keep[g] for g in groups])
        coef[:] = 0
        if cols.any():
            A = T[:, cols]
            coef[cols] = np.linalg.solve(A.T @ A + 1e-6 * len(A) * np.eye(A.shape[1]), A.T @ ys)
        new = {g: g in always or np.linalg.norm(coef[np.array(groups) == g]) >= threshold for g in names}
        if new == keep:
            break
        keep = new
    return coef / scale * (y.std() + 1e-12)


COMPART_GROUPS = {  # docs/planning/13_compartment_plan.md
    "I": ("d(1-w)", "d(1-w)/mF", "d(1-w)rho", "dw"),
    "KW": ("dw", "dw/mW", "W"),
    "KF": ("d(1-w)", "d(1-w)/mF"),
    "sigma": ("w", "w2", "w(1-mW)"),
}
COMPART_ALWAYS = {"I": "d(1-w)", "KW": "dw", "KF": "d(1-w)", "sigma": "w"}
COMPART_CTRL = ("1", "focus", "hold", "wd")
HW_GROUPS = ("d(1-w)/mF", "dw/mW", "w(1-mW)")  # the groups that need H_W


def compart_features(law: str, d, w, Wn, mF, mW, rho, U, use_hw: bool = True) -> dict:
    """Features of one compartment law: state groups (COMPART_GROUPS) times the controls 1, focus, hold, wd.
    d: health loss rate (full units / s); w: wounded share; Wn: wounded units; mF, mW: healthy / wounded mean
    health over the threshold; rho: share of units in reach. Keys "group|control"."""
    phi = {
        "d(1-w)": d * (1 - w),
        "d(1-w)/mF": d * (1 - w) / np.maximum(mF, 0.5),
        "d(1-w)rho": d * (1 - w) * rho,
        "dw": d * w,
        "dw/mW": d * w / np.maximum(mW, 0.05),
        "W": Wn,
        "w": w,
        "w2": w**2,
        "w(1-mW)": w * (1 - mW),
    }
    psi = {"1": np.ones_like(d), "focus": U[:, 3], "hold": U[:, 1], "wd": U[:, 2]}
    groups = [g for g in COMPART_GROUPS[law] if use_hw or g not in HW_GROUPS]
    return {f"{g}|{c}": phi[g] * psi[c] for g in groups for c in COMPART_CTRL}


def compart_derived(A10, W10, H10, HW10, h0):
    """From the table scales (alive and W in units / 10, health and H_W in ten full units) to counts and the
    derived quantities of the compartment laws: (A, W, w, mF, mW) with A, W in units."""
    A, Wn = 10 * A10, 10 * W10
    H, HW = 10 * H10, 10 * HW10
    w = np.divide(Wn, A, out=np.zeros_like(A), where=A > 1e-9)
    thr = 0.5 * np.maximum(h0, 1e-3)
    mF = np.divide(H - HW, (A - Wn) * thr, out=np.ones_like(A), where=(A - Wn) > 1e-9)
    mW = np.divide(HW, Wn * thr, out=np.ones_like(A), where=Wn > 1e-9)
    return A, Wn, w, mF, mW


def compart_context(L, X, W10, U) -> dict:
    """Wounded shares per (side, type) and per side (team), and the controls, for one rollout step."""
    w, team = {}, {}
    for side in SIDES:
        Ws = As = 0
        for t in L.types:
            A10 = X[:, L.idx("alive", side, t)]
            Wt = W10[:, L.pos(side, t)]
            w[(side, t)] = np.divide(Wt, A10, out=np.zeros_like(A10), where=A10 > 1e-9)
            Ws, As = Ws + Wt, As + A10
        team[side] = np.divide(Ws, As, out=np.zeros_like(As), where=As > 1e-9)
    return {"w": w, "team": team, "U": U}


def two_level_stlsq(theta, y, groups, threshold: float, base, always=()) -> np.ndarray:
    """STLSQ in two levels (docs/planning/15): a group (a state term with its control interactions) stays
    while the
    L2 norm of its standardised coefficients is at least `threshold`; inside a kept group, a non-base
    column (a
    control interaction) is dropped when its standardised coefficient is below threshold / 2. Returns the
    coefficients in the original units."""
    groups, base = np.asarray(groups), np.asarray(base, bool)
    scale = theta.std(axis=0) + 1e-12
    ysd = y.std() + 1e-12
    T, ys = theta / scale, y / ysd
    keep = np.ones(T.shape[1], bool)
    coef = np.zeros(T.shape[1])
    for _ in range(50):
        coef[:] = 0
        if keep.any():
            A = T[:, keep]
            coef[keep] = np.linalg.solve(A.T @ A + 1e-6 * len(A) * np.eye(A.shape[1]), A.T @ ys)
        new = keep.copy()
        for g in dict.fromkeys(groups):
            cols = (groups == g) & keep
            if not cols.any():
                continue
            if g not in always and np.linalg.norm(coef[cols]) < threshold:
                new[groups == g] = False
            else:
                new[cols & ~base & (np.abs(coef) < threshold / 2)] = False
        if (new == keep).all():
            break
        keep = new
    coef[~keep] = 0
    return coef / scale * ysd


def compart_step(comp: dict, key, A10, W10, H10, H10_next, HW10, h0, R10, U):
    """One rollout step of the compartment laws (docs/planning/13): returns alive, W and H_W (table
    scales)."""
    A, Wn, w, mF, mW = compart_derived(A10, W10, H10, HW10, h0)
    d = np.maximum(H10 - H10_next, 0) * 10 / DT  # health loss, full units per second
    rho = np.divide(R10, A10, out=np.zeros_like(A10), where=A10 > 1e-9)
    use_hw = comp.get("use_hw", True)
    rates = {}
    for law, (names, c) in comp["laws"][key].items():
        F = compart_features(law, d, w, Wn, mF, mW, rho, U, use_hw)
        rates[law] = np.column_stack([F[k] for k in names]) @ c if names else np.zeros_like(d)
    inflow, KW, KF = (np.maximum(rates[k], 0) for k in ("I", "KW", "KF"))
    sig = np.clip(rates["sigma"], 0, 1)
    none = Wn <= 1e-9
    KW, sig = np.where(none, 0.0, KW), np.where(none, 0.0, sig)
    A_new = np.where(H10_next <= 0, 0.0, np.maximum(A - DT * (KW + KF), 0))
    W_new = np.clip(Wn + DT * (inflow - KW), 0, A_new)
    HW_new = np.clip(10 * HW10 + DT * (-sig * d + inflow * 0.5 * h0), 0, 10 * H10_next)
    HW_new = np.where(W_new <= 1e-9, 0.0, HW_new)
    return A_new / 10, W_new / 10, HW_new / 10


class Pipeline:
    """The scenario-independent model (see the module docstring)."""

    ctrl_cols = CTRL  # the kite variant (models/kite.py) adds a fifth control

    def __init__(
        self,
        types: tuple[str, ...],
        withdraw: str = "rule",
        alloc: str = "closure",
        thr_health: float = 0.15,
        thr_distance: float = 0.05,
        max_reach: float = AGENT_TARGET_RANGE,
        movement_wd: str = "gates",
        withdraw_radius: float | None = None,
    ):
        """movement_wd: how the distance equation treats withdraw. "gates": A's fitted withdraw switches
        (close*wd, flee*wd); "rule": the scripts' rule, no switch fitted: candidates danger*wd (share of Blue
        with an enemy within `withdraw_radius`, which then walks away) and walk_r*wd (share of Red not in
        reach, which walks towards Blue)."""
        if movement_wd not in ("gates", "rule") or (movement_wd == "rule" and withdraw_radius is None):
            raise ValueError("movement_wd 'rule' needs withdraw_radius")
        self.movement_wd, self.withdraw_radius = movement_wd, withdraw_radius
        if withdraw not in ("rule", "fitted"):
            raise ValueError(withdraw)
        if len(types) > 2 and alloc != "rule":
            raise NotImplementedError("the fitted target choice is defined for one or two unit types")
        self.L = Layout(tuple(t.lower() for t in types))
        self.withdraw, self.alloc = withdraw, alloc
        self.thr_health, self.thr_distance, self.max_reach = thr_health, thr_distance, max_reach
        tt = self.L.types
        self.rho = {
            a: float(np.mean([min(pair_reach(a.upper(), t.upper()), max_reach) for t in tt])) for a in tt
        }
        self.gate_scale = (
            float(np.mean([pair_reach(a.upper(), t.upper()) for a in tt for t in tt])) / MARINE_REACH
        )

    # ---- terms ----
    def _sigma_measured(self, rows: pd.DataFrame, o: str) -> dict:
        tt = self.L.types
        if len(tt) == 1:
            return {(tt[0], tt[0]): np.ones(len(rows))}
        s = OTHER[o]
        if len(tt) > 2:  # docs/triple/00: measured share of fire per target type, proportional without fire
            al = np.column_stack([rows[f"alive_{s}_{t}"].to_numpy(float) for t in tt])
            sig = {}
            for a in tt:
                f = np.column_stack([rows[f"fire_{o}_{a}_{t}"].to_numpy(float) for t in tt])
                w = np.where(f.sum(1, keepdims=True) > 0, f, al)
                tot = w.sum(1, keepdims=True)
                w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
                sig |= {(a, t): w[:, j] for j, t in enumerate(tt)}
            return sig
        t0, t1 = tt
        tot = rows[f"alive_{s}_{t0}"] + rows[f"alive_{s}_{t1}"]
        prop = np.divide(rows[f"alive_{s}_{t1}"], tot, out=np.zeros(len(rows)), where=tot > 0)
        sig = {}
        for a in tt:
            f0, f1 = rows[f"fire_{o}_{a}_{t0}"].to_numpy(), rows[f"fire_{o}_{a}_{t1}"].to_numpy()
            on1 = np.where(
                f0 + f1 > 0, np.divide(f1, f0 + f1, out=np.zeros(len(rows)), where=f0 + f1 > 0), prop
            )
            sig[(a, t1)], sig[(a, t0)] = on1, 1 - on1
        return sig

    def _terms(self, R_o: dict, sig: dict, A_o: dict, wd: np.ndarray, o: str, t: str) -> dict:
        L = {}
        blue_rule = o == "blue" and self.withdraw == "rule"
        for a in self.L.types:
            # withdraw_fire (exploratory, docs/planning/25): the share of Blue's fire kept while withdrawing
            R = R_o[a] * (1 - wd * (1 - getattr(self, "withdraw_fire", 0.0))) if blue_rule else R_o[a]
            for t2 in self.L.types:
                L[f"R[{a}]*s[{t2}]"] = R * sig[(a, t2)]
            if o == "blue" and self.withdraw == "fitted":
                L[f"R[{a}]*s[{t}]*wd"] = R * sig[(a, t)] * wd
            L[f"A[{a}]"] = A_o[a]
        L["1"] = np.ones_like(wd)
        return L

    # ---- fitting ----
    def fit(self, rows: pd.DataFrame) -> Pipeline:
        """`rows`: from `prepare` (typed columns, rates, controls, both sides alive)."""
        tt = self.L.types
        check_rows(rows, self.L)
        self.share, self.share_r2, self.alloc_b, self.coef, self.p = {}, {}, {}, {}, {}
        n_opp = {s: 10 * sum(rows[f"alive_{OTHER[s]}_{t}"] for t in tt) for s in SIDES}
        for s in SIDES:
            o = OTHER[s]
            for a in tt:
                ok = rows[f"alive_{s}_{a}"] > 0
                r = rows[ok]
                X = (
                    (r["distance"] - r[f"front_{s}_{a}"].fillna(0)).to_numpy(),
                    r[f"depth_{s}"].to_numpy(),
                    r[f"depth_{o}"].to_numpy(),
                    r[f"width_{s}"].to_numpy(),
                    r[f"width_{o}"].to_numpy(),
                    n_opp[s][ok].to_numpy(),
                )
                y = (r[f"reach_{s}_{a}"] / r[f"alive_{s}_{a}"]).to_numpy()
                sub = np.random.default_rng(0).choice(len(y), min(200_000, len(y)), replace=False)
                f = lambda X, lam, beta, g, rho=self.rho[a]: derived_share(*X, lam, beta, g, rho=rho)  # noqa: E731
                p, _ = curve_fit(f, tuple(x[sub] for x in X), y[sub], p0=(1.0, 1.0, 1.0), maxfev=20000)
                self.share[(s, a)] = p
                self.share_r2[(s, a)] = r2(r[f"reach_{s}_{a}"], r[f"alive_{s}_{a}"] * f(X, *p))
        if len(tt) == 2:
            t0, t1 = tt
            for o in SIDES:
                s = OTHER[o]
                pi = (
                    rows[f"alive_{s}_{t1}"] / (rows[f"alive_{s}_{t0}"] + rows[f"alive_{s}_{t1}"])
                ).to_numpy()
                dfront = (rows[f"front_{s}_{t1}"].fillna(0) - rows[f"front_{s}_{t0}"].fillna(0)).to_numpy()
                for a in tt:
                    f0, f1 = rows[f"fire_{o}_{a}_{t0}"].to_numpy(), rows[f"fire_{o}_{a}_{t1}"].to_numpy()
                    ok = (f0 + f1 > 0) & (pi > 0) & (pi < 1)
                    b, _ = curve_fit(
                        lambda X, b0, b1, b2: _alloc(X[0], X[1], (b0, b1, b2)),
                        (pi[ok], dfront[ok]),
                        f1[ok] / (f0[ok] + f1[ok]),
                        p0=(0.0, 1.0, 0.0),
                        sigma=1 / np.sqrt(f0[ok] + f1[ok]),
                    )
                    self.alloc_b[(o, a)] = b
        wd = rows["withdraw"].to_numpy()
        for s in SIDES:
            o = OTHER[s]
            R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in tt}
            A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in tt}
            sig = self._sigma_measured(rows, o)
            for t in tt:
                L = self._terms(R_o, sig, A_o, wd, o, t)
                theta = np.column_stack(list(L.values()))
                c = self._health_stlsq(theta, rows[f"d_health_{s}_{t}"].to_numpy(), list(L))
                self.coef[(s, t)] = dict(zip(L, c, strict=True))
                h, al = rows[f"health_{s}_{t}"].to_numpy(), rows[f"alive_{s}_{t}"].to_numpy()
                ok = (h > 0.03) & (al > 0)
                x = (theta @ c) * al / np.maximum(h, 1e-9)
                X = np.column_stack([x, x * rows["focus"].to_numpy()]) if s == "red" else x[:, None]
                pc, *_ = np.linalg.lstsq(X[ok], rows[f"d_alive_{s}_{t}"].to_numpy()[ok], rcond=None)
                self.p[(s, t)] = (float(pc[0]), float(pc[1]) if s == "red" else 0.0)
        # distance: A's movement library with the team's measured reach
        T = team_rows(rows, self.L)
        self.P_team = {
            "gates": fit_gates(T, self.gate_scale, skip=("withdraw",) if self.movement_wd == "rule" else ()),
            "form": "derived",
        }
        rho_team = float(np.mean(list(self.rho.values())))
        for side in SIDES:
            a = T[T[f"alive_{side}"] > 0].sample(
                min(200_000, int((T[f"alive_{side}"] > 0).sum())), random_state=0
            )
            opp = OTHER[side]
            Xs = (
                a["distance"].to_numpy(),
                a[f"depth_{side}"].to_numpy(),
                a[f"depth_{opp}"].to_numpy(),
                a[f"width_{side}"].to_numpy(),
                a[f"width_{opp}"].to_numpy(),
                10 * a[f"alive_{opp}"].to_numpy(),
            )
            p, _ = curve_fit(
                lambda X, *q: derived_share(*X, *q, rho=rho_team),
                Xs,
                (a[f"reach_{side}"] / a[f"alive_{side}"]).to_numpy(),
                p0=(1.0, 1.0, 1.0),
                maxfev=20000,
            )
            self.P_team[f"share_{side}"] = np.append(p, rho_team)
        self._fit_extra_gates(T)
        if "withdraw" not in self.P_team["gates"]:  # no withdraw steps in the data: a gate that is never used
            self.P_team["gates"]["withdraw"] = np.array([0, 99, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], float)
        Xt = np.column_stack(
            [T[c].to_numpy() for c in ("health_blue", "health_red", "alive_blue", "alive_red", "distance")]
        )
        F = self._dist_library(
            Xt, T[self.ctrl_cols].to_numpy(), T[SHAPE].to_numpy(), T[["reach_blue", "reach_red"]].to_numpy()
        )
        self.dist_names = list(F)
        theta = np.column_stack([F[k] for k in self.dist_names])
        cols = [
            j for j in _family_cols(self.dist_names, MOVE_FAMILY) if self.dist_names[j] not in DIST_EXCLUDE
        ]
        if self.movement_wd == "rule":
            cols = [j for j in cols if self.dist_names[j] not in ("close*wd", "flee*wd")]
            cols += [self.dist_names.index("danger*wd"), self.dist_names.index("walk_r*wd")]
        cd = np.zeros(len(self.dist_names))
        cd[cols] = stlsq(theta[:, cols], T["d_distance"].to_numpy(), self.thr_distance)
        self.dist_coef = cd
        return self

    def _health_stlsq(self, theta: np.ndarray, y: np.ndarray, names: list[str]) -> np.ndarray:
        return stlsq(theta, y, self.thr_health)

    def _fit_extra_gates(self, T: pd.DataFrame) -> None:
        """Hook for variants with extra movements (models/kite.py)."""

    def _reach(self, n, X, S, F, u, sh, offset, B) -> dict:
        """Units in reach per (side, attacker type) at rollout step n: the closure's share plus the measured
        start offset decaying in 1 s. A hook for variants with reach as a state (models/reach_state.py)."""
        L = self.L
        return {k: X[:, L.idx("alive", *k)] * np.clip(sh[k] + offset[k] * np.exp(-n * DT), 0, 1) for k in sh}

    def _dist_library(self, X, U, S, reach) -> dict:
        """A's movement library; with movement_wd "rule", danger*wd uses the scenario's withdraw radius."""
        F = library(X, U, S, self.P_team, reach, True)
        if getattr(self, "movement_wd", "gates") == "rule":
            p = np.array(self.P_team["share_blue"], float)
            p[-1] = self.withdraw_radius  # derived share: rho is the last parameter
            F["danger*wd"] = team_shares(X, S, self.P_team | {"share_blue": p})[0] * U[:, 2]
        return F

    def coefficients(self) -> dict:
        out = {}
        for (s, t), c in self.coef.items():
            out |= {f"health_{s}_{t}: {k}": v for k, v in c.items() if v}
        out |= {f"distance: {n}": c for n, c in zip(self.dist_names, self.dist_coef, strict=True) if c}
        for (s, t), (p, pf) in self.p.items():
            out[f"alive_{s}_{t}: p"] = p
            if pf:
                out[f"alive_{s}_{t}: focus"] = pf
        return out

    def complexity(self) -> dict:
        n = sum(int(v != 0) for c in self.coef.values() for v in c.values())
        n += sum(1 + (pf != 0) for _, pf in self.p.values()) + int((self.dist_coef != 0).sum())
        return {"terms": n, "closure numbers": 3 * len(self.share) + 3 * len(self.alloc_b)}

    # ---- simulation ----
    def _shares(self, X, S, F) -> dict:
        tt, L = self.L.types, self.L
        d = X[:, -1]
        n_opp = {s: 10 * sum(X[:, L.idx("alive", OTHER[s], t)] for t in tt) for s in SIDES}
        dep, wid = {"blue": S[:, 0], "red": S[:, 1]}, {"blue": S[:, 2], "red": S[:, 3]}
        return {
            (s, a): derived_share(
                d - F[:, L.pos(s, a)],
                dep[s],
                dep[OTHER[s]],
                wid[s],
                wid[OTHER[s]],
                n_opp[s],
                *self.share[(s, a)],
                rho=self.rho[a],
            )
            for s in SIDES
            for a in tt
        }

    def rollout(self, B: dict) -> np.ndarray:
        """Typed trajectories (N, steps+1, 4n+1); `self.L.team_view` gives the team view."""
        tt, L = self.L.types, self.L
        X, S, F, U = B["X0"].astype(float).copy(), B["shape0"].astype(float).copy(), B["front0"], B["U"]
        extra = getattr(self, "intent_extra", None)  # exploratory variant, docs/planning/06_intent_variant.md
        spread = None  # exploratory spread state per side and type, in the reach / front column order
        if getattr(self, "spread_model", None) is not None:
            spread = np.clip(B["spread0"].astype(float).copy(), 0.05, 1)
        comp = getattr(self, "compart_model", None)  # exploratory, docs/planning/13_compartment_plan.md
        if comp is not None:  # wounded units / 10 and their health (ten full units), reach / front order
            Wc, HWc, h0c = (B[k].astype(float).copy() for k in ("W0", "HW0", "h00"))
        sh0 = self._shares(X, S, F)
        offset = {}
        for (s, a), v in sh0.items():
            A0 = X[:, L.idx("alive", s, a)]
            meas = np.divide(B["reach0"][:, L.pos(s, a)], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
            offset[(s, a)] = np.where(A0 > 1e-9, meas - v, 0.0)
        out = [X.copy()]
        for n in range(U.shape[1]):
            u = U[:, n]
            self._u = u  # the step's controls, for variants whose terms use more of them (models/kite.py)
            wd, focus = u[:, 2], u[:, 3]
            if "shape_traj" in B:  # diagnostic (docs/planning/24): the true formation shape at every step
                S = np.nan_to_num(B["shape_traj"][:, min(n, B["shape_traj"].shape[1] - 1)])
            if comp is not None:  # wounded shares for equations that use them (models/discovered.py)
                self._ctx = compart_context(L, X, Wc, u)
            sh = self._shares(X, S, F)
            R = self._reach(n, X, S, F, u, sh, offset, B)
            A = {(s, t): X[:, L.idx("alive", s, t)] for s in SIDES for t in tt}
            newX = X.copy()
            new_spread = None if spread is None else spread.copy()
            for s in SIDES:
                o = OTHER[s]
                if len(tt) == 1:
                    sig = {(tt[0], tt[0]): np.ones(len(X))}
                elif len(tt) > 2:
                    al = np.column_stack([A[(s, t)] for t in tt])
                    tot = al.sum(1, keepdims=True)
                    al = np.divide(al, tot, out=np.zeros_like(al), where=tot > 0)
                    sig = {(a, t): al[:, j] for a in tt for j, t in enumerate(tt)}
                else:
                    t0, t1 = tt
                    tot = A[(s, t0)] + A[(s, t1)]
                    pi = np.divide(A[(s, t1)], tot, out=np.zeros_like(tot), where=tot > 0)
                    dfront = F[:, L.pos(s, t1)] - F[:, L.pos(s, t0)]
                    sig = {}
                    for a in tt:
                        on1 = _alloc(pi, dfront, self.alloc_b[(o, a)]) if self.alloc == "closure" else pi
                        sig[(a, t1)], sig[(a, t0)] = on1, 1 - on1
                R_o = {a: R[(o, a)] for a in tt}
                if o == "blue" and getattr(self, "focus_cost", 0.0):
                    # exploratory variant (docs/planning/04_focus_variant.md): under focus, fewer Blue units
                    # with an enemy in reach have their own target in reach; `focus_norm` keeps the train mean
                    scale = (1 - self.focus_cost * focus) / getattr(self, "focus_norm", 1.0)
                    R_o = {a: v * scale for a, v in R_o.items()}
                A_o = {a: A[(o, a)] for a in tt}
                for t in tt:
                    terms = self._terms(R_o, sig, A_o, wd, o, t)
                    rate = sum(c * terms[k] for k, c in self.coef[(s, t)].items() if c)
                    rate = np.where(A[(s, t)] > 0, np.minimum(rate, 0), 0.0)
                    h0 = X[:, L.idx("health", s, t)]
                    h1 = np.maximum(h0 + DT * rate, 0)
                    ratio = np.divide(h1, h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                    p, pf = self.p[(s, t)]
                    expo = p + pf * focus
                    if spread is not None:  # exploratory spread state (docs/planning/09_spread_state_plan.md)
                        j = L.pos(s, t)
                        sv, d = spread[:, j], X[:, -1]
                        names, c = self.spread_model["alive"][(s, t)]
                        Fa = spread_alive_features(sv, u, d)
                        expo = np.column_stack([Fa[k] for k in names]) @ c
                        rr = np.divide(h0 - h1, DT * h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                        names, c = self.spread_model["rate"][(s, t)]
                        Fr = spread_rate_features(sv, rr, u, d)
                        new_spread[:, j] = np.clip(
                            sv + DT * np.column_stack([Fr[k] for k in names]) @ c, 0.05, 1
                        )
                    newX[:, L.idx("health", s, t)] = h1
                    newX[:, L.idx("alive", s, t)] = A[(s, t)] * ratio**expo
                    if comp is not None:
                        j = L.pos(s, t)
                        a_new, w_new, hw_new = compart_step(
                            comp, (s, t), A[(s, t)], Wc[:, j], h0, h1, HWc[:, j], h0c[:, j], R[(s, t)], u
                        )
                        newX[:, L.idx("alive", s, t)] = a_new
                        Wc[:, j], HWc[:, j] = w_new, hw_new
            TX = L.team_view(X)
            reach_team = np.column_stack([sum(R[("blue", a)] for a in tt), sum(R[("red", a)] for a in tt)])
            Fd = self._dist_library(TX, u, S, reach_team)
            theta = np.column_stack([Fd[k] for k in self.dist_names])
            dd = theta @ self.dist_coef
            if extra:
                G = intent_features(Fd, TX, u)
                if extra.get("dist"):
                    names, c = extra["dist"]
                    dd = dd + np.column_stack([G[k] for k in names]) @ c
                if extra.get("shape"):
                    S = shape_step(extra["shape"], TX[:, -1], S, u)
            newX[:, -1] = np.maximum(X[:, -1] + DT * dd, 0)
            X = newX
            if spread is not None:
                spread = new_spread
            out.append(X.copy())
        self._ctx = None
        return np.stack(out, axis=1)
