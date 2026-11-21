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
from scipy.special import expit, logit

from sindy_wm.models.intent_model import (
    ALIVE_CONTROL_TERMS,
    CTRL,
    DT,
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
    def batch(self, states: pd.DataFrame, horizon=64, every=8, history=8, min_future=8) -> dict:
        """Start points (as `make_batch`): X0, U, Y (typed), shape0, front0, reach0, plus the team view."""
        keys = ["X0", "U", "Y", "shape0", "front0", "reach0", "ep", "k0"]
        out = {k: [] for k in keys}
        for ep, g in states.groupby(level=0, sort=False):
            S, C = g[self.x_cols].to_numpy(float), g[CTRL].to_numpy(float)
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
def prepare(states: pd.DataFrame, layout: Layout, window: int = 9) -> pd.DataFrame:
    """Controls and SG rates (per battle) for the typed columns and distance; steps with both sides alive."""
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


class Pipeline:
    """The scenario-independent model (see the module docstring)."""

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
        if len(types) > 2:
            raise NotImplementedError("target choice is defined for one or two unit types")
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
        s, t0, t1 = OTHER[o], tt[0], tt[1]
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
            R = R_o[a] * (1 - wd) if blue_rule else R_o[a]
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
                c = stlsq(theta, rows[f"d_health_{s}_{t}"].to_numpy(), self.thr_health)
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
        if "withdraw" not in self.P_team["gates"]:  # no withdraw steps in the data: a gate that is never used
            self.P_team["gates"]["withdraw"] = np.array([0, 99, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], float)
        Xt = np.column_stack(
            [T[c].to_numpy() for c in ("health_blue", "health_red", "alive_blue", "alive_red", "distance")]
        )
        F = self._dist_library(
            Xt, T[CTRL].to_numpy(), T[SHAPE].to_numpy(), T[["reach_blue", "reach_red"]].to_numpy()
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
        sh0 = self._shares(X, S, F)
        offset = {}
        for (s, a), v in sh0.items():
            A0 = X[:, L.idx("alive", s, a)]
            meas = np.divide(B["reach0"][:, L.pos(s, a)], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
            offset[(s, a)] = np.where(A0 > 1e-9, meas - v, 0.0)
        out = [X.copy()]
        for n in range(U.shape[1]):
            u = U[:, n]
            wd, focus = u[:, 2], u[:, 3]
            sh = self._shares(X, S, F)
            R = {k: X[:, L.idx("alive", *k)] * np.clip(sh[k] + offset[k] * np.exp(-n * DT), 0, 1) for k in sh}
            A = {(s, t): X[:, L.idx("alive", s, t)] for s in SIDES for t in tt}
            newX = X.copy()
            for s in SIDES:
                o = OTHER[s]
                if len(tt) == 1:
                    sig = {(tt[0], tt[0]): np.ones(len(X))}
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
                    newX[:, L.idx("health", s, t)] = h1
                    newX[:, L.idx("alive", s, t)] = A[(s, t)] * ratio ** (p + pf * focus)
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
            out.append(X.copy())
        return np.stack(out, axis=1)
