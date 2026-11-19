"""A world model with one group state per unit type (branch mixed-marauders, step 3).

State, per side s and type t: health H_st (ten full units of t = 1) and units alive A_st (/10); plus
the centroid distance d. Formation shape (depth, width per side; how far forward each type sits) is
held at its start value, as in A.

    H_st'  = sum_a [ k_{a,t} R_oa sigma_{oa,t}  (+ k^wd_{a,t} R_oa sigma_{oa,t} wd  for Blue attackers) ]
    A_st   = A_st (H_st(new) / H_st) ** p_st            (A*-2k's alive law, + focus term for Red)
    d'     = the team model's distance equation (A*-2k's gates), with the team's reach from the closures

    R_sa   = A_sa * derived_share(d - front_sa, depth, width, n_enemy; rho_a)   reach closure per type,
             with the measured start offset decaying (tau = 1 s), as in A
    sigma  = allocation closure: share of attacker type a's fire on the enemy's marauders,
             logit sigma = b0 + b1 logit(pi) + b2 (front_marauder - front_marine of the target side),
             pi = the marauders' share of the target side's alive units. Fitted per side and attacker type.

Health coefficients by STLSQ on the measured reach x measured allocation share (checkpoint 2, form (ii)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit, logit

from sindy_wm.models.intent_model import CTRL, DT, SHAPE, derived_share, library, r2, stlsq
from sindy_wm.states.typed import pair_reach, unit_stats

TYPES = ("marine", "marauder")
SIDES = ("blue", "red")
OTHER = {"blue": "red", "red": "blue"}
HP = {t: unit_stats(t.upper())["hp"] for t in TYPES}
RHO = {a: float(np.mean([pair_reach(a.upper(), t.upper()) for t in TYPES])) for a in TYPES}
H_COLS = [f"health_{s}_{t}" for s in SIDES for t in TYPES]
A_COLS = [f"alive_{s}_{t}" for s in SIDES for t in TYPES]
R_COLS = [f"reach_{s}_{t}" for s in SIDES for t in TYPES]
FRONT = [f"front_{s}_{t}" for s in SIDES for t in TYPES]
X_COLS = H_COLS + A_COLS + ["distance"]  # the typed state, 9 columns
FIRE = {(s, a, t): f"fire_{s}_{a}_{t}" for s in SIDES for a in TYPES for t in TYPES}


def idx(kind: str, side: str, t: str) -> int:
    """Column of (kind, side, type) in the typed state X."""
    base = {"health": 0, "alive": 4}[kind]
    return base + 2 * SIDES.index(side) + TYPES.index(t)


def team_view(X: np.ndarray) -> np.ndarray:
    """Typed state (..., 9) -> team state (..., 5): health in ten marines' hp, alive / 10, distance."""
    out = []
    for s in SIDES:
        out.append(sum(X[..., idx("health", s, t)] * HP[t] / HP["marine"] for t in TYPES))
    for s in SIDES:
        out.append(sum(X[..., idx("alive", s, t)] for t in TYPES))
    out.append(X[..., 8])
    return np.stack(out, axis=-1)


def team_table(states: pd.DataFrame) -> pd.DataFrame:
    """A typed state table as a team table (the columns `IntentModel` / the harness expect)."""
    T = states.copy()
    for s in SIDES:
        T[f"health_{s}"] = sum(T[f"health_{s}_{t}"] * HP[t] / HP["marine"] for t in TYPES)
        T[f"alive_{s}"] = sum(T[f"alive_{s}_{t}"] for t in TYPES)
        T[f"reach_{s}"] = sum(T[f"reach_{s}_{t}"] for t in TYPES)
    return T


# --------------------------------------------------------------------------------------- batches
def typed_batch(
    states: pd.DataFrame, horizon: int = 64, every: int = 8, history: int = 8, min_future: int = 8
):
    """Start points with typed truth: X0 (N, 9), shape0, front0, reach0 (N, 4), U, Y (N, horizon+1, 9),
    plus the team view (TX0, TY, Treach0) for the team model and the harness metrics."""
    keys = ["X0", "U", "Y", "shape0", "front0", "reach0", "ep", "k0"]
    out = {k: [] for k in keys}
    for ep, g in states.groupby(level=0, sort=False):
        S, C = g[X_COLS].to_numpy(float), g[CTRL].to_numpy(float)
        Sh, Fr, R = g[SHAPE].to_numpy(float), g[FRONT].to_numpy(float), g[R_COLS].to_numpy(float)
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
    B["TX0"], B["TY"] = team_view(B["X0"]), team_view(B["Y"])
    B["Treach0"] = np.column_stack(
        [B["reach0"][:, 0] + B["reach0"][:, 1], B["reach0"][:, 2] + B["reach0"][:, 3]]
    )
    return B


def team_horizon_batch(B: dict) -> dict:
    """The harness's horizon-batch layout (team state) for the same start points."""
    return {
        "X0": B["TX0"],
        "Y": B["TY"],
        "U": B["U"],
        "shape0": B["shape0"],
        "reach0": B["Treach0"],
        "ep": B["ep"],
        "group": B["ep"] % 10_000,
    }


def team_battle_batch(states: pd.DataFrame, steps: int = 720) -> tuple[dict, dict]:
    """One start per battle at step 0 to its end: (typed batch, harness battle-batch layout)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        B = typed_batch(states, horizon=steps, every=10**9, history=0, min_future=1)
    won = states.groupby(level=0)["battle_won"].first()
    T = team_horizon_batch(B)
    T["won"] = won.loc[B["ep"]].to_numpy().astype(bool)
    T["end"] = (~np.isnan(B["Y"][:, :, 0])).sum(1) - 1
    return B, T


# ------------------------------------------------------------------------------------- the model
def _alloc(pi, dfront, b):
    """Share of fire on marauders: logistic in logit(pi) and the front offset (0 / 1 if a type is gone)."""
    z = b[0] + b[1] * logit(np.clip(pi, 1e-4, 1 - 1e-4)) + b[2] * dfront
    return np.where(pi <= 0, 0.0, np.where(pi >= 1, 1.0, expit(z)))


def _pi(A_t: dict) -> np.ndarray:
    tot = A_t["marine"] + A_t["marauder"]
    return np.divide(A_t["marauder"], tot, out=np.zeros_like(tot), where=tot > 0)


class TypedA:
    """The per-type model (see the module docstring). `team` is a fitted team model (A*-2k on team
    totals) whose distance equation and movement gates are reused."""

    def __init__(self, team, thr: float = 0.15, alloc: str = "closure", max_reach: float | None = None):
        """max_reach: the cap the state tables were built with (e.g. 6.0); sets each type's closure rho."""
        self.team, self.thr, self.alloc, self.max_reach = team, thr, alloc, max_reach
        cap = max_reach or np.inf
        self.rho = {
            a: float(np.mean([min(pair_reach(a.upper(), t.upper()), cap) for t in TYPES])) for a in TYPES
        }

    # ---- terms ----
    @staticmethod
    def _terms(R_o: dict, sig: dict, A_o: dict, wd: np.ndarray, o: str, t: str) -> dict:
        t2 = [x for x in TYPES if x != t][0]
        L = {}
        for a in TYPES:
            L[f"R[{a}]*s[{t}]"] = R_o[a] * sig[(a, t)]
            L[f"R[{a}]*s[{t2}]"] = R_o[a] * sig[(a, t2)]
            if o == "blue":
                L[f"R[{a}]*s[{t}]*wd"] = R_o[a] * sig[(a, t)] * wd
            L[f"A[{a}]"] = A_o[a]
        L["1"] = np.ones_like(wd)
        return L

    # ---- fitting ----
    def fit(self, rows: pd.DataFrame) -> TypedA:
        """`rows`: typed rows with d_health_* / d_alive_* rates, CTRL columns, both sides alive."""
        self.share, self.share_r2, self.alloc_b, self.coef, self.p = {}, {}, {}, {}, {}
        n_opp = {s: 10 * sum(rows[f"alive_{OTHER[s]}_{t}"] for t in TYPES) for s in SIDES}
        for s in SIDES:
            o = OTHER[s]
            for a in TYPES:
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
        # allocation closure, per attacking side and attacker type
        for o in SIDES:
            s = OTHER[o]  # target side
            pi = (
                rows[f"alive_{s}_marauder"] / (rows[f"alive_{s}_marine"] + rows[f"alive_{s}_marauder"])
            ).to_numpy()
            dfront = (rows[f"front_{s}_marauder"].fillna(0) - rows[f"front_{s}_marine"].fillna(0)).to_numpy()
            for a in TYPES:
                fm, fM = rows[FIRE[(o, a, "marine")]].to_numpy(), rows[FIRE[(o, a, "marauder")]].to_numpy()
                ok = (fm + fM > 0) & (pi > 0) & (pi < 1)
                y = fM[ok] / (fm[ok] + fM[ok])
                b, _ = curve_fit(
                    lambda X, b0, b1, b2: _alloc(X[0], X[1], (b0, b1, b2)),
                    (pi[ok], dfront[ok]),
                    y,
                    p0=(0.0, 1.0, 0.0),
                    sigma=1 / np.sqrt(fm[ok] + fM[ok]),
                )
                self.alloc_b[(o, a)] = b
        # health: STLSQ on measured reach x measured allocation share (checkpoint 2, (ii))
        wd = rows["withdraw"].to_numpy()
        for s in SIDES:
            o = OTHER[s]
            R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in TYPES}
            A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in TYPES}
            sig = self._measured_sigma(rows, o)
            for t in TYPES:
                L = self._terms(R_o, sig, A_o, wd, o, t)
                self.names = list(L)
                self.coef[(s, t)] = dict(
                    zip(
                        L,
                        stlsq(
                            np.column_stack(list(L.values())), rows[f"d_health_{s}_{t}"].to_numpy(), self.thr
                        ),
                        strict=True,
                    )
                )
                # alive law: A' = p Hhat' A / H (+ focus for Red)
                hhat = sum(self.coef[(s, t)][k] * v for k, v in L.items())
                h, al = rows[f"health_{s}_{t}"].to_numpy(), rows[f"alive_{s}_{t}"].to_numpy()
                ok = (h > 0.01) & (al > 0)
                x = hhat * al / np.maximum(h, 1e-9)
                X = np.column_stack([x, x * rows["focus"].to_numpy()]) if s == "red" else x[:, None]
                c, *_ = np.linalg.lstsq(X[ok], rows[f"d_alive_{s}_{t}"].to_numpy()[ok] / 1.0, rcond=None)
                self.p[(s, t)] = (float(c[0]), float(c[1]) if s == "red" else 0.0)
        return self

    @staticmethod
    def _measured_sigma(rows: pd.DataFrame, o: str) -> dict:
        s = OTHER[o]
        tot = rows[f"alive_{s}_marine"] + rows[f"alive_{s}_marauder"]
        prop = np.divide(rows[f"alive_{s}_marauder"], tot, out=np.zeros(len(rows)), where=tot > 0)
        sig = {}
        for a in TYPES:
            fm, fM = rows[FIRE[(o, a, "marine")]].to_numpy(), rows[FIRE[(o, a, "marauder")]].to_numpy()
            on_M = np.where(
                fm + fM > 0, np.divide(fM, fm + fM, out=np.zeros(len(rows)), where=fm + fM > 0), prop
            )
            sig[(a, "marauder")], sig[(a, "marine")] = on_M, 1 - on_M
        return sig

    def coefficients(self) -> dict:
        out = {}
        for (s, t), c in self.coef.items():
            out |= {f"health_{s}_{t}: {k}": v for k, v in c.items() if v}
        for (s, t), (p, pf) in self.p.items():
            out[f"alive_{s}_{t}: p"] = p
            if pf:
                out[f"alive_{s}_{t}: focus"] = pf
        return out

    def complexity(self) -> dict:
        n = sum(int(v != 0) for c in self.coef.values() for v in c.values()) + sum(
            1 + (pf != 0) for _, pf in self.p.values()
        )
        closure = 3 * len(self.share) + 3 * len(self.alloc_b)
        return {"terms": n + self.team.m.coef["distance"].astype(bool).sum(), "closure numbers": closure}

    # ---- simulation ----
    def _shares(self, X, S, F):
        """Closure share in reach per (side, type), (N,) arrays."""
        d = X[:, 8]
        n_opp = {s: 10 * sum(X[:, idx("alive", OTHER[s], t)] for t in TYPES) for s in SIDES}
        dep = {"blue": S[:, 0], "red": S[:, 1]}
        wid = {"blue": S[:, 2], "red": S[:, 3]}
        out = {}
        for s in SIDES:
            for a in TYPES:
                fr = F[:, 2 * SIDES.index(s) + TYPES.index(a)]
                out[(s, a)] = derived_share(
                    d - fr,
                    dep[s],
                    dep[OTHER[s]],
                    wid[s],
                    wid[OTHER[s]],
                    n_opp[s],
                    *self.share[(s, a)],
                    rho=self.rho[a],
                )
        return out

    def rollout(self, B: dict, steps: int | None = None) -> np.ndarray:
        X, S, F, U = B["X0"].astype(float).copy(), B["shape0"], B["front0"], B["U"]
        steps = U.shape[1] if steps is None else steps
        sh0 = self._shares(X, S, F)
        offset = {}
        for s in SIDES:
            for a in TYPES:
                A0 = X[:, idx("alive", s, a)]
                j = 2 * SIDES.index(s) + TYPES.index(a)
                meas = np.divide(B["reach0"][:, j], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
                offset[(s, a)] = np.where(A0 > 1e-9, meas - sh0[(s, a)], 0.0)
        out = [X.copy()]
        Pteam = self.team.m.P
        cd = self.team.m.coef["distance"]
        for n in range(steps):
            u = U[:, n]
            wd, focus = u[:, 2], u[:, 3]
            sh = self._shares(X, S, F)
            R = {k: X[:, idx("alive", *k)] * np.clip(sh[k] + offset[k] * np.exp(-n * DT), 0, 1) for k in sh}
            A = {(s, t): X[:, idx("alive", s, t)] for s in SIDES for t in TYPES}
            newX = X.copy()
            for s in SIDES:
                o = OTHER[s]
                A_s = {t: A[(s, t)] for t in TYPES}
                pi = _pi(A_s)
                dfront = F[:, 2 * SIDES.index(s) + 1] - F[:, 2 * SIDES.index(s)]
                sig = {}
                for a in TYPES:
                    on_M = _alloc(pi, dfront, self.alloc_b[(o, a)]) if self.alloc == "closure" else pi
                    sig[(a, "marauder")], sig[(a, "marine")] = on_M, 1 - on_M
                R_o = {a: R[(o, a)] for a in TYPES}
                A_o = {a: A[(o, a)] for a in TYPES}
                for t in TYPES:
                    L = self._terms(R_o, sig, A_o, wd, o, t)
                    rate = sum(self.coef[(s, t)][k] * v for k, v in L.items() if self.coef[(s, t)][k])
                    rate = np.where(A_s[t] > 0, np.minimum(rate, 0), 0.0)
                    h0 = X[:, idx("health", s, t)]
                    h1 = np.maximum(h0 + DT * rate, 0)
                    ratio = np.divide(h1, h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                    p, pf = self.p[(s, t)]
                    newX[:, idx("health", s, t)] = h1
                    newX[:, idx("alive", s, t)] = A_s[t] * ratio ** (p + pf * focus)
            # distance: the team model's equation with the team's reach
            TX = team_view(X)
            reach_team = np.column_stack(
                [
                    R[("blue", "marine")] + R[("blue", "marauder")],
                    R[("red", "marine")] + R[("red", "marauder")],
                ]
            )
            Fd = library(TX, u, S, Pteam, reach_team, True)
            theta = np.column_stack([Fd[k] for k in self.team.m.names])
            newX[:, 8] = np.maximum(X[:, 8] + DT * theta @ cd, 0)
            X = newX
            out.append(X.copy())
        return np.stack(out, axis=1)
