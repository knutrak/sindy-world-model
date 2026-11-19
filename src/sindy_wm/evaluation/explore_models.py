"""Exploration models for the final exploration (axes E, G, J, D/I). All follow the harness interface.

EnsembleA         E-SINDy on A's library: bootstrap the training battles, refit STLSQ each time
                  (closures fixed), keep terms with inclusion probability >= `keep`, use median
                  coefficients. Ensemble members' rollouts give a win probability.
AliveVariantA     A with another death law: no control terms in alive (fixes G3b's moving
                  term set?), alive in proportion to health (p = 1), or p fitted on rollouts.
MLPWorld          black-box reference on the same state, shape and intents: an MLP for the
                  one-step change, trained on 1-16-step rollout loss (torch).
SizeScaledA       A with each health / distance coefficient linear in the battle's starting size
                  (ParameterizedLibrary idea: c(n) = c0 + c1 * (n - 20) / 10), fitted by STLSQ on
                  the expanded library.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from sindy_wm.evaluation import harness as H
from sindy_wm.evaluation.harness_models import (
    ALIVE,
    ModelA,
    _closure_numbers,
    _LabModel,
    closures_for,
    intent_coefficients,
    lab,
)
from sindy_wm.models.intent_model import CTRL, DT, SHAPE, STATE, IntentModel, stlsq


# ---------------------------------------------------------------------------------- resampling
def resample_battles(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Battles drawn with replacement; each draw gets its own episode id (as evaluate_final)."""
    codes, uniq = pd.factorize(rows.index.get_level_values(0))
    positions = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    pick = np.random.default_rng(seed).integers(0, len(uniq), len(uniq))
    idx = np.concatenate([positions[g] for g in pick])
    new_ep = np.repeat(np.arange(len(pick)), [len(positions[g]) for g in pick])
    out = rows.iloc[idx].copy()
    out.index = pd.MultiIndex.from_arrays(
        [new_ep, rows.index.get_level_values(1)[idx]], names=rows.index.names
    )
    return out


# ------------------------------------------------------------------------------------ alive laws
def refit_alive(m: IntentModel, rows: pd.DataFrame, controls: bool, thr: float = 0.02) -> None:
    """Refit the alive exponents of a fitted model, with or without intent terms (as IntentModel.fit)."""
    U = rows[CTRL].to_numpy()
    for side in ("blue", "red"):
        ok = ((rows[f"health_{side}"] > 0.03) & (rows[f"alive_{side}"] > 0)).to_numpy()
        g = (rows[f"d_health_{side}"] / rows[f"health_{side}"]).to_numpy()[ok]
        y = (rows[f"d_alive_{side}"] / rows[f"alive_{side}"]).to_numpy()[ok]
        feats = {"1": g}
        if controls:
            feats |= {"hold": g * U[ok, 1], "wd": g * U[ok, 2], "focus": g * U[ok, 3]}
        c = stlsq(np.column_stack(list(feats.values())), y, thr)
        m.p[side] = {"1": 0.0, "hold": 0.0, "wd": 0.0, "focus": 0.0} | dict(zip(feats, c, strict=True))


class AliveVariantA(ModelA):
    """law: "no-control" (p per side only), "proportional" (p = 1), "rollout" (see AliveRolloutA)."""

    def __init__(self, law: str = "no-control", **kw):
        super().__init__(**kw)
        self.law = law

    def fit(self, data):
        super().fit(data)
        if self.law == "no-control":
            refit_alive(self.m, data.rates, controls=False)
        elif self.law == "reach":
            self._fit_reach_law(data.rates)
        elif self.law == "proportional":
            for side in ("blue", "red"):
                self.m.p[side] = {"1": 1.0, "hold": 0.0, "wd": 0.0, "focus": 0.0}
        else:
            raise ValueError(self.law)
        return self

    def _fit_reach_law(self, rows: pd.DataFrame) -> None:
        """A's law, (ln A)' = p (ln H)', fitted with the regressor the rollout actually uses.

        In a rollout H' = k R_enemy, so A' = p k R_enemy A / H. Regressing the alive rate on
        R_enemy A / H (one SINDy library term, plus x focus for Red) avoids dividing by the noisy
        smoothed health rate (errors in variables, which pulls p towards 0). p = c / k.
        """
        names = self.m.names
        k = {
            "blue": self.m.coef["health_blue"][names.index("Rr")],
            "red": self.m.coef["health_red"][names.index("Rb")],
        }
        for side, enemy in (("blue", "red"), ("red", "blue")):
            ok = ((rows[f"health_{side}"] > 0.03) & (rows[f"alive_{side}"] > 0)).to_numpy()
            x = (rows[f"reach_{enemy}"] * rows[f"alive_{side}"] / rows[f"health_{side}"]).to_numpy()[ok]
            y = rows[f"d_alive_{side}"].to_numpy()[ok]
            T = x[:, None] if side == "blue" else np.column_stack([x, x * rows["focus"].to_numpy()[ok]])
            c, *_ = np.linalg.lstsq(T, y, rcond=None)
            p = c / k[side]
            focus = float(p[1]) if side == "red" else 0.0
            self.m.p[side] = {"1": float(p[0]), "hold": 0.0, "wd": 0.0, "focus": focus}


class AliveRolloutA(_LabModel):
    """A with only the alive exponents (p_b, p_r, p_r_focus) refitted on 1-8 s rollout error."""

    def __init__(self, n_start: int = 20_000, focus: bool = True):
        self.n_start, self.focus = n_start, focus

    def fit(self, data):
        a = ModelA().fit(data)
        self.n_closure = _closure_numbers(a.P)
        names = ALIVE if self.focus else ["p_b", "p_r"]
        p0 = a.params() | ({} if self.focus else {"p_r_focus": 0.0})
        self.p, _ = lab().refit(names, p0, data.subsample(self.n_start))
        return self


# ------------------------------------------------------------------------------------- ensemble
class EnsembleA(ModelA):
    """E-SINDy: bagging by battle around A's STLSQ fit. Closures are fitted once and held fixed."""

    def __init__(self, n_models: int = 30, keep: float = 0.5, alive_controls: bool = True, **kw):
        super().__init__(**kw)
        self.n_models, self.keep, self.alive_controls = n_models, keep, alive_controls

    def _fit_one(self, rows):
        m = IntentModel(self.P, exclude=self.exclude).fit(
            rows, thr_health=self.thr_health, reach_true=True, thr_shape=None
        )
        if not self.alive_controls:
            refit_alive(m, rows, controls=False)
        return m

    def fit(self, data):
        self.P = self.P or closures_for(data)
        base = self._fit_one(data.rates)
        self.members = [self._fit_one(resample_battles(data.rates, 10_000 + b)) for b in range(self.n_models)]
        m = copy.deepcopy(base)
        self.inclusion = {}
        for eq in m.coef:
            stack = np.stack([mm.coef[eq] for mm in self.members])
            incl = (stack != 0).mean(0)
            med = np.median(np.where(stack != 0, stack, np.nan), axis=0)
            m.coef[eq] = np.where(incl >= self.keep, np.nan_to_num(med), 0.0)
            for n, v in zip(m.names, incl, strict=True):
                if v > 0:
                    self.inclusion[f"{eq}: {n}"] = float(v)
        for side in ("blue", "red"):
            for k in m.p[side]:
                vals = np.array([mm.p[side].get(k, 0.0) for mm in self.members])
                incl = float((vals != 0).mean())
                if incl > 0:
                    self.inclusion[f"alive_{side}: {k}"] = incl
                m.p[side][k] = float(np.median(vals[vals != 0])) if incl >= self.keep else 0.0
        self.m = m
        return self

    def win_probability(self, B, stalkers=False):
        from sindy_wm.models.switch_library import scaled

        wins = []
        for mm in self.members:
            m = scaled(mm, "marine", "stalker") if stalkers else mm
            Y = m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])
            wins.append(H.rollout_end(Y)[1])
        return np.mean(wins, axis=0)

    def coefficients(self):
        return intent_coefficients(self.m) | {f"incl {k}": v for k, v in self.inclusion.items()}


# --------------------------------------------------------------------------- size-scaled library
class SizeScaledA(ModelA):
    """A's library with every health and distance candidate also multiplied by z = (n0 - 20) / 10.

    n0 is the battle's starting number of units (both sides), a constant per battle, so this is a
    one-parameter ParameterizedLibrary. STLSQ decides whether any coefficient depends on size.
    """

    def fit(self, data):
        self.P = self.P or closures_for(data)
        rows = data.rates
        self.m = IntentModel(self.P, exclude=self.exclude).fit(
            rows, thr_health=self.thr_health, reach_true=True, thr_shape=None
        )
        from sindy_wm.models.intent_model import FIRE_FAMILY, MOVE_FAMILY, REACH_COLS, _family_cols, library

        X, U, S = rows[STATE].to_numpy(), rows[CTRL].to_numpy(), rows[SHAPE].to_numpy()
        F = library(X, U, S, self.P, rows[REACH_COLS].to_numpy(), True)
        names = list(F)
        theta = np.column_stack([F[n] for n in names])
        z = ((rows["n_blue"] + rows["n_red"]).to_numpy() - 20) / 10
        self.size_coef = {}
        for eq, family in (
            ("health_blue", FIRE_FAMILY),
            ("health_red", FIRE_FAMILY),
            ("distance", MOVE_FAMILY),
        ):
            cols = [j for j in _family_cols(names, family) if names[j] not in self.exclude]
            T = np.column_stack([theta[:, cols], theta[:, cols] * z[:, None]])
            thr = self.thr_health if eq != "distance" else 0.05
            c = stlsq(T, rows[f"d_{eq}"].to_numpy(), thr)
            base, slope = np.zeros(len(names)), np.zeros(len(names))
            base[cols], slope[cols] = c[: len(cols)], c[len(cols) :]
            self.m.coef[eq], self.size_coef[eq] = base, slope
        return self

    def rollout(self, B, stalkers=False):
        """Per battle: coefficients c0 + c1 z with z from the start state's battle size."""
        from sindy_wm.models.switch_library import scaled

        z = (B["n0"] - 20) / 10
        out = np.empty((len(B["X0"]), B["U"].shape[1] + 1, 5))
        for zz in np.unique(z):
            idx = np.flatnonzero(z == zz)
            m = copy.deepcopy(self.m)
            for eq in self.size_coef:
                m.coef[eq] = self.m.coef[eq] + zz * self.size_coef[eq]
            m = scaled(m, "marine", "stalker") if stalkers else m
            sub = {k: B[k][idx] for k in ("X0", "shape0", "U", "reach0")}
            out[idx] = m.simulate(sub["X0"], sub["shape0"], sub["U"], reach0=sub["reach0"])
        return out

    def complexity(self):
        n = self.m.n_terms() + sum(int(np.count_nonzero(v)) for v in self.size_coef.values())
        return {"terms": n, "numbers": n + _closure_numbers(self.P)}

    def coefficients(self):
        out = intent_coefficients(self.m)
        for eq, c in self.size_coef.items():
            for n, v in zip(self.m.names, c, strict=True):
                if v:
                    out[f"{eq}: {n} * z"] = v
        return out


# ------------------------------------------------------------------------------------ black box
FEATS = STATE + SHAPE  # + CTRL as inputs


class MLPWorld:
    """Black-box world model on the same inputs: x_{t+1} = x_t + DT * f(x_t, shape, u_t, reach share_0).

    Trained with torch on rollout loss over 16 steps (2 s) from random start points, so it is
    compared fairly with the rollout-fitted B. Outputs are clipped like the SINDy models (health,
    alive, distance >= 0; alive and health never above their start).
    """

    can_scale = False

    def __init__(
        self, hidden: int = 64, steps: int = 16, epochs: int = 30, n_start: int = 40_000, seed: int = 0
    ):
        self.hidden, self.steps, self.epochs, self.n_start, self.seed = hidden, steps, epochs, n_start, seed

    def _inputs(self, X, S, u, r0):
        import torch

        return torch.cat([X, S, u, r0], -1)

    def fit(self, data):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        B = data.subsample(self.n_start, seed=self.seed)
        X0 = torch.tensor(B["X0"], dtype=torch.float32)
        S0 = torch.tensor(B["shape0"], dtype=torch.float32)
        R0 = torch.tensor(np.nan_to_num(B["reach0"] / np.maximum(B["X0"][:, 2:4], 1e-6)), dtype=torch.float32)
        U = torch.tensor(B["U"][:, : self.steps], dtype=torch.float32)
        Y = torch.tensor(B["Y"][:, : self.steps + 1], dtype=torch.float32)
        mask = ~torch.isnan(Y[..., 0])
        Y = torch.nan_to_num(Y)
        self.mu = torch.cat([X0, S0, U[:, 0], R0], -1).mean(0)
        self.sd = torch.cat([X0, S0, U[:, 0], R0], -1).std(0) + 1e-3
        self.scale = torch.tensor([0.05, 0.05, 0.1, 0.1, 0.5])  # as the B refit's residual weights
        n_in = len(self.mu)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 5),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = len(X0)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for k in range(0, n, 512):
                i = perm[k : k + 512]
                pred = self._roll(X0[i], S0[i], U[i], R0[i], self.steps)
                err = ((pred - Y[i]) / self.scale) ** 2
                loss = (err * mask[i][..., None]).sum() / mask[i].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.n_params = sum(p.numel() for p in self.net.parameters())
        return self

    def _roll(self, X0, S0, U, R0, steps):
        import torch

        X, out = X0, [X0]
        for t in range(steps):
            z = (torch.cat([X, S0, U[:, t], R0], -1) - self.mu) / self.sd
            Xn = X + DT * self.net(z)
            Xn = torch.clamp(Xn, min=0)
            Xn = torch.cat([torch.minimum(Xn[:, :4], X0[:, :4]), Xn[:, 4:]], -1)
            X = Xn
            out.append(X)
        return torch.stack(out, 1)

    def rollout(self, B, stalkers=False):
        import torch

        with torch.no_grad():
            X0 = torch.tensor(B["X0"], dtype=torch.float32)
            S0 = torch.tensor(B["shape0"], dtype=torch.float32)
            R0 = torch.tensor(
                np.nan_to_num(B["reach0"] / np.maximum(B["X0"][:, 2:4], 1e-6)), dtype=torch.float32
            )
            U = torch.tensor(B["U"], dtype=torch.float32)
            Y = self._roll(X0, S0, U, R0, U.shape[1]).numpy().astype(float)
        # float32 rounding must not count as "alive above the start" in the blow-up check
        Y[:, :, :4] = np.minimum(Y[:, :, :4], B["X0"][:, None, :4])
        Y[:, 0] = B["X0"]
        return Y

    def complexity(self):
        return {"terms": self.n_params, "numbers": self.n_params}

    def coefficients(self):
        return {}


# ---------------------------------------------------------------------------- other optimizers
def _standardised_fit(opt, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a PySINDy optimizer on standardised features and target (as `stlsq`), original units back."""
    scale, ys = theta.std(axis=0) + 1e-12, y.std() + 1e-12
    opt.fit(theta / scale, (y / ys)[:, None])
    return np.asarray(opt.coef_).ravel() * ys / scale


class OptimizerA(ModelA):
    """A's library and alive law, with the health and distance equations fitted by another optimizer.

    optimizer: "sr3" (L0, lam = thr^2 / 2 so the implied threshold matches STLSQ's), "ssr", "frols"
    (both with an L0 penalty kappa on the path), on A's candidate columns.
    """

    def __init__(self, optimizer: str = "sr3", strength: float = 0.15, **kw):
        super().__init__(**kw)
        self.optimizer, self.strength = optimizer, strength

    def _opt(self):
        import pysindy as ps

        s = self.strength
        return {
            "sr3": lambda: ps.SR3(reg_weight_lam=s**2 / 2, regularizer="L0", max_iter=100),
            "sr3-unbiased": lambda: ps.SR3(
                reg_weight_lam=s**2 / 2, regularizer="L0", max_iter=100, unbias=True
            ),
            "ssr": lambda: ps.SSR(criteria="model_residual", kappa=s),
            "frols": lambda: ps.FROLS(kappa=s),
        }[self.optimizer]()

    def fit(self, data):
        import warnings

        from sindy_wm.models.intent_model import FIRE_FAMILY, MOVE_FAMILY, REACH_COLS, _family_cols, library

        super().fit(data)
        rows = data.rates
        F = library(
            rows[STATE].to_numpy(),
            rows[CTRL].to_numpy(),
            rows[SHAPE].to_numpy(),
            self.P,
            rows[REACH_COLS].to_numpy(),
            True,
        )
        names = list(F)
        theta = np.column_stack([F[n] for n in names])
        for eq, family in (
            ("health_blue", FIRE_FAMILY),
            ("health_red", FIRE_FAMILY),
            ("distance", MOVE_FAMILY),
        ):
            cols = [j for j in _family_cols(names, family) if names[j] not in self.exclude]
            c = np.zeros(len(names))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c[cols] = _standardised_fit(self._opt(), theta[:, cols], rows[f"d_{eq}"].to_numpy())
            self.m.coef[eq] = c
        return self


# ---------------------------------------------------------------------------------- symmetry
class SymmetricA(ModelA):
    """A with side symmetry imposed (identical units): one firepower coefficient k for both sides
    (H_B' = k R_R, H_R' = k R_B + k_wd R_B wd) and, optionally, one alive exponent p for both sides
    (Red keeps its focus term). Equality constraints by reparametrisation: exact least squares on
    A's selected terms, the same as ConstrainedSR3 with those constraints and no further pruning.
    """

    def __init__(self, alive: bool = True, **kw):
        super().__init__(**kw)
        self.alive = alive

    def fit(self, data):
        super().fit(data)
        r, names = data.rates, self.m.names
        Rb, Rr, wd = r["reach_blue"].to_numpy(), r["reach_red"].to_numpy(), r["withdraw"].to_numpy()
        y = np.concatenate([r["d_health_blue"].to_numpy(), r["d_health_red"].to_numpy()])
        T = np.column_stack([np.concatenate([Rr, Rb]), np.concatenate([0 * Rr, Rb * wd])])
        (k, k_wd), *_ = np.linalg.lstsq(T, y, rcond=None)
        for eq in ("health_blue", "health_red"):
            self.m.coef[eq] = np.zeros(len(names))
        self.m.coef["health_blue"][names.index("Rr")] = k
        self.m.coef["health_red"][names.index("Rb")] = k
        self.m.coef["health_red"][names.index("Rb*wd")] = k_wd
        if self.alive:
            gs, ys, fs = [], [], []
            for side in ("blue", "red"):
                ok = ((r[f"health_{side}"] > 0.03) & (r[f"alive_{side}"] > 0)).to_numpy()
                g = (r[f"d_health_{side}"] / r[f"health_{side}"]).to_numpy()[ok]
                gs.append(g)
                ys.append((r[f"d_alive_{side}"] / r[f"alive_{side}"]).to_numpy()[ok])
                fs.append(g * r["focus"].to_numpy()[ok] if side == "red" else 0 * g)
            T = np.column_stack([np.concatenate(gs), np.concatenate(fs)])
            (p, p_focus), *_ = np.linalg.lstsq(T, np.concatenate(ys), rcond=None)
            self.m.p["blue"] = {"1": p, "hold": 0.0, "wd": 0.0, "focus": 0.0}
            self.m.p["red"] = {"1": p, "hold": 0.0, "wd": 0.0, "focus": p_focus}
        return self

    def complexity(self):
        n = 2 + 5 + (2 if self.alive else 3)  # shared k, k_wd; distance 5; alive p, p_focus (or A's 3)
        return {"terms": n, "numbers": n + _closure_numbers(self.P)}


# ------------------------------------------------------------------------------------ phases
PHASES = ("approach", "engaged", "mopup")


def phase_of(X: np.ndarray, reach: np.ndarray, mop: float = 0.3) -> np.ndarray:
    """0 approach (under half a unit in reach on both sides), 2 mop-up (a side at <= 3 units), else 1."""
    ph = np.ones(len(X), int)
    ph[(reach[:, 0] < 0.05) & (reach[:, 1] < 0.05)] = 0
    ph[np.minimum(X[:, 2], X[:, 3]) <= mop] = 2
    return ph


class PhaseA(ModelA):
    """Hybrid SINDy: A's library fitted separately in each phase, switched by an explicit rule on the
    current (predicted) state. Phases with < 2000 rows fall back to the global fit."""

    def fit(self, data):
        super().fit(data)
        r = data.rates
        ph = phase_of(r[STATE].to_numpy(), r[["reach_blue", "reach_red"]].to_numpy())
        self.models, self.phase_rows = [], {}
        for k, name in enumerate(PHASES):
            sub = r[ph == k]
            self.phase_rows[name] = len(sub)
            if len(sub) < 2000:
                self.models.append(self.m)
                continue
            self.models.append(
                IntentModel(self.P, exclude=self.exclude).fit(
                    sub, thr_health=self.thr_health, reach_true=True, thr_shape=None
                )
            )
        return self

    def rollout(self, B, stalkers=False):
        from sindy_wm.models.intent_model import library, team_shares
        from sindy_wm.models.switch_library import scaled

        ms = [scaled(m, "marine", "stalker") for m in self.models] if stalkers else self.models
        P = ms[0].P
        X, S, U = B["X0"].astype(float).copy(), B["shape0"].astype(float), B["U"]
        offset = B["reach0"] / np.maximum(X[:, 2:4], 1e-9) - np.column_stack(team_shares(X, S, P))
        out = [X.copy()]
        self.phase_trace = []
        for n in range(U.shape[1]):
            u = U[:, n]
            s = np.column_stack(team_shares(X, S, P))
            reach = X[:, 2:4] * np.clip(s + offset * np.exp(-n * DT), 0, 1)
            ph = phase_of(X, reach)
            self.phase_trace.append(np.bincount(ph, minlength=3))
            F = library(X, u, S, P, reach, True)
            theta = np.column_stack([F[k] for k in ms[0].names])
            Xn = X.copy()
            for k, m in enumerate(ms):
                i = ph == k
                if not i.any():
                    continue
                Ch = np.stack([m.coef["health_blue"], m.coef["health_red"]])
                Hn = np.maximum(X[i, :2] + DT * theta[i] @ Ch.T, 0)
                ratio = np.divide(Hn, X[i, :2], out=np.zeros_like(Hn), where=X[i, :2] > 1e-9)
                A = X[i, 2:4] * np.column_stack(
                    [ratio[:, 0] ** m._p_of("blue", u[i]), ratio[:, 1] ** m._p_of("red", u[i])]
                )
                d = np.maximum(X[i, 4] + DT * theta[i] @ m.coef["distance"], 0)
                Xn[i] = np.column_stack([Hn, A, d])
            X = Xn
            out.append(X.copy())
        return np.stack(out, 1)

    def complexity(self):
        n = sum(m.n_terms() for m in {id(m): m for m in self.models}.values())
        return {"terms": n, "numbers": n + _closure_numbers(self.P) + 2}  # + the two phase thresholds

    def coefficients(self):
        out = {}
        for name, m in zip(PHASES, self.models, strict=True):
            out |= {f"{name} {k}": v for k, v in intent_coefficients(m).items()}
        return out | {f"rows {k}": v for k, v in self.phase_rows.items()}


# ------------------------------------------------------------------ share in reach, one-step SINDy
SHARE_TERMS = ("sa", "sk", "sf_adv", "sf_hold", "sf_wd", "so_wd", "sd")


def _sg_rate(values: np.ndarray, groups: np.ndarray, window: int = 9) -> np.ndarray:
    """Savitzky-Golay rate of a per-step series, per battle (as `add_rates`)."""
    from scipy.signal import savgol_filter

    out = np.full(len(values), np.nan)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for a, b in zip(starts, ends, strict=True):
        if b - a > window:
            out[a:b] = savgol_filter(values[a:b], window, 2, deriv=1, delta=DT)
    return out


class ShareSINDy(_LabModel):
    """S's structure with every coefficient from one-step regressions (no rollout fitting).

    Health and distance: A. Alive: A's law fitted on the reach regressor (G/alive-reach) if
    `alive_reach`. Share in reach s = R / A per side, a state of its own, with S's seven candidates

        s' = sa clo' + sk (clo - s) + C (1 - s)(sf_adv adv + sf_hold hold + sf_wd wd)
             - so_wd wd s + sd (1 - s)(ln A)'

    (clo = the closure's share, C = chance that any Blue unit is in reach), selected by STLSQ on the
    SG-9 rate of the measured share. Rolled out with the same simulator as S.
    """

    def __init__(self, thr: float = 0.05, alive_reach: bool = True, signs: bool = False):
        self.thr, self.alive_reach, self.signs = thr, alive_reach, signs

    def fit(self, data):
        from sindy_wm.models.intent_model import team_shares

        a = AliveVariantA("reach").fit(data) if self.alive_reach else ModelA().fit(data)
        self.n_closure = _closure_numbers(a.P)
        p = a.params()
        if self.alive_reach:
            p |= {"p_b": a.m.p["blue"]["1"], "p_r": a.m.p["red"]["1"], "p_r_focus": a.m.p["red"]["focus"]}
        r = data.raw_rates
        r = r[(r["alive_blue"] > 0) & (r["alive_red"] > 0)]
        ep = r.index.get_level_values(0).to_numpy()
        X, S, U = r[STATE].to_numpy(), r[SHAPE].to_numpy(), r[CTRL].to_numpy()
        clo = np.column_stack(team_shares(X, S, a.P))
        share = np.clip(r[["reach_blue", "reach_red"]].to_numpy() / np.maximum(X[:, 2:4], 1e-9), 0, 1)
        C = 1 - (1 - share[:, 0]) ** (10 * X[:, 2])
        self.share_fit = {}
        p["share_state"] = True
        for j, t in enumerate("br"):
            side = "blue" if t == "b" else "red"
            ds, dclo = _sg_rate(share[:, j], ep), _sg_rate(clo[:, j], ep)
            dlnA = r[f"d_alive_{side}"].to_numpy() / np.maximum(X[:, 2 + j], 1e-9)
            s = share[:, j]
            T = np.column_stack(
                [
                    dclo,
                    clo[:, j] - s,
                    C * (1 - s) * U[:, 0],
                    C * (1 - s) * U[:, 1],
                    C * (1 - s) * U[:, 2],
                    -U[:, 2] * s,
                    (1 - s) * dlnA,
                ]
            )
            ok = np.isfinite(T).all(1) & np.isfinite(ds)
            c = stlsq(T[ok], ds[ok], self.thr)
            if self.signs:  # S's sign rules (G2): all >= 0 except sd <= 0; refit without violators
                bad = np.r_[c[:6] < 0, c[6] > 0]
                if bad.any():
                    keep = (c != 0) & ~bad
                    c = np.zeros(7)
                    if keep.any():
                        c[keep], *_ = np.linalg.lstsq(T[ok][:, keep], ds[ok], rcond=None)
            self.share_fit[side] = {
                "r2": float(1 - np.sum((ds[ok] - T[ok] @ c) ** 2) / np.sum((ds[ok] - ds[ok].mean()) ** 2))
            }
            for k, v in zip(SHARE_TERMS, c, strict=True):
                p[f"{k}_{t}"] = float(v)
        self.p = p
        return self

    def coefficients(self):
        return super().coefficients() | {f"share R2 {k}": v["r2"] for k, v in self.share_fit.items()}


# ------------------------------------------------------------------------------ combined (depth)
class CombinedA(SymmetricA):
    """A with the two changes that held up in screening: one firepower coefficient for both sides
    (identical units) and the alive law fitted on the reach regressor (no attenuation). One-step
    STLSQ / least squares only; same closures and distance equation as A."""

    def __init__(self, **kw):
        super().__init__(alive=False, **kw)

    def fit(self, data):
        super().fit(data)
        AliveVariantA._fit_reach_law(self, data.rates)
        return self

    def complexity(self):
        n = 2 + 5 + 3  # shared k and k_wd; five distance terms; p_b, p_r, p_r_focus
        return {"terms": n, "numbers": n + _closure_numbers(self.P)}


# ------------------------------------------------------------------------------- Red's scan reach
RED_SCAN = {"marine": 5 + 0.375, "stalker": 5 + 0.625}  # SMAClite minimum_scan_range + target radius


class RedScanA(SymmetricA):
    """A's recipe on tables whose `reach_red` counts Red units with a Blue unit within Red's
    acquisition radius (5.375 for marines; `scripts/build_red_scan_states.py`), with Red's reach
    closure fitted at that radius. Blue's reach stays 5.75.

    shared=True: one firepower coefficient for both sides (as A*); False: k_B, k_R separately (A's
    STLSQ). The alive law is A*'s reach-regressor law in both cases. Only meaningful on the
    `_redscan` tables: on the original tables the measured Red reach and the closure disagree.
    """

    def __init__(self, shared: bool = True, **kw):
        super().__init__(alive=False, **kw)
        self.shared = shared

    def fit(self, data):
        from sindy_wm.models.intent_model import fit_derived_share

        if self.P is None:
            P = closures_for(data)
            P["share_red"] = fit_derived_share(data.rates, "red", rho=RED_SCAN["marine"])[0]
            self.P = P
        if self.shared:
            SymmetricA.fit(self, data)  # alive=False: only k, k_wd are replaced
        else:
            ModelA.fit(self, data)
        AliveVariantA._fit_reach_law(self, data.rates)
        return self

    def rollout(self, B, stalkers=False):
        from sindy_wm.models.switch_library import scaled

        if not stalkers:
            return self.m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])
        m = scaled(self.m, "marine", "stalker")
        m.P["share_red"][3] = RED_SCAN["stalker"]  # scaled() scales rho by attack reach; Red: scan radius
        return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])

    def complexity(self):
        n = (2 if self.shared else 3) + 5 + 3
        return {"terms": n, "numbers": n + _closure_numbers(self.P)}
