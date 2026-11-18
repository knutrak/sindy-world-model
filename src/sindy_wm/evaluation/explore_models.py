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
        elif self.law == "proportional":
            for side in ("blue", "red"):
                self.m.p[side] = {"1": 1.0, "hold": 0.0, "wd": 0.0, "focus": 0.0}
        else:
            raise ValueError(self.law)
        return self


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
