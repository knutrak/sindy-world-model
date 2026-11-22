"""Reach from the distribution of distance margins: μ, σ as states (docs/generalisation/10_mu_sigma_plan.md).

For each side s and unit type a, μ_sa and σ_sa are the mean and spread of the alive units' reach margins
(`states.margins`). Units in reach: R_sa = A_sa · Φ((c − μ_sa) / (k σ_sa)). μ′ and σ′ are discovered from a
generic library: polynomials of degree <= 2 in the blocks (μ, σ [, p], μ_o, σ_o [, p_o], A_own, A_opp),
deaths terms ((ln A_own)′, (ln A_opp)′ × {1, μ, σ}), all × the intents; two-level STLSQ in weak form
(`reach_state`'s machinery). Rolled out inside the frozen pipeline through `Pipeline._reach`.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.special import ndtr

from sindy_wm.models.intent_model import CTRL, DT
from sindy_wm.models.kite import control_vector_k
from sindy_wm.models.pipeline import OTHER, SIDES
from sindy_wm.models.reach_state import (
    BLOW,
    ReachStatePipeline,
    bump,
    control_arrays,
    design,
)

DEATHS = ("dA", "mu*dA", "sd*dA", "dAo", "mu*dAo", "sd*dAo")


def blocks_names(with_p: bool) -> list[str]:
    own = ["mu", "sd", "p"] if with_p else ["mu", "sd"]
    opp = ["mu_o", "sd_o", "p_o"] if with_p else ["mu_o", "sd_o"]
    return [*own, *opp, "A_own", "A_opp"]


def group_names(with_p: bool) -> list[str]:
    b = blocks_names(with_p)
    names = ["1"]
    for k in (1, 2):
        names += ["*".join(c) for c in combinations_with_replacement(b, k)]
    return names + list(DEATHS)


def share(mu, sd, c: float, k: float):
    with np.errstate(all="ignore"):
        return np.nan_to_num(ndtr((c - mu) / np.maximum(k * sd, 1e-3)))


def edge_density(mu, sd, c: float, k: float):
    """φ(z), z = (c − μ)/(kσ): the normal density at the reach edge (0 where undefined)."""
    with np.errstate(all="ignore"):
        z = (c - mu) / np.maximum(k * sd, 1e-3)
        return np.nan_to_num(np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))


def state_groups(b: dict, names: list[str]) -> np.ndarray:
    n = len(b["mu"])
    out = np.empty((n, len(names)))
    for j, nm in enumerate(names):
        if nm == "1":
            out[:, j] = 1.0
        else:
            v = np.ones(n)
            for f in nm.split("*"):
                v = v * b[f]
            out[:, j] = v
    return out


def opp_blocks(mu: dict, sd: dict, A: dict, side: str, types, c, k):
    """The opposing side's μ, σ, p, averaged over its types weighted by units alive."""
    o = OTHER[side]
    w = [A[(o, t)] for t in types]
    tot = sum(w)
    with np.errstate(all="ignore"):
        m = np.nan_to_num(sum(np.nan_to_num(mu[(o, t)]) * wt for t, wt in zip(types, w, strict=True)) / tot)
        s = np.nan_to_num(sum(np.nan_to_num(sd[(o, t)]) * wt for t, wt in zip(types, w, strict=True)) / tot)
        p = np.nan_to_num(
            sum(share(mu[(o, t)], sd[(o, t)], c, k) * wt for t, wt in zip(types, w, strict=True)) / tot
        )
    return m, s, p


# ------------------------------------------------------------------ rows from data
def rows_for(
    states: pd.DataFrame,
    types,
    side: str,
    a: str,
    target: str,
    with_p: bool,
    controls,
    method: str,
    c: float,
    k: float,
    radius: int = 8,
    stride: int = 8,
    sg: int = 9,
    names: list[str] | None = None,
):
    """Regression rows for μ′ (target "mu") or σ′ ("sd") of (side, a). `states`: typed + margins + controls,
    indexed (episode_idx, step). Weak: one row per window (∫φΘ, −∫φ′x); step: SG-9 rate. Returns Theta
    (float32),
    b, battle ids, names."""
    from scipy.signal import savgol_filter

    names = names or group_names(with_p)
    ctrl_cols = [*CTRL, "kite"] if "kite" in controls else CTRL
    phi, dphi, w = bump(radius)
    Th, bs, ids = [], [], []
    tt = [t for t in types]
    for ep, g in states.groupby(level=0, sort=False):
        A = {(s, t): g[f"alive_{s}_{t}"].to_numpy(float) for s in SIDES for t in tt}
        mu = {(s, t): g[f"mu_{s}_{t}"].to_numpy(float) for s in SIDES for t in tt}
        sd = {(s, t): g[f"sd_{s}_{t}"].to_numpy(float) for s in SIDES for t in tt}
        own_ok = (A[(side, a)] > 0) & np.isfinite(mu[(side, a)])
        opp_ok = sum(A[(OTHER[side], t)] for t in tt) > 0
        ok = own_ok & opp_ok
        n = len(ok)
        if n < 2 * radius + 3:
            continue
        mo, so, po = opp_blocks(mu, sd, A, side, tt, c, k)
        A_opp = sum(A[(OTHER[side], t)] for t in tt)
        b = {
            "mu": np.nan_to_num(mu[(side, a)]),
            "sd": np.nan_to_num(sd[(side, a)]),
            "p": share(mu[(side, a)], sd[(side, a)], c, k),
            "mu_o": mo,
            "sd_o": so,
            "p_o": po,
            "A_own": A[(side, a)],
            "A_opp": A_opp,
        }
        # moment-closure blocks (docs/generalisation/12): units still walking, density at the reach edge
        b["q"], b["q_o"] = 1 - b["p"], 1 - po
        b["phi"] = edge_density(mu[(side, a)], sd[(side, a)], c, k)
        wts = [A[(OTHER[side], t)] for t in tt]
        with np.errstate(all="ignore"):
            b["phi_o"] = np.nan_to_num(
                sum(
                    edge_density(mu[(OTHER[side], t)], sd[(OTHER[side], t)], c, k) * wt
                    for t, wt in zip(tt, wts, strict=True)
                )
                / sum(wts)
            )
        lnA, lnAo = np.log(np.maximum(A[(side, a)], 1e-9)), np.log(np.maximum(A_opp, 1e-9))
        b["dA"] = np.r_[np.diff(lnA) / DT, 0.0]
        b["dAo"] = np.r_[np.diff(lnAo) / DT, 0.0]
        x = b[target]
        Theta = design(state_groups(b, names), control_arrays(g[ctrl_cols].to_numpy(float), controls))
        if method == "weak":
            for cc in range(radius, n - radius - 1, stride):
                if not ok[cc - radius : cc + radius + 2].all():
                    continue
                sl = slice(cc - radius, cc + radius + 1)
                Th.append(((w * phi) @ Theta[sl]).astype(np.float32))
                bs.append(-(w * dphi) @ x[sl])
                ids.append(ep)
        else:
            dx = savgol_filter(x, sg, 2, deriv=1, delta=DT)
            h = sg // 2
            for kk in range(h, n - h - 1, stride):
                if ok[kk - h : kk + h + 2].all():
                    Th.append(Theta[kk].astype(np.float32))
                    bs.append(dx[kk])
                    ids.append(ep)
    return np.array(Th, np.float32), np.array(bs), np.array(ids), names


# ------------------------------------------------------------------ the model
class MuSigmaPipeline(ReachStatePipeline):
    """Frozen pipeline with reach from μ, σ states. `eqs[(side, type)] = {"mu": coef, "sd": coef}`; `names`,
    `controls`, `with_p`, `c`, `k` as fitted. B needs "mu0", "sd0" (N, keys in L.pos order). `eqs=None`
    keeps the
    frozen reach (inherited)."""

    control_vector = staticmethod(control_vector_k)  # five controls (kite last), for the planner

    @classmethod
    def build(cls, model, eqs, names, controls, with_p: bool, c: float, k: float):
        m = cls.from_frozen(model, eqs)
        m.ms_names, m.ms_controls, m.with_p, m.c, m.k = names, controls, with_p, c, k
        return m

    def _reach(self, n, X, S, F, u, sh, offset, B) -> dict:
        if self.eqs is None:
            return super()._reach(n, X, S, F, u, sh, offset, B)
        L = self.L
        tt = L.types
        keys = [(s, a) for s in SIDES for a in tt]
        A = {kk: X[:, L.idx("alive", *kk)] for kk in keys}
        if n == 0:
            self._mu = {kk: np.nan_to_num(B["mu0"][:, L.pos(*kk)]) for kk in keys}
            self._sd = {kk: np.maximum(np.nan_to_num(B["sd0"][:, L.pos(*kk)]), 0) for kk in keys}
            self.clipped = np.zeros(len(X), bool)
        else:
            Xp, up = self._prev[0], self._prev[3]
            Ap = {kk: Xp[:, L.idx("alive", *kk)] for kk in keys}
            new_mu, new_sd = {}, {}
            for kk in keys:
                side, a = kk
                mo, so, po = opp_blocks(self._mu, self._sd, Ap, side, tt, self.c, self.k)
                A_opp_p = sum(Ap[(OTHER[side], t)] for t in tt)
                A_opp_n = sum(A[(OTHER[side], t)] for t in tt)
                b = {
                    "mu": self._mu[kk],
                    "sd": self._sd[kk],
                    "p": share(self._mu[kk], self._sd[kk], self.c, self.k),
                    "mu_o": mo,
                    "sd_o": so,
                    "p_o": po,
                    "A_own": Ap[kk],
                    "A_opp": A_opp_p,
                    "dA": (np.log(np.maximum(A[kk], 1e-9)) - np.log(np.maximum(Ap[kk], 1e-9))) / DT,
                    "dAo": (np.log(np.maximum(A_opp_n, 1e-9)) - np.log(np.maximum(A_opp_p, 1e-9))) / DT,
                }
                Th = design(state_groups(b, self.ms_names), control_arrays(up, self.ms_controls))
                mu = self._mu[kk] + DT * (Th @ self.eqs[kk]["mu"])
                sd = self._sd[kk] + DT * (Th @ self.eqs[kk]["sd"])
                alive = A[kk] > 1e-9
                self.blown |= alive & (~np.isfinite(mu) | ~np.isfinite(sd) | (sd < -BLOW) | (np.abs(mu) > 20))
                mu, sd = np.nan_to_num(mu), np.maximum(np.nan_to_num(sd), 0)
                bounds = getattr(self, "bounds", None)  # docs/planning/34: train range [0.5%, 99.5%]
                if bounds is not None:
                    lo, hi, sd_hi = bounds[kk]
                    self.clipped |= alive & ((mu < lo) | (mu > hi) | (sd > sd_hi))
                    mu, sd = np.clip(mu, lo, hi), np.minimum(sd, sd_hi)
                new_mu[kk], new_sd[kk] = mu, sd
            self._mu, self._sd = new_mu, new_sd
        R = {kk: A[kk] * share(self._mu[kk], self._sd[kk], self.c, self.k) for kk in keys}
        self._prev = (X.copy(), S.copy(), F.copy(), u.copy(), {kk: v.copy() for kk, v in sh.items()})
        self._log.append(np.column_stack([sum(R[(s, a)] for a in tt) for s in SIDES]))
        return R


def train_bounds(states: pd.DataFrame, types) -> dict:
    """Per (side, type): the [0.5%, 99.5%] range of μ and the 99.5% value of σ over steps with units alive
    (docs/planning/34; train only)."""
    out = {}
    for s in SIDES:
        for t in types:
            ok = states[f"n_{s}_{t}"] > 0
            mu, sd = states.loc[ok, f"mu_{s}_{t}"], states.loc[ok, f"sd_{s}_{t}"]
            out[(s, t)] = (float(mu.quantile(0.005)), float(mu.quantile(0.995)), float(sd.quantile(0.995)))
    return out


def attach_margins(B: dict, states: pd.DataFrame, L) -> dict:
    """B["mu0"], B["sd0"]: the measured μ, σ at each start point (keys in L.pos order)."""
    keys = [f"{s}_{a}" for s in SIDES for a in L.types]
    for k0, n in (("mu0", "mu"), ("sd0", "sd")):
        cols = [f"{n}_{kk}" for kk in keys]
        arr = {ep: g[cols].to_numpy(float) for ep, g in states.groupby(level=0, sort=False)}
        B[k0] = np.array([arr[e][kk] for e, kk in zip(B["ep"], B["k0"], strict=True)])
    return B
