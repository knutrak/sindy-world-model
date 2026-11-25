# ruff: noqa: E501, E731, B023
"""One model with unit properties as inputs, with switchable extensions (docs/overnight/00_plan.md).

`Param` is rolled out for any army (its `Layout`); the numbers in `par` are shared by all armies and fitted on a
pool of them (`experiments/overnight/fit.py`). With `opt = {}` it is the small single model of docs/single:
H_s,t' = -eta_s sum_a r(a, t) R_a share_t, a pooled alive law, reach closure and distance equation, with distances
scaled by 5.75 / reach and speeds by speed / 3.15.

Options (opt):
  share  "scaled": the reach closure sees lengths over reach; tau: "scaled": the start offset decays in time units
         of reach / speed; tau_s: the decay time (1 s)
  over   overkill: "shots" r / (1 + c / shots), "power" r^g, "sat" 1 / (1 + k R / A_target), "both", "int" whole shots
  alive  extra terms of the exponent: "shots", "hpu" (health per unit), "n" (units alive)
  form   formation depth and width follow a sparse equation instead of being held fixed
  vd     closing speed: the start's measured d' relaxes to the equation's value in vd seconds
  marg   reach from margins (mean, spread): "pure", "offset" or "half" (mean of margin and closure)
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from sindy_wm.models import pipeline as PL
from sindy_wm.models.intent_model import DT, derived_share
from sindy_wm.models.pipeline import OTHER, SIDES, Pipeline
from sindy_wm.states.typed import unit_stats

KW = {
    "withdraw": "rule",
    "alloc": "rule",
    "thr_health": 0.15,
    "thr_distance": 0.05,
    "max_reach": 6.0,
    "movement_wd": "rule",
}
BONUS, ARMORED = {"MARAUDER": 10.0, "STALKER": 5.0}, {"MARAUDER", "STALKER"}
MARINE_RATE = 1 / (7.5 * 10 / 16)


def shots(a: str, t: str, whole: bool = False) -> float:
    """Shots of a to kill a full unit of t."""
    a, t = a.upper(), t.upper()
    sa, st = unit_stats(a), unit_stats(t)
    dmg = sa["damage"] + (BONUS.get(a, 0.0) if t in ARMORED else 0.0)
    n = st["shield"] / dmg + st["hp"] / max(dmg - st["armor"], 0.5)
    return float(np.ceil(n)) if whole else n


def rate(a: str, t: str, whole: bool = False) -> float:
    """Full units of t removed per second by one firing a, relative to marine on marine."""
    cd = np.ceil(unit_stats(a.upper())["cooldown"] * 16) / 16
    return 1 / (shots(a, t, whole) * cd) / MARINE_RATE


def fire(a: str, t: str, opt: dict, par: dict) -> float:
    """The rate with the overkill option's per-pair factor."""
    over = opt.get("over")
    r = rate(a, t, over == "int")
    if over in ("shots", "both"):
        c = par["over"]["c"]
        r *= (1 + c / 7.5) / (1 + c / shots(a, t))
    if over == "power":
        r = r ** par["over"]["g"]
    return r


def form_features(dep, wid, q_s, q_o, dln, u, blue: bool) -> dict:
    one = np.ones_like(dep)
    f = {"1": one, "dep": dep, "wid": wid, "q": q_s, "q_o": q_o, "dep*q": dep * q_s, "wid*q": wid * q_s}
    f |= {
        "wd": u[:, 2],
        "wd*q": u[:, 2] * q_s,
        "hold*q": u[:, 1] * q_s,
        "dep*dln": dep * dln,
        "wid*dln": wid * dln,
    }
    return f


def margin_features(mu, sd, p, p_o, dd, dln, dlo, u, blue: bool, v: float) -> dict:
    """Candidates of the margin equations, lengths over reach (x 5.75); dd: scaled d'."""
    wd, one = u[:, 2], np.ones_like(mu)
    q, w = 1 - p, (1 - u[:, 2])
    return {
        "dd": dd,
        "dd*q": dd * q,
        "q": v * q,
        "q*(1-wd)": v * q * w,
        "p*wd": v * p * wd,
        "q_o": v * (1 - p_o),
        "mu": v * mu,
        "sd": v * sd,
        "mu*p": v * mu * p,
        "dln": dln,
        "dlo": dlo,
        "mu*dlo": mu * dlo,
        "sd*dln": sd * dln,
        "1": v * one,
    }


class Param(Pipeline):
    def __init__(self, types, withdraw_radius: float, par: dict):
        super().__init__(tuple(types), withdraw_radius=withdraw_radius, **KW)
        self.par, self.opt = par, par.get("opt", {})
        self.needs_margins = "marg" in self.opt
        self.needs_vd = "vd" in self.opt
        self.dscale = 5.75 / float(np.mean(list(self.rho.values())))
        self.vscale = float(np.mean([unit_stats(t.upper())["speed"] for t in self.L.types])) / 3.15
        self.dist_names, self.dist_coef, self.P_team = par["dist_names"], par["dist_coef"], par["P_team"]
        self.fire = {(a, t): fire(a, t, self.opt, par) for a in self.L.types for t in self.L.types}
        self.inv_shots = {(a, t): 7.5 / shots(a, t) for a in self.L.types for t in self.L.types}

    def _dist_library(self, X, U, S, reach):
        X = X.copy()
        X[:, 4] = X[:, 4] * self.dscale
        r, self.withdraw_radius = self.withdraw_radius, self.withdraw_radius * self.dscale
        try:
            F = super()._dist_library(X, U, S * self.dscale, reach)
        finally:
            self.withdraw_radius = r
        return {k: v * self.vscale for k, v in F.items()}

    def _share(self, X, S, F):
        tt, L, par = self.L.types, self.L, self.par
        d, n_opp = X[:, -1], {s: 10 * sum(X[:, L.idx("alive", OTHER[s], t)] for t in tt) for s in SIDES}
        dep, wid = {"blue": S[:, 0], "red": S[:, 1]}, {"blue": S[:, 2], "red": S[:, 3]}
        out = {}
        for s in SIDES:
            for a in tt:
                k = 5.75 / self.rho[a] if self.opt.get("share") == "scaled" else 1.0
                out[(s, a)] = derived_share(
                    k * (d - F[:, L.pos(s, a)]),
                    k * dep[s],
                    k * dep[OTHER[s]],
                    k * wid[s],
                    k * wid[OTHER[s]],
                    n_opp[s],
                    *par["share"][s],
                    rho=k * self.rho[a],
                )
        return out

    def rollout(self, B: dict) -> np.ndarray:
        tt, L, par, opt = self.L.types, self.L, self.par, self.opt
        X, S, F = B["X0"].astype(float).copy(), np.nan_to_num(B["shape0"].astype(float)), B["front0"]
        U = B["U"]
        tau = opt.get("tau_s", 1.0)
        if opt.get("tau") == "scaled":
            tau = tau / self.dscale / self.vscale
        sh0 = self._share(X, S, F)
        marg = opt.get("marg")
        if marg:
            mu = {
                (s, a): np.nan_to_num(B["mu0"][:, L.pos(s, a)]) * 5.75 / self.rho[a]
                for s in SIDES
                for a in tt
            }
            sd = {
                (s, a): np.nan_to_num(B["sd0"][:, L.pos(s, a)]) * 5.75 / self.rho[a]
                for s in SIDES
                for a in tt
            }
            c, k = par["marg"]["ck"]
        off, off_m = {}, {}
        for key in sh0:
            A0 = X[:, L.idx("alive", *key)]
            meas = np.divide(B["reach0"][:, L.pos(*key)], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
            off[key] = np.where(A0 > 1e-9, meas - sh0[key], 0.0)
            if marg:
                m0 = ndtr((c - mu[key]) / np.maximum(k * sd[key], 0.05))
                off_m[key] = np.where(A0 > 1e-9, meas - m0, 0.0)
        lag = None
        out = [X.copy()]
        for n in range(U.shape[1]):
            u = U[:, n]
            wd, focus = u[:, 2], u[:, 3]
            A = {(s, t): X[:, L.idx("alive", s, t)] for s in SIDES for t in tt}
            fade = np.exp(-n * DT / tau)
            sh = self._share(X, S, F)
            p = {key: np.clip(sh[key] + off[key] * fade, 0, 1) for key in sh}
            if marg:
                pm = {}
                for key in sh:
                    base = ndtr((c - mu[key]) / np.maximum(k * sd[key], 0.05))
                    pm[key] = np.clip(base + (off_m[key] * fade if marg != "pure" else 0.0), 0, 1)
                p = pm if marg != "half" else {key: 0.5 * (p[key] + pm[key]) for key in p}
            R = {key: A[key] * p[key] for key in p}
            newX, dlnA = X.copy(), {}
            for s in SIDES:
                o = OTHER[s]
                if len(tt) == 1:
                    sig = {tt[0]: np.ones(len(X))}
                else:
                    tot = sum(A[(s, t)] for t in tt)
                    sig = {t: np.divide(A[(s, t)], tot, out=np.zeros_like(tot), where=tot > 0) for t in tt}
                Ro = {a: R[(o, a)] * ((1 - wd) if o == "blue" else 1.0) for a in tt}
                sat = 1.0
                if opt.get("over") in ("sat", "both"):
                    tot_s = sum(A[(s, t)] for t in tt)
                    ratio = np.divide(sum(Ro.values()), tot_s, out=np.zeros_like(tot_s), where=tot_s > 1e-9)
                    sat = 1 / (1 + par["over"]["k"] * ratio)
                for t in tt:
                    r_tot = sum(self.fire[(a, t)] * Ro[a] for a in tt)
                    rt = np.minimum(-par["eta"][s] * r_tot * sig[t] * sat, 0)
                    h0, a0 = X[:, L.idx("health", s, t)], A[(s, t)]
                    rt = np.where(a0 > 0, rt, 0.0)
                    h1 = np.maximum(h0 + DT * rt, 0)
                    ratio = np.divide(h1, h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                    e = par["p"][s]
                    expo = e["p"] + e.get("focus", 0.0) * focus
                    if "shots" in e:
                        w = {a: A[(o, a)] for a in tt}
                        wt = sum(w.values())
                        z = sum(w[a] * self.inv_shots[(a, t)] for a in tt)
                        expo = expo + e["shots"] * np.divide(z, wt, out=np.ones_like(wt), where=wt > 1e-9)
                    if "hpu" in e:
                        expo = expo + e["hpu"] * np.clip(
                            np.divide(h0, a0, out=np.ones_like(h0), where=a0 > 1e-9), 0, 1
                        )
                    if "n" in e:
                        expo = expo + e["n"] * a0
                    a1 = a0 * ratio ** np.clip(expo, 0.05, 3)
                    newX[:, L.idx("health", s, t)], newX[:, L.idx("alive", s, t)] = h1, a1
                    dlnA[(s, t)] = np.divide(a1 - a0, a0, out=np.zeros_like(a0), where=a0 > 1e-9) / DT
            TX = L.team_view(X)
            rt = np.column_stack([sum(R[(s, a)] for a in tt) for s in SIDES])
            Fd = self._dist_library(TX, u, S, rt)
            dd = np.column_stack([Fd[j] for j in self.dist_names]) @ self.dist_coef
            if self.needs_vd:
                if lag is None:
                    lag = np.clip(B["vd0"] / self.vscale, -8, 8) * self.vscale - dd
                dd = dd + lag * np.exp(-n * DT / opt["vd"])
            newX[:, -1] = np.maximum(X[:, -1] + DT * dd, 0)
            if marg:
                tot = {s: sum(A[(s, a)] for a in tt) for s in SIDES}
                avg = lambda s, f, w: np.divide(
                    sum(w[(s, a)] * f[(s, a)] for a in tt), tot[s], out=np.zeros(len(X)), where=tot[s] > 1e-9
                )  # noqa: E731, B023
                P = {s: avg(s, p, A) for s in SIDES}
                DL = {s: avg(s, dlnA, A) for s in SIDES}
                new = {}
                for s in SIDES:
                    for a in tt:
                        ds = 5.75 / self.rho[a]
                        ff = margin_features(
                            mu[(s, a)],
                            sd[(s, a)],
                            p[(s, a)],
                            P[OTHER[s]],
                            dd * ds,
                            dlnA[(s, a)],
                            DL[OTHER[s]],
                            u,
                            s == "blue",
                            self.vscale,
                        )
                        new[(s, a)] = (
                            np.clip(
                                mu[(s, a)] + DT * sum(co * ff[j] for j, co in par["marg"][(s, "mu")].items()),
                                -7,
                                30,
                            ),
                            np.clip(
                                sd[(s, a)] + DT * sum(co * ff[j] for j, co in par["marg"][(s, "sd")].items()),
                                0.0,
                                8,
                            ),
                        )
                for key, (m1, s1) in new.items():
                    mu[key], sd[key] = m1, s1
            if opt.get("form"):
                S1 = S.copy()
                for i, s in enumerate(SIDES):
                    tot_a = sum(A[(s, a)] for a in tt)
                    ps = np.divide(
                        sum(R[(s, a)] for a in tt), tot_a, out=np.zeros(len(X)), where=tot_a > 1e-9
                    )
                    tot_o = sum(A[(OTHER[s], a)] for a in tt)
                    po = np.divide(
                        sum(R[(OTHER[s], a)] for a in tt), tot_o, out=np.zeros(len(X)), where=tot_o > 1e-9
                    )
                    dl = np.divide(
                        sum(A[(s, a)] * dlnA[(s, a)] for a in tt),
                        tot_a,
                        out=np.zeros(len(X)),
                        where=tot_a > 1e-9,
                    )
                    ff = form_features(
                        S[:, i] * self.dscale, S[:, 2 + i] * self.dscale, 1 - ps, 1 - po, dl, u, s == "blue"
                    )
                    for j, kind in ((i, "dep"), (2 + i, "wid")):
                        g = (
                            sum(co * ff[q] for q, co in par["form"][(s, kind)].items())
                            * self.vscale
                            / self.dscale
                        )
                        S1[:, j] = np.clip(S[:, j] + DT * g, 0, 8)
                S = S1
            X = newX
            out.append(X.copy())
        return np.stack(out, axis=1)


__all__ = ["Param", "PL", "fire", "form_features", "margin_features", "rate", "shots"]
