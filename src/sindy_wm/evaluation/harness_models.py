"""Reference models for the final-exploration harness (`evaluation.harness`).

    Persistence       nothing changes
    Lanchester        square, linear or mixed law on team health, with no range or control.
                      Alive follows health in proportion, and the distance is held
    NoControl         the physics library without intents (`IntentModel(ctrl=False)`)
    ModelA            `model_derived`'s recipe: closures and derived reach curve fitted on the
                      split's training battles, then STLSQ (threshold 0.15) on SG-9 rates
    ModelB            A's terms, coefficients and offset decay refitted on rollout error
                      (1-8 s, 20,000 start points), as in `sindyc_07`
    ModelS            B plus the share in reach as a state (the 11 kept share terms refitted)

Every model is refitted from scratch on each split's training battles. Nothing is read from the
pickles except S's term set and starting values.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np

from sindy_wm.models.intent_model import (
    ALIVE_CONTROL_TERMS,
    DT,
    IntentModel,
    fit_closures,
    fit_derived_share,
)
from sindy_wm.models.switch_library import UNITS, scaled
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

PROCESSED = DATA_DIR / "processed" / "exp06_intents"
SPLIT_EXCLUDE = ALIVE_CONTROL_TERMS + ("close*hold",)  # as scripts/build_models.py
ALIVE, HEALTH = ["p_b", "p_r", "p_r_focus"], ["kb", "kr", "kr_wd"]
DIST, TAU = ["d_adv", "d_wd", "d_flee", "d_hold_far", "d_hold_contact"], ["tau_b", "tau_r"]
SHARE = ("sa", "sk", "sf_adv", "sf_hold", "sf_wd", "so_wd", "sd")


def lab():
    """`notebooks/sindyc_07_lab.py`: the parameter-dict simulator B and S are defined in."""
    path = str(PROJECT_ROOT / "notebooks")
    if path not in sys.path:
        sys.path.insert(0, path)
    import sindyc_07_lab

    return sindyc_07_lab


class Persistence:
    can_scale = True

    def fit(self, data):
        return self

    def rollout(self, B, stalkers=False):
        return np.repeat(B["X0"][:, None].astype(float), B["U"].shape[1] + 1, 1)

    def complexity(self):
        return {"terms": 0, "numbers": 0}

    def coefficients(self):
        return {}


class Lanchester:
    """H_B' = -(a H_R + c H_B H_R), H_R' = -(b H_B + e H_B H_R): square (a, b), linear (c, e), mixed."""

    can_scale = True

    def __init__(self, law: str = "square"):
        self.law = law

    def fit(self, data):
        r = data.rates
        Hb, Hr = r["health_blue"].to_numpy(), r["health_red"].to_numpy()
        cols = {"square": [0], "linear": [1], "mixed": [0, 1]}[self.law]
        self.k = {}
        for side, own, opp in (("blue", Hb, Hr), ("red", Hr, Hb)):
            T = np.column_stack([opp, own * opp])[:, cols]
            c, *_ = np.linalg.lstsq(T, -r[f"d_health_{side}"].to_numpy(), rcond=None)
            full = np.zeros(2)
            full[cols] = c
            self.k[side] = full
        return self

    def rollout(self, B, stalkers=False):
        f = UNITS["marine"]["tau"] / UNITS["stalker"]["tau"] if stalkers else 1.0
        X0 = B["X0"].astype(float)
        X, out = X0.copy(), [X0.copy()]
        for _ in range(B["U"].shape[1]):
            Hb, Hr = X[:, 0], X[:, 1]
            db = -f * (self.k["blue"][0] * Hr + self.k["blue"][1] * Hb * Hr)
            dr = -f * (self.k["red"][0] * Hb + self.k["red"][1] * Hb * Hr)
            X = X.copy()
            X[:, :2] = np.maximum(X[:, :2] + DT * np.column_stack([db, dr]), 0)
            X[:, 2:4] = X0[:, 2:4] * np.divide(
                X[:, :2], X0[:, :2], out=np.zeros_like(X[:, :2]), where=X0[:, :2] > 0
            )
            out.append(X)
        return np.stack(out, 1)

    def complexity(self):
        n = sum(int(np.count_nonzero(v)) for v in self.k.values())
        return {"terms": n, "numbers": n}

    def coefficients(self):
        names = ("square", "linear")
        return {f"{s}_{n}": v[i] for s, v in self.k.items() for i, n in enumerate(names) if v[i]}


def closures_for(data) -> dict:
    """Movement switches and the derived reach curve, fitted on these training battles only."""
    cl = fit_closures(data.rates)
    return {
        "gates": cl["gates"],
        "share_blue": fit_derived_share(data.rates, "blue")[0],
        "share_red": fit_derived_share(data.rates, "red")[0],
        "form": "derived",
    }


def _closure_numbers(P: dict) -> int:
    n = sum(np.size(v) for v in P["gates"].values())
    return int(n + 3 + 3)  # lam, beta, g per side (rho is the game's)


class NoControl:
    can_scale = True

    def fit(self, data):
        self.P = closures_for(data)
        self.m = IntentModel(self.P, ctrl=False).fit(data.rates, thr_shape=None)
        return self

    def rollout(self, B, stalkers=False):
        m = scaled(self.m, "marine", "stalker") if stalkers else self.m
        return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])

    def complexity(self):
        return {"terms": self.m.n_terms(), "numbers": self.m.n_terms() + _closure_numbers(self.P)}

    def coefficients(self):
        return intent_coefficients(self.m)


def intent_coefficients(m: IntentModel) -> dict:
    out = {}
    for eq, c in m.coef.items():
        for n, v in zip(m.names, c, strict=True):
            if v:
                out[f"{eq}: {n}"] = v
    for side in ("blue", "red"):
        for k, v in m.p[side].items():
            if v:
                out[f"alive_{side}: {k}"] = v
    return out


class ModelA:
    """`model_derived` refitted: closures, then STLSQ on SG-9 rates with the measured reach."""

    can_scale = True

    def __init__(self, thr_health: float = 0.15, exclude=SPLIT_EXCLUDE, P: dict | None = None):
        self.thr_health, self.exclude, self.P = thr_health, tuple(exclude), P

    def fit(self, data):
        self.P = self.P or closures_for(data)
        self.m = IntentModel(self.P, exclude=self.exclude).fit(
            data.rates, thr_health=self.thr_health, reach_true=True, thr_shape=None
        )
        return self

    def rollout(self, B, stalkers=False):
        m = scaled(self.m, "marine", "stalker") if stalkers else self.m
        return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])

    def complexity(self):
        return {"terms": self.m.n_terms(), "numbers": self.m.n_terms() + _closure_numbers(self.P)}

    def coefficients(self):
        return intent_coefficients(self.m)

    def params(self) -> dict:
        """The lab parameter dict of this model (what `sindyc_07_lab.base_params` gives for A)."""
        c, n = self.m.coef, self.m.names

        def g(eq, k):
            return float(c[eq][n.index(k)]) if k in n else 0.0

        return {
            "kb": g("health_blue", "Rr"),
            "kr": g("health_red", "Rb"),
            "kr_wd": g("health_red", "Rb*wd"),
            "d_adv": g("distance", "close*adv"),
            "d_wd": g("distance", "close*wd"),
            "d_flee": g("distance", "flee*wd"),
            "d_hold_far": g("distance", "close*hold*far"),
            "d_hold_contact": g("distance", "close*hold*contact"),
            "p_b": self.m.p["blue"]["1"],
            "p_r": self.m.p["red"]["1"],
            "p_r_focus": self.m.p["red"].get("focus", 0.0),
            "tau_b": 1.0,
            "tau_r": 1.0,
            "lam_b": self.P["share_blue"][0],
            "beta_b": self.P["share_blue"][1],
            "g_b": self.P["share_blue"][2],
            "lam_r": self.P["share_red"][0],
            "beta_r": self.P["share_red"][1],
            "g_r": self.P["share_red"][2],
            "rho": self.P["share_blue"][3],
            "gates": self.P["gates"],
            "rho_wd_b": 0.0,
            "rho_wd_r": 0.0,
            "rho_hold_b": 0.0,
            "rho_hold_r": 0.0,
        }


class _LabModel:
    """A model defined by a `sindyc_07_lab` parameter dict."""

    can_scale = True
    p: dict
    n_closure: int

    def rollout(self, B, stalkers=False):
        L = lab()
        return L.simulate(L.scale_params(self.p) if stalkers else self.p, B)

    def _terms(self) -> dict:
        p = self.p
        out = {k: p[k] for k in HEALTH + DIST + ALIVE if p.get(k)}
        if p.get("share_state"):
            out |= {k: p[k] for k in p if k.startswith(tuple(f"{s}_" for s in SHARE)) and p[k]}
        return out

    def complexity(self):
        n = len(self._terms())
        return {"terms": n, "numbers": n + 2 + self.n_closure}  # + the two offset decay times

    def coefficients(self):
        return {**self._terms(), **{k: self.p[k] for k in TAU}}


class ModelB(_LabModel):
    """A's support with coefficients refitted on rollout error; `base` gives the starting model."""

    def __init__(self, n_start: int = 20_000, base=None):
        self.n_start, self.base = n_start, base

    def fit(self, data):
        a = (self.base or ModelA)().fit(data)
        self.n_closure = _closure_numbers(a.P)
        Bfit = data.subsample(self.n_start)
        self.p, _ = lab().refit(ALIVE + HEALTH + DIST + TAU, a.params(), Bfit)
        return self


class ModelS(_LabModel):
    """B, then the share-in-reach terms kept in `sindyc_07` refitted with the reach error weighted in."""

    def __init__(self, n_start: int = 20_000, base=None):
        self.n_start, self.base = n_start, base

    def fit(self, data):
        b = ModelB(self.n_start, base=self.base).fit(data)
        self.n_closure = b.n_closure
        S0 = pickle.load(open(PROCESSED / "sindyc_07_models.pkl", "rb"))["S share as a state"]
        keep = [k for k in S0 if k.startswith(tuple(f"{s}_" for s in SHARE)) and S0[k]]
        start = dict(
            b.p, share_state=True, **{k: S0[k] for k in S0 if k.startswith(tuple(f"{s}_" for s in SHARE))}
        )
        self.p, _ = lab().refit(keep, start, data.subsample(self.n_start), reach_w=1.0)
        return self
