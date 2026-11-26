# ruff: noqa: E501, B023
"""Candidates from the battle shapes (docs/checks/curve_shapes.md), selected by STLSQ: health, alive, distance.

Plan: docs/generalisation/24_shape_library_plan.md (exploratory, 26 Nov). Run from the repository root:
uv run --no-sync python experiments/generalisation/shape_library.py [n_battles]
Fit on train (all battles, or the first n), roll out on val against the frozen pipeline.
"""

import copy
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import DT, add_controls, stlsq
from sindy_wm.models.pipeline import (
    DIST_EXCLUDE,
    MOVE_FAMILY,
    OTHER,
    SIDES,
    Pipeline,
    _alloc,
    _family_cols,
    prepare,
    team_rows,
)
from sindy_wm.paths import DATA_DIR

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
OUT = DATA_DIR / "processed" / "generalisation" / "shape_library"
SCEN = {"marines": "exp06_intents_typed", "mixed": "exp08_mixed_typed"}
HORIZONS = (1.0, 2.0, 4.0, 8.0)
THR_ALIVE = 0.15
MOVES = (("adv", 0), ("hold", 1), ("wd", 2))
# expected signs (plan 24); None = no expectation
SIGN_HEALTH = {"H_opp": -1, "H_opp*A_own": 1, "H_own": -1}
SIGN_ALIVE = {"law": 1, "law*focus": 1, "A_opp": -1, "R_opp": -1, "A_own": -1, "1": -1}
SIGN_DIST = {
    "settle_d*adv": -1,
    "settle_d*hold": -1,
    "settle_d*wd": None,
    "settle_1*adv": 1,
    "settle_1*hold": 1,
    "settle_1*wd": None,
}
NEW_HEALTH, NEW_DIST = tuple(SIGN_HEALTH), tuple(SIGN_DIST)


def signed_stlsq(theta, y, names, thr, signs):
    """STLSQ; a kept term whose expected sign (signs[name], if given) is wrong is dropped and the rest refitted."""
    cols, dropped = list(range(len(names))), []
    while True:
        c = np.zeros(len(names))
        c[cols] = stlsq(theta[:, cols], y, thr)
        bad = [j for j in cols if c[j] and signs.get(names[j]) and np.sign(c[j]) != signs[names[j]]]
        if not bad:
            return c, dropped
        worst = max(bad, key=lambda j: abs(c[j]) * theta[:, j].std())
        dropped.append(names[worst])
        cols.remove(worst)


class ShapeModel(Pipeline):
    """The frozen pipeline with optional new health terms, a SINDy alive equation and settling distance terms."""

    @classmethod
    def from_frozen(cls, m):
        x = copy.deepcopy(m)
        x.__class__ = cls
        x.alive_eq, x.new_dist = None, False
        return x

    # ---- terms ----
    def _terms(self, R_o, sig, A_o, wd, o, t):
        L = super()._terms(R_o, sig, A_o, wd, o, t)
        if getattr(self, "new_health", False):
            s, H, A = OTHER[o], self._st["H"], self._st["A"]
            h_opp = sum(H[(o, a)] * sig[(a, t)] for a in self.L.types)
            L["H_opp"] = h_opp
            L["H_opp*A_own"] = h_opp * A[(s, t)]
            L["H_own"] = H[(s, t)]
        return L

    def _dist_library(self, X, U, S, reach):
        F = super()._dist_library(X, U, S, reach)
        if self.new_dist:
            d = X[:, -1]
            for u, i in MOVES:
                F[f"settle_d*{u}"] = d * U[:, i]
                F[f"settle_1*{u}"] = U[:, i].astype(float)
        return F

    def _alive_feats(self, rate, A, H, focus, A_opp, R_opp, side):
        law = np.divide(rate * A, H, out=np.zeros_like(A), where=H > 1e-9)
        F = {"law": law, "A_opp": A_opp, "R_opp": R_opp, "A_own": A, "1": np.ones_like(A)}
        if side == "red":
            F["law*focus"] = law * focus
        return F

    # ---- fitting the new parts on measured rows ----
    def fit_new(self, rows, health: bool, alive: bool, dist: bool, log):
        tt, L = self.L.types, self.L
        self.new_health = health
        self._st = {
            "H": {(s, t): rows[f"health_{s}_{t}"].to_numpy() for s in SIDES for t in tt},
            "A": {(s, t): rows[f"alive_{s}_{t}"].to_numpy() for s in SIDES for t in tt},
        }
        wd, focus = rows["withdraw"].to_numpy(), rows["focus"].to_numpy()
        if health or alive:
            self.alive_eq = {} if alive else None
            for s in SIDES:
                o = OTHER[s]
                R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in tt}
                A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in tt}
                sig = self._sigma_measured(rows, o)
                for t in tt:
                    terms = self._terms(R_o, sig, A_o, wd, o, t)
                    names = list(terms)
                    theta = np.column_stack(list(terms.values()))
                    if health:
                        signs = {
                            k: -1 for k in names if k.startswith("R[") or k.startswith("A[") or k == "1"
                        } | SIGN_HEALTH
                        c, dropped = signed_stlsq(
                            theta, rows[f"d_health_{s}_{t}"].to_numpy(), names, self.thr_health, signs
                        )
                        old = self.coef[(s, t)]
                        self.coef[(s, t)] = dict(zip(names, c, strict=True))
                        log.append(
                            f"health {s} {t}\n   old: {fmt(old)}\n   new: {fmt(self.coef[(s, t)])}   [sign rule dropped {dropped}]"
                        )
                    if alive:
                        rate = theta @ np.array([self.coef[(s, t)].get(k, 0.0) for k in names])
                        h, al = rows[f"health_{s}_{t}"].to_numpy(), rows[f"alive_{s}_{t}"].to_numpy()
                        ok = (h > 0.03) & (al > 0)
                        R_blue_rule = {a: R_o[a] * (1 - wd) if o == "blue" else R_o[a] for a in tt}
                        F = self._alive_feats(
                            rate,
                            al,
                            h,
                            focus,
                            sum(A_o[a] * sig[(a, t)] for a in tt),
                            sum(R_blue_rule[a] * sig[(a, t)] for a in tt),
                            s,
                        )
                        an = list(F)
                        th = np.column_stack([F[k][ok] for k in an])
                        c, dropped = signed_stlsq(
                            th, rows[f"d_alive_{s}_{t}"].to_numpy()[ok], an, THR_ALIVE, SIGN_ALIVE
                        )
                        self.alive_eq[(s, t)] = (an, c)
                        p, pf = self.p[(s, t)]
                        log.append(
                            f"alive {s} {t}\n   old: A' = {p:.3f}·law"
                            + (f" + {pf:.3f}·law*focus" if pf else "")
                            + f"  (hand law)\n   new: {fmt(dict(zip(an, c, strict=True)))}   [sign rule dropped {dropped}]"
                        )
        if dist:
            self.new_dist = True
            T = team_rows(rows, L)
            Xt = np.column_stack(
                [
                    T[c].to_numpy()
                    for c in ("health_blue", "health_red", "alive_blue", "alive_red", "distance")
                ]
            )
            F = self._dist_library(
                Xt,
                T[self.ctrl_cols].to_numpy(),
                T[["depth_blue", "depth_red", "width_blue", "width_red"]].to_numpy(),
                T[["reach_blue", "reach_red"]].to_numpy(),
            )
            names = list(F)
            theta = np.column_stack([F[k] for k in names])
            cols = [j for j in _family_cols(names, MOVE_FAMILY) if names[j] not in DIST_EXCLUDE]
            cols = [j for j in cols if names[j] not in ("close*wd", "flee*wd")]
            cols += [names.index("danger*wd"), names.index("walk_r*wd")] + [names.index(k) for k in NEW_DIST]
            sub = [names[j] for j in cols]
            c, dropped = signed_stlsq(
                theta[:, cols], T["d_distance"].to_numpy(), sub, self.thr_distance, SIGN_DIST
            )
            old = dict(zip(self.dist_names, self.dist_coef, strict=True))
            cd = np.zeros(len(names))
            cd[cols] = c
            self.dist_names, self.dist_coef = names, cd
            log.append(
                f"distance\n   old: {fmt(old)}\n   new: {fmt(dict(zip(names, cd, strict=True)))}   [sign rule dropped {dropped}]"
            )
        return self

    # ---- simulation: Pipeline.rollout without the exploratory branches, alive by the SINDy equation if fitted ----
    def rollout(self, B):
        tt, L = self.L.types, self.L
        X, S, F, U = B["X0"].astype(float).copy(), B["shape0"].astype(float).copy(), B["front0"], B["U"]
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
            self._st = {
                "H": {(s, t): X[:, L.idx("health", s, t)] for s in SIDES for t in tt},
                "A": {(s, t): X[:, L.idx("alive", s, t)] for s in SIDES for t in tt},
            }
            sh = self._shares(X, S, F)
            R = self._reach(n, X, S, F, u, sh, offset, B)
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
                A_o = {a: A[(o, a)] for a in tt}
                for t in tt:
                    terms = self._terms(R_o, sig, A_o, wd, o, t)
                    rate = sum(c * terms[k] for k, c in self.coef[(s, t)].items() if c)
                    rate = np.where(A[(s, t)] > 0, np.minimum(rate, 0), 0.0)
                    h0 = X[:, L.idx("health", s, t)]
                    h1 = np.maximum(h0 + DT * rate, 0)
                    if self.alive_eq is None:
                        ratio = np.divide(h1, h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                        p, pf = self.p[(s, t)]
                        a1 = A[(s, t)] * ratio ** (p + pf * focus)
                    else:
                        an, c = self.alive_eq[(s, t)]
                        Rr = {a: R_o[a] * (1 - wd) if o == "blue" else R_o[a] for a in tt}
                        Fa = self._alive_feats(
                            rate,
                            A[(s, t)],
                            h0,
                            focus,
                            sum(A_o[a] * sig[(a, t)] for a in tt),
                            sum(Rr[a] * sig[(a, t)] for a in tt),
                            s,
                        )
                        da = sum(ci * Fa[k] for k, ci in zip(an, c, strict=True) if ci)
                        a1 = np.clip(A[(s, t)] + DT * np.minimum(da, 0), 0, A[(s, t)])
                        a1 = np.where(h1 <= 1e-9, 0.0, a1)
                    newX[:, L.idx("health", s, t)] = h1
                    newX[:, L.idx("alive", s, t)] = a1
            TX = L.team_view(X)
            reach_team = np.column_stack([sum(R[("blue", a)] for a in tt), sum(R[("red", a)] for a in tt)])
            Fd = self._dist_library(TX, u, S, reach_team)
            dd = np.column_stack([Fd[k] for k in self.dist_names]) @ self.dist_coef
            newX[:, -1] = np.maximum(X[:, -1] + DT * dd, 0)
            X = newX
            out.append(X.copy())
        return np.stack(out, axis=1)


def fmt(c: dict) -> str:
    return " ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v) or "0"


def load(scen, split, nb=None):
    d = pd.read_parquet(DATA_DIR / "processed" / SCEN[scen] / f"{split}.parquet")
    if nb:
        d = d.loc[d.index.get_level_values(0).unique()[:nb]]
    return add_controls(d)


def boot(x, grp, rng, n=1000):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = [
        (w * sums).sum() / (w * cnt).sum()
        for w in (np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(n))
    ]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


VARIANTS = {
    "health": (True, False, False),
    "alive": (False, True, False),
    "distance": (False, False, True),
    "all": (True, True, True),
}


def main(nb=None):
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pickle.load(open(FROZEN, "rb"))
    rng = np.random.default_rng(0)
    log, rows_out, models = [], [], {}
    for scen in SCEN:
        t0 = time.time()
        m = frozen[scen]
        rows = prepare(load(scen, "train", nb), m.L)
        models[scen] = {"frozen": ShapeModel.from_frozen(m)}
        for v, flags in VARIANTS.items():
            log.append(f"\n=== {scen}, variant '{v}'")
            models[scen][v] = ShapeModel.from_frozen(m).fit_new(rows, *flags, log)
        print("\n".join(log[-20:]))
        print(f"{scen} fitted in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        sv = load(scen, "val", nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = m.L.batch(sv, horizon=round(max(HORIZONS) / DT))
        TY, ep = B["team"]["Y"], B["ep"]
        Ys = {}
        for v, mod in models[scen].items():
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                Ys[v] = m.L.team_view(mod.rollout(B))
        for v, Y in Ys.items():
            for h in HORIZONS:
                n = round(h / DT)
                ok = ~np.isnan(TY[:, n, 0])
                real = TY[ok, n, 0] - TY[ok, n, 1] - (TY[ok, 0, 0] - TY[ok, 0, 1])

                def errs(Y):
                    pred = Y[ok, n, 0] - Y[ok, n, 1] - (Y[ok, 0, 0] - Y[ok, 0, 1])
                    return (
                        np.abs(Y[ok, n, :2] - TY[ok, n, :2]).mean(1),
                        np.abs(pred - real),
                        10 * np.abs(Y[ok, n, 2:4] - TY[ok, n, 2:4]).mean(1),
                        np.abs(Y[ok, n, 4] - TY[ok, n, 4]),
                        pred,
                    )

                he, me, ae, de, pred = errs(Y)
                hf, mf, af, df, _ = errs(Ys["frozen"])
                row = {
                    "scenario": scen,
                    "variant": v,
                    "h": h,
                    "starts": int(ok.sum()),
                    "health error": he.mean(),
                    "margin R²": 1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum(),
                    "alive error (units)": ae.mean(),
                    "distance error": de.mean(),
                }
                if v != "frozen":
                    for lab, a_, b_ in (
                        ("health", he, hf),
                        ("margin", me, mf),
                        ("alive", ae, af),
                        ("distance", de, df),
                    ):
                        lo, hi = boot(a_ - b_, ep[ok], rng)
                        row[f"{lab} − frozen 95%"] = f"[{lo:+.4f}, {hi:+.4f}]"
                rows_out.append(row)
        print(f"{scen} rolled out in {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(rows_out)
    with pd.option_context("display.width", 260):
        print(t.round(4).to_string(index=False))
    if nb is None:
        (OUT / "terms.txt").write_text("\n".join(log) + "\n")
        json.dump(rows_out, open(OUT / "val.json", "w"), indent=1, default=float)
        pickle.dump(
            {
                s: {v: (m.coef, m.alive_eq, m.dist_names, m.dist_coef) for v, m in d.items()}
                for s, d in models.items()
            },
            open(OUT / "models.pkl", "wb"),
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
