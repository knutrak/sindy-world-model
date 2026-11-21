"""The pipeline refitted with health compartments as candidates in every equation (docs/planning/15).

`DiscoveredPipeline.fit_discovered` first runs the ordinary `Pipeline.fit` (closures, gates, reach shares,
target
rule: their inputs do not involve the compartments), then refits

  health      the pipeline's candidates plus, per attacker type a, R·s·w_att and R·s·w_tgt (attackers' and
              targets' wounded shares), each times the controls 1, focus, hold, wd;
  distance    the pipeline's movement library plus w_blue and w_red (team wounded shares) times the controls;
  alive       the four compartment laws of docs/planning/13 (I, K_W, K_F, sigma), fitted on step rates
  (smoothed
              per-step event counts) with d-hat from the refitted health equation (the A2
              errors-in-variables fix),

all with the two-level STLSQ (`two_level_stlsq`): state groups at the pipeline's thresholds, control
interactions
inside a kept group at half of it. The rollout is `Pipeline.rollout` with `compart_model` set; the wounded
shares
reach `_terms` and `_dist_library` through `self._ctx`, set per step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sindy_wm.models.intent_model import CTRL, DT, MOVE_FAMILY, SHAPE, _family_cols
from sindy_wm.models.pipeline import (
    COMPART_ALWAYS,
    DIST_EXCLUDE,
    OTHER,
    SIDES,
    Pipeline,
    compart_derived,
    compart_features,
    team_rows,
    two_level_stlsq,
)

CTRLS = ("1", "focus", "hold", "wd")
THR_HEALTH, THR_LAWS = 0.15, 0.05


def _psi(U) -> dict:
    return {"1": np.ones(len(U)), "focus": U[:, 3], "hold": U[:, 1], "wd": U[:, 2]}


def _group(name: str) -> tuple[str, bool]:
    """(group, is base) of a candidate name: "state|control" for the new candidates, otherwise a single
    group."""
    if "|" in name:
        g, c = name.split("|")
        return g, c == "1"
    return name, True


class DiscoveredPipeline(Pipeline):
    _ctx = None

    # ---- candidates ----
    def _terms(self, R_o, sig, A_o, wd, o, t):
        L = super()._terms(R_o, sig, A_o, wd, o, t)
        ctx = self._ctx
        if ctx is None:
            return L
        s = OTHER[o]
        psi = _psi(ctx["U"])
        blue_rule = o == "blue" and self.withdraw == "rule"
        for a in self.L.types:
            R = R_o[a] * (1 - wd) if blue_rule else R_o[a]
            base = R * sig[(a, t)]
            for c, v in psi.items():
                L[f"R[{a}]*s[{t}]*w_att|{c}"] = base * ctx["w"][(o, a)] * v
                L[f"R[{a}]*s[{t}]*w_tgt|{c}"] = base * ctx["w"][(s, t)] * v
        return L

    def _dist_library(self, X, U, S, reach):
        F = super()._dist_library(X, U, S, reach)
        ctx = self._ctx
        if ctx is None:
            return F
        for c, v in _psi(U).items():
            F[f"w_blue|{c}"] = ctx["team"]["blue"] * v
            F[f"w_red|{c}"] = ctx["team"]["red"] * v
        return F

    # ---- fitting ----
    def _rows_context(self, rows: pd.DataFrame) -> dict:
        w, team = {}, {}
        for side in SIDES:
            Ws = As = 0
            for t in self.L.types:
                A = rows[f"alive_{side}_{t}"].to_numpy(float)
                Wt = rows[f"W_{side}_{t}"].to_numpy(float)
                w[(side, t)] = np.divide(Wt, A, out=np.zeros_like(A), where=A > 1e-9)
                Ws, As = Ws + Wt, As + A
            team[side] = np.divide(Ws, As, out=np.zeros_like(As), where=As > 1e-9)
        return {"w": w, "team": team, "U": rows[CTRL].to_numpy(float)}

    def health_rate_rows(self, rows: pd.DataFrame, s: str, t: str) -> np.ndarray:
        """The (refitted) health equation's rate on measured reach (`self._ctx` must be set from `rows`)."""
        o, tt = OTHER[s], self.L.types
        R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in tt}
        A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in tt}
        L = self._terms(R_o, self._sigma_measured(rows, o), A_o, rows["withdraw"].to_numpy(), o, t)
        return sum(c * L[k] for k, c in self.coef[(s, t)].items() if c)

    def fit_discovered(
        self, rows: pd.DataFrame, events: pd.DataFrame, use_hw: bool = True
    ) -> DiscoveredPipeline:
        """rows: the pipeline's rate rows joined with the compartment states (W_, HW_, h0_ per side and type);
        events: per-step events (`states.compartments.compartment_events`) on the same index."""
        self._ctx = None
        self.fit(rows)
        self._ctx = self._rows_context(rows)
        self.selection = {}
        wd = rows["withdraw"].to_numpy()
        # health
        for s in SIDES:
            o = OTHER[s]
            R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in self.L.types}
            A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in self.L.types}
            sig = self._sigma_measured(rows, o)
            for t in self.L.types:
                L = self._terms(R_o, sig, A_o, wd, o, t)
                names = list(L)
                g = [_group(n) for n in names]
                c = two_level_stlsq(
                    np.column_stack([L[n] for n in names]),
                    rows[f"d_health_{s}_{t}"].to_numpy(),
                    [x[0] for x in g],
                    THR_HEALTH,
                    [x[1] for x in g],
                )
                self.coef[(s, t)] = dict(zip(names, c, strict=True))
                self.selection[("health", s, t)] = sorted(
                    {_group(n)[0] for n, v in zip(names, c, strict=True) if v and "|" in n}
                )
        # distance
        T = team_rows(rows, self.L)
        Xt = np.column_stack(
            [T[c].to_numpy() for c in ("health_blue", "health_red", "alive_blue", "alive_red", "distance")]
        )
        F = self._dist_library(
            Xt, T[CTRL].to_numpy(), T[SHAPE].to_numpy(), T[["reach_blue", "reach_red"]].to_numpy()
        )
        names = list(F)
        cols = [j for j in _family_cols(names, MOVE_FAMILY) if names[j] not in DIST_EXCLUDE]
        if self.movement_wd == "rule":
            cols = [j for j in cols if names[j] not in ("close*wd", "flee*wd")]
            cols += [names.index("danger*wd"), names.index("walk_r*wd")]
        cols += [j for j, n in enumerate(names) if n.startswith(("w_blue|", "w_red|"))]
        g = [_group(names[j]) for j in cols]
        cd = np.zeros(len(names))
        cd[cols] = two_level_stlsq(
            np.column_stack([F[names[j]] for j in cols]),
            T["d_distance"].to_numpy(),
            [x[0] for x in g],
            0.05,
            [x[1] for x in g],
        )
        self.dist_names, self.dist_coef = names, cd
        self.selection[("distance",)] = sorted(
            {_group(n)[0] for n, v in zip(names, cd, strict=True) if v and "|" in n}
        )
        # alive: the compartment laws on step rates
        laws = {}
        self.law_r2 = {}
        for s in SIDES:
            for t in self.L.types:
                A10 = rows[f"alive_{s}_{t}"].to_numpy(float)
                H10 = rows[f"health_{s}_{t}"].to_numpy(float)
                A, Wn, w, mF, mW = compart_derived(
                    A10,
                    rows[f"W_{s}_{t}"].to_numpy(float),
                    H10,
                    rows[f"HW_{s}_{t}"].to_numpy(float),
                    rows[f"h0_{s}_{t}"].to_numpy(float),
                )
                d = np.maximum(-self.health_rate_rows(rows, s, t), 0) * 10
                rho = np.divide(
                    rows[f"reach_{s}_{t}"].to_numpy(), A10, out=np.zeros_like(A10), where=A10 > 1e-9
                )
                rate = {
                    k: smooth_rate(events[f"{k}_{s}_{t}"], rows.index) for k in ("I", "KW", "KF", "D", "DW")
                }
                sigma = np.divide(rate["DW"], rate["D"], out=np.zeros_like(rate["D"]), where=rate["D"] > 0.05)
                ok = (A10 > 0) & np.isfinite(rate["I"])
                U = rows[CTRL].to_numpy(float)
                L = {}
                for law, y in (("I", rate["I"]), ("KW", rate["KW"]), ("KF", rate["KF"]), ("sigma", sigma)):
                    m = ok & ((rate["D"] > 0.05) if law == "sigma" else True)
                    Fl = compart_features(law, d[m], w[m], Wn[m], mF[m], mW[m], rho[m], U[m], use_hw)
                    nm = list(Fl)
                    gg = [_group(n) for n in nm]
                    X = np.column_stack([Fl[n] for n in nm])
                    c = two_level_stlsq(
                        X,
                        y[m],
                        [x[0] for x in gg],
                        THR_LAWS,
                        [x[1] for x in gg],
                        always=(COMPART_ALWAYS[law],),
                    )
                    L[law] = ([n for n, v in zip(nm, c, strict=True) if v], c[c != 0])
                    pred = X @ c
                    self.law_r2[(law, s, t)] = float(
                        1 - ((y[m] - pred) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
                    )
                laws[(s, t)] = L
        self.compart_model = {"laws": laws, "use_hw": use_hw}
        self._ctx = None
        return self


def smooth_rate(counts: pd.Series, index: pd.MultiIndex) -> np.ndarray:
    """Per-step event counts smoothed per battle (Savitzky-Golay, window 9, order 2, as the rates), per
    second."""
    out = pd.Series(np.nan, index=counts.index)
    for _, g in counts.groupby(level=0, sort=False):
        v = g.to_numpy(float)
        if len(v) >= 9:
            out.loc[g.index] = savgol_filter(np.nan_to_num(v), 9, 2, mode="interp") / DT
    return out.reindex(index).to_numpy()
