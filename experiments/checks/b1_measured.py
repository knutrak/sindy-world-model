# ruff: noqa: E402, E501, B023
"""Where B1's error sits: B1 rolled out with its own closures, the measured share in reach, the measured alive, or both
(val, 300 battles per army). Exploratory check, docs/overnight/04_models_and_notes.md. Run from the repository root:
uv run --no-sync python experiments/checks/b1_measured.py [n_battles]"""

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
for p in ("experiments/overnight",):
    sys.path.insert(0, str(ROOT / p))
import numpy as np
import pandas as pd
import prep
import search as Se

from sindy_wm.models.intent_model import add_controls
from sindy_wm.models.param import DT, OTHER, Param
from sindy_wm.models.pipeline import SIDES, Layout

ARMIES = {
    "marines": ("exp06_intents", ("MARINE",), 7.0),
    "marauders": ("exp09_marauders", ("MARAUDER",), 7 * 7.125 / 5.75),
    "stalkers": ("exp07_stalkers", ("STALKER",), 7 * 7.25 / 5.75),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER"), 8.0),
    "triple": ("exp12_triple", ("MARINE", "MARAUDER", "STALKER"), 8.0),
}
MODES = {
    "own closures": (False, False),
    "measured share": (True, False),
    "measured alive": (False, True),
    "both": (True, True),
}


def carry(M):
    M = M.copy()
    for n in range(1, M.shape[1]):
        bad = np.isnan(M[:, n])
        M[:, n][bad] = M[:, n - 1][bad]
    return M


def reach_traj(L, st, horizon=64, every=8, history=8, min_future=8):
    """Units in reach per (side, type) along each start's window, the same start points as Layout.batch."""
    out = []
    for _, g in st.groupby(level=0, sort=False):
        R = g[L.r_cols].to_numpy(float)
        for k0 in range(history, len(R) - min_future, every):
            m = min(horizon + 1, len(R) - k0)
            y = np.full((horizon + 1, R.shape[1]), np.nan)
            y[:m] = R[k0 : k0 + m]
            out.append(y)
    return np.array(out)


class Measured(Param):
    """B1's rollout (opt = {}) with the share in reach and / or alive taken from the battle instead of the model."""

    def __init__(self, *a, share=False, alive=False, **k):
        super().__init__(*a, **k)
        assert not self.opt, "B1 only"
        self.m_share, self.m_alive = share, alive

    def rollout(self, B):
        tt, L, par = self.L.types, self.L, self.par
        X, S, F, U = (
            B["X0"].astype(float).copy(),
            np.nan_to_num(B["shape0"].astype(float)),
            B["front0"],
            B["U"],
        )
        Yc, Rc = carry(B["Y"]), carry(B["Yreach"])
        sh0 = self._share(X, S, F)
        off = {}
        for key in sh0:
            A0 = X[:, L.idx("alive", *key)]
            meas = np.divide(B["reach0"][:, L.pos(*key)], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
            off[key] = np.where(A0 > 1e-9, meas - sh0[key], 0.0)
        out = [X.copy()]
        for n in range(U.shape[1]):
            u = U[:, n]
            wd, focus = u[:, 2], u[:, 3]
            A = {(s, t): X[:, L.idx("alive", s, t)] for s in SIDES for t in tt}
            sh = self._share(X, S, F)
            p = {key: np.clip(sh[key] + off[key] * np.exp(-n * DT), 0, 1) for key in sh}
            if self.m_share:
                for key in p:
                    Am = Yc[:, n, L.idx("alive", *key)]
                    p[key] = np.clip(
                        np.divide(Rc[:, n, L.pos(*key)], Am, out=np.zeros_like(Am), where=Am > 1e-9), 0, 1
                    )
            R = {key: A[key] * p[key] for key in p}
            newX = X.copy()
            for s in SIDES:
                o = OTHER[s]
                if len(tt) == 1:
                    sig = {tt[0]: np.ones(len(X))}
                else:
                    tot = sum(A[(s, t)] for t in tt)
                    sig = {t: np.divide(A[(s, t)], tot, out=np.zeros_like(tot), where=tot > 0) for t in tt}
                Ro = {a: R[(o, a)] * ((1 - wd) if o == "blue" else 1.0) for a in tt}
                for t in tt:
                    r_tot = sum(self.fire[(a, t)] * Ro[a] for a in tt)
                    rt = np.minimum(-par["eta"][s] * r_tot * sig[t], 0)
                    h0, a0 = X[:, L.idx("health", s, t)], A[(s, t)]
                    rt = np.where(a0 > 0, rt, 0.0)
                    h1 = np.maximum(h0 + DT * rt, 0)
                    ratio = np.divide(h1, h0, out=np.zeros_like(h1), where=h0 > 1e-9)
                    e = par["p"][s]
                    expo = e["p"] + e.get("focus", 0.0) * focus
                    a1 = a0 * ratio ** np.clip(expo, 0.05, 3)
                    if self.m_alive:
                        a1 = Yc[:, n + 1, L.idx("alive", s, t)]
                    newX[:, L.idx("health", s, t)], newX[:, L.idx("alive", s, t)] = h1, a1
            TX = L.team_view(X)
            rt = np.column_stack([sum(R[(s, a)] for a in tt) for s in SIDES])
            Fd = self._dist_library(TX, u, S, rt)
            dd = np.column_stack([Fd[j] for j in self.dist_names]) @ self.dist_coef
            newX[:, -1] = np.maximum(X[:, -1] + DT * dd, 0)
            X = newX
            out.append(X.copy())
        return np.stack(out, 1)


def r2(x, h):
    ok = np.isfinite(x[f"m{h}"]) & np.isfinite(x[f"v{h}"])
    return 1 - x[f"m{h}"][ok].sum() / ((x[f"v{h}"][ok] - x[f"v{h}"][ok].mean()) ** 2).sum()


def main(nb):
    par = pickle.load(open(prep.PROC / "overnight" / "finalists.pkl", "rb"))[("B1", "all")]
    rows = []
    for name, (exp, types, wr) in ARMIES.items():
        st = pd.read_parquet(prep.PROC / f"{exp}_typed" / "val.parquet")
        ids = st.index.get_level_values(0).unique()[:nb]
        st = st.loc[ids].copy()
        st["vd"] = st.groupby(level=0)["distance"].diff().fillna(0.0) / 0.125
        st = add_controls(st)
        L = Layout(tuple(t.lower() for t in types))
        B = prep.batch(L, st)
        B["Yreach"] = reach_traj(L, st)
        assert B["Yreach"].shape[:2] == B["Y"].shape[:2]
        with np.errstate(all="ignore"):  # the copied rollout must reproduce B1 exactly
            Bs = {**B, "U": B["U"][:, :32]}
            same = np.allclose(
                Param(types, wr, par).rollout(Bs), Measured(types, wr, par).rollout(Bs), equal_nan=True
            )
        assert same, "copied rollout differs from Param.rollout"
        for mode, (s, a) in MODES.items():
            x = Se.errors(Measured(types, wr, par, share=s, alive=a), B)
            rows.append(
                {
                    "army": name,
                    "mode": mode,
                    "starts": len(B["X0"]),
                    "health 1s": np.nanmean(x["e1"]),
                    "health 2s": np.nanmean(x["e2"]),
                    "health 4s": np.nanmean(x["e4"]),
                    "margin R2 2s": r2(x, 2),
                    "margin R2 4s": r2(x, 4),
                    "distance 2s": np.nanmean(x["d2"]),
                    "alive 2s": np.nanmean(x["a2"]),
                }
            )
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    print(t.round(4).to_string())
    t.to_csv(ROOT / "docs" / "checks" / "logs" / "b1_measured.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
