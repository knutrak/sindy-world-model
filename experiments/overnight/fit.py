# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Pooled fit of `models.param.Param` (docs/overnight/00_plan.md). Rows come from prep.py (train battles only)."""

import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.special import ndtr

from sindy_wm.models import pipeline as PL
from sindy_wm.models.param import KW, Param, fire, form_features, margin_features, shots
from sindy_wm.models.pipeline import OTHER, SIDES, Pipeline
from sindy_wm.paths import DATA_DIR
from sindy_wm.states.typed import unit_stats

CACHE = DATA_DIR / "processed" / "overnight"
VARIANTS = ("DM", "DP", "CM", "CP", "HM", "A1", "SM", "SP", "R4", "R3")
_mem = {}


def army(name):
    if name not in _mem:
        _mem[name] = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    return _mem[name]


def pool_of(names, big=40_000, small=8_000):
    return {n: (small if n in VARIANTS else big) for n in names}


def geometry(types):
    m = Pipeline(tuple(types), withdraw_radius=7.0, **KW)
    return (
        5.75 / float(np.mean(list(m.rho.values()))),
        float(np.mean([unit_stats(t)["speed"] for t in types])) / 3.15,
        m,
    )


def fit(pool: dict, opt: dict) -> dict:
    par = {"opt": opt, "eta": {}, "p": {}, "share": {}, "over": {}}
    data = []
    for n, k in pool.items():
        d = army(n)
        data.append((d["rows"].head(k), d["types"], d["wr"], *geometry(d["types"])))
    over = opt.get("over")
    # --- health: one number per side, plus the overkill option's shared numbers
    H = {s: [] for s in SIDES}
    for r, types, _, _, _, m in data:
        tt, wd = m.L.types, r["withdraw"].to_numpy(float)
        for s in SIDES:
            o = OTHER[s]
            sig = m._sigma_measured(r, o)
            tot_s = sum(r[f"alive_{s}_{t}"] for t in tt).to_numpy()
            R = {a: r[f"reach_{o}_{a}"].to_numpy() * ((1 - wd) if o == "blue" else 1.0) for a in tt}
            ratio = np.divide(sum(R.values()), tot_s, out=np.zeros_like(tot_s), where=tot_s > 1e-9)
            for t in tt:
                H[s].append(
                    {
                        "t": t,
                        "tt": tt,
                        "R": {a: R[a] * sig[(a, t)] for a in tt},
                        "A": {a: r[f"alive_{o}_{a}"].to_numpy() for a in tt},
                        "ratio": ratio,
                        "y": r[f"d_health_{s}_{t}"].to_numpy(),
                        "h": r[f"health_{s}_{t}"].to_numpy(),
                        "al": r[f"alive_{s}_{t}"].to_numpy(),
                        "da": r[f"d_alive_{s}_{t}"].to_numpy(),
                        "fo": r["focus"].to_numpy(float),
                    }
                )

    def xy(s, ov):
        X = []
        for b in H[s]:
            x = sum(fire(a, b["t"], opt, {"over": ov}) * b["R"][a] for a in b["tt"])
            if over in ("sat", "both"):
                x = x / (1 + ov["k"] * b["ratio"])
            X.append(x)
        return X

    keys = {"shots": ["c"], "power": ["g"], "sat": ["k"], "both": ["c", "k"]}.get(over, [])
    if keys:

        def res(q):
            ov = dict(zip(keys, q, strict=True))
            e = []
            for s in SIDES:
                X = np.concatenate(xy(s, ov))
                Y = np.concatenate([b["y"] for b in H[s]])
                e.append(Y - X * (X @ Y) / (X @ X))
            return np.concatenate(e)[::7] * 10

        lo = {"c": 0.0, "g": 0.3, "k": 0.0}
        hi = {"c": 10.0, "g": 1.5, "k": 5.0}
        st = {"c": 0.5, "g": 0.9, "k": 0.1}
        f = least_squares(
            res, [st[k] for k in keys], bounds=([lo[k] for k in keys], [hi[k] for k in keys]), diff_step=1e-2
        )
        par["over"] = dict(zip(keys, map(float, f.x), strict=True))
    extra = opt.get("alive", ())
    for s in SIDES:
        X = xy(s, par["over"])
        Y = np.concatenate([b["y"] for b in H[s]])
        xx = np.concatenate(X)
        par["eta"][s] = float(-(xx @ Y) / (xx @ xx))
        Z, DA = [], []
        for x, b in zip(X, H[s], strict=True):
            ok = (b["h"] > 0.03) & (b["al"] > 0)
            base = -par["eta"][s] * x * b["al"] / np.maximum(b["h"], 1e-9)
            cols = {"p": base}
            if s == "red":
                cols["focus"] = base * b["fo"]
            if "shots" in extra:
                wt = sum(b["A"].values())
                z = sum(b["A"][a] * 7.5 / shots(a, b["t"]) for a in b["tt"])
                cols["shots"] = base * np.divide(z, wt, out=np.ones_like(wt), where=wt > 1e-9)
            if "hpu" in extra:
                cols["hpu"] = base * np.clip(
                    np.divide(b["h"], b["al"], out=np.ones_like(base), where=b["al"] > 0), 0, 1
                )
            if "n" in extra:
                cols["n"] = base * b["al"]
            names = list(cols)
            Z.append(np.column_stack(list(cols.values()))[ok])
            DA.append(b["da"][ok])
        c, *_ = np.linalg.lstsq(np.vstack(Z), np.concatenate(DA), rcond=None)
        par["p"][s] = dict(zip(names, map(float, c), strict=True))
    # --- reach closure
    for s in SIDES:
        o, Q, y = OTHER[s], [], []
        for r, _, _, _, _, m in data:
            for t in m.L.types:
                q = r[r[f"alive_{s}_{t}"] > 0]
                k = 5.75 / m.rho[t] if opt.get("share") == "scaled" else 1.0
                nopp = 10 * sum(q[f"alive_{o}_{a}"] for a in m.L.types)
                Q.append(
                    np.column_stack(
                        [
                            k * (q["distance"] - q[f"front_{s}_{t}"].fillna(0)),
                            k * q[f"depth_{s}"],
                            k * q[f"depth_{o}"],
                            k * q[f"width_{s}"],
                            k * q[f"width_{o}"],
                            nopp,
                            np.full(len(q), k * m.rho[t]),
                        ]
                    )
                )
                y.append((q[f"reach_{s}_{t}"] / q[f"alive_{s}_{t}"]).to_numpy())
        Q, y = np.vstack(Q), np.concatenate(y)
        i = np.random.default_rng(0).choice(len(y), min(100_000, len(y)), replace=False)
        par["share"][s], _ = curve_fit(
            lambda X, la, be, g: PL.derived_share(*X[:6], la, be, g, rho=X[6]),
            Q[i].T,
            y[i],
            p0=(1.0, 1.0, 1.0),
            maxfev=20000,
        )
    # --- distance, in scaled units
    Ts = []
    for r, _, wr, ds, vs, _ in data:
        T = r[
            [
                "movement",
                "distance",
                *PL.SHAPE,
                "d_distance",
                *PL.CTRL,
                *[f"{k}_{s}" for k in ("health", "alive", "reach") for s in SIDES],
            ]
        ].copy()
        T[["distance", *PL.SHAPE]] *= ds
        T["d_distance"] /= vs
        T["wr"] = wr * ds
        Ts.append(T)
    T = pd.concat(Ts, ignore_index=True)
    P = {"gates": PL.fit_gates(T, 1.0, skip=("withdraw",)), "form": "derived"}
    P["gates"]["withdraw"] = np.array([0, 99, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], float)
    for s in SIDES:
        a, o = T[T[f"alive_{s}"] > 0], OTHER[s]
        a = a.sample(min(100_000, len(a)), random_state=0)
        Q = tuple(
            a[c].to_numpy() for c in ("distance", f"depth_{s}", f"depth_{o}", f"width_{s}", f"width_{o}")
        ) + (10 * a[f"alive_{o}"].to_numpy(),)
        q, _ = curve_fit(
            lambda X, *p: PL.derived_share(*X, *p, rho=5.75),
            Q,
            (a[f"reach_{s}"] / a[f"alive_{s}"]).to_numpy(),
            p0=(1.0, 1.0, 1.0),
            maxfev=20000,
        )
        P[f"share_{s}"] = np.append(q, 5.75)
    parts, yd = [], []
    for wr, g in T.groupby("wr"):
        m = Pipeline(("MARINE",), withdraw_radius=wr, **KW)
        m.P_team = P
        F = m._dist_library(
            g[["health_blue", "health_red", "alive_blue", "alive_red", "distance"]].to_numpy(),
            g[PL.CTRL].to_numpy(float),
            g[PL.SHAPE].to_numpy(),
            g[["reach_blue", "reach_red"]].to_numpy(),
        )
        N = list(F)
        parts.append(np.column_stack(list(F.values())))
        yd.append(g["d_distance"].to_numpy())
    cols = [
        j for j in PL._family_cols(N, PL.MOVE_FAMILY) if N[j] not in (*PL.DIST_EXCLUDE, "close*wd", "flee*wd")
    ] + [N.index("danger*wd"), N.index("walk_r*wd")]
    cd = np.zeros(len(N))
    cd[cols] = PL.stlsq(np.vstack(parts)[:, cols], np.concatenate(yd), KW["thr_distance"])
    keep = [j for j in range(len(N)) if cd[j]]
    par["dist_names"], par["dist_coef"], par["P_team"] = [N[j] for j in keep], cd[keep], P
    # --- formation
    if opt.get("form"):
        par["form"] = {}
        for i, s in enumerate(SIDES):
            o, TH, YD, YW = OTHER[s], [], [], []
            for r, _, _, ds, vs, m in data:
                ps, po = (
                    np.divide(
                        r[f"reach_{q}"], r[f"alive_{q}"], out=np.zeros(len(r)), where=r[f"alive_{q}"] > 0
                    )
                    for q in (s, o)
                )
                dl = np.divide(
                    sum(r[f"d_alive_{s}_{t}"] for t in m.L.types),
                    r[f"alive_{s}"],
                    out=np.zeros(len(r)),
                    where=r[f"alive_{s}"] > 0,
                )
                f = form_features(
                    r[f"depth_{s}"].to_numpy() * ds,
                    r[f"width_{s}"].to_numpy() * ds,
                    1 - ps,
                    1 - po,
                    dl,
                    r[PL.CTRL].to_numpy(float),
                    s == "blue",
                )
                TH.append(np.column_stack(list(f.values())))
                YD.append(r[f"d_depth_{s}"].to_numpy() * ds / vs)
                YW.append(r[f"d_width_{s}"].to_numpy() * ds / vs)
            for kind, Y in (("dep", YD), ("wid", YW)):
                c = PL.stlsq(np.vstack(TH), np.concatenate(Y), opt.get("thr_form", 0.1))
                par["form"][(s, kind)] = {n: float(v) for n, v in zip(f, c, strict=True) if v}
    # --- margins
    if "marg" in opt:
        par["marg"] = {}
        M, SD, SH = [], [], []
        for r, _, _, _, _, m in data:
            for s in SIDES:
                for t in m.L.types:
                    q = r[r[f"alive_{s}_{t}"] > 0]
                    M.append(q[f"mu_{s}_{t}"] * 5.75 / m.rho[t])
                    SD.append(q[f"sd_{s}_{t}"] * 5.75 / m.rho[t])
                    SH.append(q[f"reach_{s}_{t}"] / q[f"alive_{s}_{t}"])
        M, SD, SH = (np.concatenate(v) for v in (M, SD, SH))
        i = np.random.default_rng(0).choice(len(M), min(100_000, len(M)), replace=False)
        ck, _ = curve_fit(
            lambda X, c, k: ndtr((c - X[0]) / np.maximum(k * X[1], 0.05)),
            (M[i], SD[i]),
            SH[i],
            p0=(0.0, 1.0),
            maxfev=20000,
        )
        par["marg"]["ck"] = tuple(map(float, ck))
        for s in SIDES:
            o, TH, YM, YS = OTHER[s], [], [], []
            for r, _, _, _, vs, m in data:
                L = m.L.types
                po = np.divide(
                    r[f"reach_{o}"], r[f"alive_{o}"], out=np.zeros(len(r)), where=r[f"alive_{o}"] > 0
                )
                dlo = np.divide(
                    sum(r[f"d_alive_{o}_{t}"] for t in L),
                    r[f"alive_{o}"],
                    out=np.zeros(len(r)),
                    where=r[f"alive_{o}"] > 0,
                )
                for t in L:
                    ok = (r[f"alive_{s}_{t}"] > 0).to_numpy()
                    ds = 5.75 / m.rho[t]
                    q = r[ok]
                    p = (q[f"reach_{s}_{t}"] / q[f"alive_{s}_{t}"]).to_numpy()
                    f = margin_features(
                        q[f"mu_{s}_{t}"].to_numpy() * ds,
                        q[f"sd_{s}_{t}"].to_numpy() * ds,
                        p,
                        po[ok],
                        q["d_distance"].to_numpy() * ds,
                        (q[f"d_alive_{s}_{t}"] / q[f"alive_{s}_{t}"]).to_numpy(),
                        dlo[ok],
                        q[PL.CTRL].to_numpy(float),
                        s == "blue",
                        vs,
                    )
                    TH.append(np.column_stack(list(f.values())))
                    YM.append(q[f"d_mu_{s}_{t}"].to_numpy() * ds)
                    YS.append(q[f"d_sd_{s}_{t}"].to_numpy() * ds)
            names = list(f)
            for kind, Y in (("mu", YM), ("sd", YS)):
                thr = opt.get("thr_marg", 0.1)
                for _ in range(12):  # raise the threshold until at most 8 terms remain (plan, block B)
                    c = PL.stlsq(np.vstack(TH), np.concatenate(Y), thr)
                    if (c != 0).sum() <= 8:
                        break
                    thr *= 1.25
                par["marg"][(s, kind)] = {n: float(v) for n, v in zip(names, c, strict=True) if v}
    return par


def model(par, name) -> Param:
    d = army(name)
    return Param(d["types"], d["wr"], par)


def complexity(par) -> int:
    n = 2 + len(par["over"]) + sum(len(v) for v in par["p"].values()) + len(par["dist_coef"])
    n += sum(len(v) for v in par.get("form", {}).values()) + sum(
        len(v) for k, v in par.get("marg", {}).items() if k != "ck"
    )
    return n
