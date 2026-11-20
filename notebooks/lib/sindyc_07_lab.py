"""Helpers for notebooks/sindyc_07: start-point batches, a lean simulator of `model_derived`
with every number exposed as a parameter, rollout metrics, paired bootstrap, and fitting on
rollout error (least squares on multi-step residuals).

Exploratory code. `simulate(base_params(), B)` reproduces `IntentModel.simulate(..., reach0=...)`
of model_derived exactly (checked in the notebook).
"""

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

from sindy_wm.models.intent_model import DT, _shape_library, add_controls, make_batch
from sindy_wm.paths import DATA_DIR

PROCESSED = DATA_DIR / "processed" / "exp06_intents"
SIDES = ("blue", "red")
H = (1, 2, 4, 8)
current = pickle.load(open(PROCESSED / "model_derived.pkl", "rb"))


def load_batch(name, experiment="exp06_intents", **kw):
    if experiment != "exp06_intents":
        df = pd.read_parquet(DATA_DIR / "processed" / experiment / f"{name}.parquet")
        B = make_batch(add_controls(df), **kw)
        B["group"] = B["ep"]
        return B
    if name == "test":
        parts = []
        for k, n in enumerate(sorted(p.stem for p in PROCESSED.glob("cf_*.parquet"))):
            d = pd.read_parquet(PROCESSED / f"{n}.parquet").reset_index()
            d["episode_idx"] += 10_000 * (k + 1)
            parts.append(d.set_index(["episode_idx", "step"]))
        df = pd.concat(parts)
    else:
        df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    B = make_batch(add_controls(df), **kw)
    B["group"] = B["ep"] % 10_000 if name == "test" else B["ep"]  # bootstrap unit: the battle (seed)
    return B


def scale_params(p, source="marine", target="stalker", rates="movement"):
    """p rewritten for another unit type without refitting, like `switch_library.scaled`.

    Distances scale with reach, firepower with 1 / (time to kill), closing speeds with walking
    speed; exponents and the closure's lam, beta, g are unit-free. The share-state rates and the
    offset decay times are per second: `rates` picks their time scale: "movement" (reach /
    speed), "combat" (time to kill) or "none" (left as fitted).
    """
    from sindy_wm.models.switch_library import UNITS

    a, b = UNITS[source], UNITS[target]
    s, ft, fv = b["reach"] / a["reach"], a["tau"] / b["tau"], b["speed"] / a["speed"]
    fr = {"movement": fv / s, "combat": ft, "none": 1.0}[rates]
    q = dict(p)
    for k in ("kb", "kr", "kr_wd"):
        q[k] = p[k] * ft
    for k in ("d_adv", "d_wd", "d_flee", "d_hold_far", "d_hold_contact"):
        q[k] = p[k] * fv
    q["rho"] = p.get("rho", current.P["share_blue"][3]) * s
    gates = {g: np.array(v, float) for g, v in p.get("gates", current.P["gates"]).items()}
    for v in gates.values():
        v[[1, 6]] *= s
    gates["withdraw"][[9, 10]] *= s
    q["gates"] = gates
    for k in p:
        if k.startswith(("sk_", "sf_", "so_")):
            q[k] = p[k] * fr
        if k.startswith("tau_"):
            q[k] = p[k] / fr
    return q


def base_params():
    c = current.coef
    n = current.names

    def g(eq, k):
        return float(c[eq][n.index(k)])

    return {
        "kb": g("health_blue", "Rr"),
        "kr": g("health_red", "Rb"),
        "kr_wd": g("health_red", "Rb*wd"),
        "d_adv": g("distance", "close*adv"),
        "d_wd": g("distance", "close*wd"),
        "d_flee": g("distance", "flee*wd"),
        "d_hold_far": g("distance", "close*hold*far"),
        "d_hold_contact": g("distance", "close*hold*contact"),
        "p_b": current.p["blue"]["1"],
        "p_r": current.p["red"]["1"],
        "p_r_focus": current.p["red"]["focus"],
        "tau_b": 1.0,
        "tau_r": 1.0,
        "lam_b": current.P["share_blue"][0],
        "beta_b": current.P["share_blue"][1],
        "g_b": current.P["share_blue"][2],
        "lam_r": current.P["share_red"][0],
        "beta_r": current.P["share_red"][1],
        "g_r": current.P["share_red"][2],
        # structural candidates, off by default
        "rho_wd_b": 0.0,
        "rho_wd_r": 0.0,
        "rho_hold_b": 0.0,
        "rho_hold_r": 0.0,
    }


def gate(g, d, S):
    db, dr, wb, wr = S.T
    _, c, bb, br, eb, er, w, a = g[:8]
    return ndtr((d - c - bb * db - br * dr - eb * wb - er * wr) / np.sqrt(w**2 + (a * (db + dr)) ** 2 + 1e-4))


SHARE_FN = None  # optional override: f(X, S, u, p) -> (N, 2)


def shares(X, S, u, p):
    if SHARE_FN is not None:
        return SHARE_FN(X, S, u, p)
    d, db, dr, wb, wr = X[:, 4], *S.T
    rho = p.get("rho", current.P["share_blue"][3])

    def one(own, opp, w_own, w_opp, n_opp, lam, beta, g, c, shift):
        m = np.sqrt(2 * np.log(np.maximum(n_opp, 1.0)))
        d_eff = np.sqrt(d**2 + lam * (w_own - w_opp) ** 2)
        front = beta * m * (opp if c is None else c * np.tanh(opp / c))
        return ndtr((rho + shift - d_eff + front) / np.sqrt(0.05**2 + (g * own) ** 2 + 1e-4))

    sb = one(
        db,
        dr,
        wb,
        wr,
        10 * X[:, 3],
        p["lam_b"],
        p["beta_b"],
        p["g_b"],
        p.get("c_b"),
        p["rho_wd_b"] * u[:, 2] + p["rho_hold_b"] * u[:, 1],
    )
    sr = one(
        dr,
        db,
        wr,
        wb,
        10 * X[:, 2],
        p["lam_r"],
        p["beta_r"],
        p["g_r"],
        p.get("c_r"),
        p["rho_wd_r"] * u[:, 2] + p["rho_hold_r"] * u[:, 1],
    )
    return np.column_stack([sb, sr])


def simulate(p, B, steps=None, truth=None, return_reach=False):
    """Lean copy of IntentModel.simulate for model_derived (shape held), with parameters p.

    truth: optional oracle inputs {"share": (N, T, 2), "distance": (N, T), "shape": (N, T, 4)}.

    Options carried in p:
      shape_coef   list of four coefficient vectors: integrate the formation equations
      c_b, c_r     saturation of the closure's enemy-front term (None = off)
      share_state  True: the share in reach is a state of its own, started at the measured share:
                   s' = sa·(closure)' + sk·(closure − s) + C·(1 − s)·(sf_adv·adv + sf_hold·hold + sf_wd·wd)
                        − so_wd·wd·s + sd·(1 − s)·(ln A)'
                   with C the chance that any Blue unit is in reach. sa = 1, sk = 1/tau and the
                   rest 0 is the baseline's decaying offset.
    """
    truth = truth or {}
    G = p.get("gates", current.P["gates"])
    X, S, U = B["X0"].astype(float).copy(), B["shape0"].astype(float).copy(), B["U"]
    steps = steps or U.shape[1]
    tau = np.array([p["tau_b"], p["tau_r"]])
    s0 = shares(X, S, U[:, 0], p)
    offset = B["reach0"] / np.maximum(X[:, 2:4], 1e-9) - s0
    out, reach_out = [X.copy()], []
    for n in range(steps):
        u = U[:, n]
        adv, hold, wd, focus = u.T
        if "shape" in truth:
            S = truth["shape"][:, n]
        elif p.get("shape_coef") is not None and n > 0:
            S = shape_step(X[:, 4], S, U[:, n - 1], p["shape_coef"])
        if "distance" in truth:
            X[:, 4] = truth["distance"][:, n]
        d = X[:, 4]
        if "share" in truth:
            s = truth["share"][:, n]
        elif p.get("share_state"):
            # share in reach as a state:
            # s' = alpha·(closure)' + k·(closure − s) + fan-out − withdraw-out + deaths
            clo = shares(X, S, u, p)
            if n == 0:
                s_state, clo_prev, A_prev = B["reach0"] / np.maximum(X[:, 2:4], 1e-9), clo, X[:, 2:4].copy()
            else:
                C = 1 - (1 - np.clip(s_state[:, :1], 0, 1)) ** (10 * X[:, 2:3])  # any Blue unit in reach
                dlnA = np.divide(X[:, 2:4] - A_prev, np.maximum(A_prev, 1e-9))
                ds = np.zeros_like(s_state)
                for j, t in enumerate("br"):
                    ds[:, j] = (
                        p[f"sa_{t}"] * (clo[:, j] - clo_prev[:, j])
                        + DT * p[f"sk_{t}"] * (clo[:, j] - s_state[:, j])
                        + DT
                        * C[:, 0]
                        * (1 - s_state[:, j])
                        * (p[f"sf_adv_{t}"] * adv + p[f"sf_hold_{t}"] * hold + p[f"sf_wd_{t}"] * wd)
                        - DT * p[f"so_wd_{t}"] * wd * s_state[:, j]
                        + p[f"sd_{t}"] * (1 - s_state[:, j]) * dlnA[:, j]
                    )
                s_state = np.clip(s_state + ds, 0, 1)
                clo_prev, A_prev = clo, X[:, 2:4].copy()
            s = s_state
        else:
            s = np.clip(shares(X, S, u, p) + offset * np.exp(-n * DT / tau), 0, 1)
        R = X[:, 2:4] * s
        reach_out.append(R)
        Hb = X[:, 0] + DT * p["kb"] * R[:, 1]
        Hr = X[:, 1] + DT * (p["kr"] + p["kr_wd"] * wd) * R[:, 0]
        Hn = np.maximum(np.column_stack([Hb, Hr]), 0)
        ratio = np.divide(Hn, X[:, :2], out=np.zeros_like(Hn), where=X[:, :2] > 1e-9)
        A = X[:, 2:4] * np.column_stack(
            [ratio[:, 0] ** p["p_b"], ratio[:, 1] ** (p["p_r"] + p["p_r_focus"] * focus)]
        )
        ga, gh, gw = gate(G["advance"], d, S), gate(G["hold"], d, S), gate(G["withdraw"], d, S)
        _, c2, w2, b2 = G["withdraw"][8:]
        flee = ndtr((c2 + b2 * S[:, 0] - d) / w2)
        sb_c = s[:, 0]  # contact uses the share with the offset, as `library` does when reach is given
        contact = 1 - (1 - np.clip(sb_c, 0, 1)) ** (10 * X[:, 2])
        dd = (
            p["d_adv"] * adv * ga
            + p["d_wd"] * wd * gw
            + p["d_flee"] * wd * flee
            + p["d_hold_far"] * hold * gh * (1 - contact)
            + p["d_hold_contact"] * hold * ga * contact
        )
        X = np.column_stack([Hn, A, np.maximum(d + DT * dd, 0)])
        out.append(X.copy())
    Y = np.stack(out, 1)
    return (Y, np.stack(reach_out, 1)) if return_reach else Y


def shape_step(d, S, u, coef):
    rates = [np.column_stack(list(_shape_library(d, S[:, j], u).values())) @ coef[j] for j in range(4)]
    return np.maximum(S + DT * np.column_stack(rates), 0)


def metrics(Y, B, horizons=H):
    Yt = B["Y"]
    row = {}
    for h in horizons:
        n = round(h / DT)
        ok = ~np.isnan(Yt[:, n, 0])
        row[f"health {h}s"] = np.abs(Y[ok, n, :2] - Yt[ok, n, :2]).mean()
        real = Yt[ok, n, 0] - Yt[ok, n, 1] - (Yt[ok, 0, 0] - Yt[ok, 0, 1])
        pred = Y[ok, n, 0] - Y[ok, n, 1] - (Y[ok, 0, 0] - Y[ok, 0, 1])
        row[f"margin R2 {h}s"] = 1 - np.sum((pred - real) ** 2) / np.sum((real - real.mean()) ** 2)
        row[f"alive {h}s"] = 10 * np.abs(Y[ok, n, 2:4] - Yt[ok, n, 2:4]).mean()
        row[f"distance {h}s"] = np.abs(Y[ok, n, 4] - Yt[ok, n, 4]).mean()
    return pd.Series(row)


def paired_bootstrap(Ya, Yb, B, h=4, n_boot=1000, seed=0):
    """95% intervals, resampling battles, of: health error a, b, b−a; margin R² a, b, b−a at horizon h."""
    Yt = B["Y"]
    n = round(h / DT)
    ok = ~np.isnan(Yt[:, n, 0])
    groups, inv = np.unique(B["group"], return_inverse=True)
    rng = np.random.default_rng(seed)
    real = np.nan_to_num(Yt[:, n, 0] - Yt[:, n, 1] - (Yt[:, 0, 0] - Yt[:, 0, 1]))
    res = {k: [] for k in ("health a", "health b", "health b-a", "margin a", "margin b", "margin b-a")}
    ea = np.nan_to_num(np.abs(Ya[:, n, :2] - Yt[:, n, :2]).mean(1))
    eb = np.nan_to_num(np.abs(Yb[:, n, :2] - Yt[:, n, :2]).mean(1))
    pa = Ya[:, n, 0] - Ya[:, n, 1] - (Ya[:, 0, 0] - Ya[:, 0, 1])
    pb = Yb[:, n, 0] - Yb[:, n, 1] - (Yb[:, 0, 0] - Yb[:, 0, 1])
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[inv] * ok
        ha, hb = (w * ea).sum() / w.sum(), (w * eb).sum() / w.sum()
        mu = (w * real).sum() / w.sum()
        den = (w * (real - mu) ** 2).sum()
        ma, mb = 1 - (w * (pa - real) ** 2).sum() / den, 1 - (w * (pb - real) ** 2).sum() / den
        for k, v in zip(res, (ha, hb, hb - ha, ma, mb, mb - ma), strict=True):
            res[k].append(v)
    return {k: (np.quantile(v, 0.025), np.quantile(v, 0.975)) for k, v in res.items()}


FIT_N = [8, 16, 24, 32, 48, 64]
SCALE = np.array([0.05, 0.05, 0.1, 0.1, 0.5])  # health, health, alive, alive, distance
LOWER = {
    "tau_b": 0.05,
    "tau_r": 0.05,
    "g_b": 0.05,
    "g_r": 0.05,
    "p_b": 0.0,
    "p_r": 0.0,
    "lam_b": 0.0,
    "lam_r": 0.0,
    "c_b": 0.05,
    "c_r": 0.05,
    "beta_b": 0.0,
    "beta_r": 0.0,
}
UPPER = {"tau_b": 20.0, "tau_r": 20.0, "c_b": 20.0, "c_r": 20.0}


def subsample(B, n, seed=0):
    idx = np.random.default_rng(seed).choice(len(B["X0"]), min(n, len(B["X0"])), replace=False)
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(B["X0"]) else v) for k, v in B.items()}


def residuals(x, names, p0, B, weights=SCALE, margin_w=0.0, reach_w=0.0):
    p = dict(p0, **dict(zip(names, x, strict=True)))
    Y, Rm = simulate(p, B, steps=max(FIT_N), return_reach=True)
    res = []
    for n in FIT_N:
        if reach_w:
            res.append(reach_w * np.nan_to_num((Rm[:, n - 1] - B["Yreach"][:, n - 1]) / 0.1).ravel())
        r = (Y[:, n] - B["Y"][:, n]) / weights
        r = np.nan_to_num(r)
        res.append(r.ravel())
        if margin_w:
            res.append(
                margin_w
                * np.nan_to_num((Y[:, n, 0] - Y[:, n, 1]) - (B["Y"][:, n, 0] - B["Y"][:, n, 1]))
                / 0.05
            )
    return np.concatenate(res) / np.sqrt(len(B["X0"]))


def refit(names, p0, B, **kw):
    x0 = np.array([p0[k] for k in names], float)
    lo = np.array([LOWER.get(k, -np.inf) for k in names])
    sol = least_squares(
        residuals,
        x0,
        args=(names, p0, B),
        kwargs=kw,
        bounds=(lo, np.array([UPPER.get(k, np.inf) for k in names])),
        x_scale="jac",
        diff_step=1e-3,
        max_nfev=60,
    )
    return dict(p0, **dict(zip(names, sol.x, strict=True))), sol
