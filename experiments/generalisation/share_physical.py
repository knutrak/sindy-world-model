# ruff: noqa: E501, B023
"""Share in reach as a state with a small physical library, selected by STLSQ on SG rates.

Plan: docs/generalisation/22_share_physical_plan.md (exploratory, 26 Nov). Run from the repository root:
uv run --no-sync python experiments/generalisation/share_physical.py dry      200 battles per scenario, timed
uv run --no-sync python experiments/generalisation/share_physical.py run      fit on train, roll out on val
uv run --no-sync python experiments/generalisation/share_physical.py weak     weak form, radius 8 and 16 (amendment 1)
uv run --no-sync python experiments/generalisation/share_physical.py sat      + fill and saturate (amendment 2), one-step
uv run --no-sync python experiments/generalisation/share_physical.py health   health library + own units alive (amendment 2)
"""

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sindy_wm.models.intent_model import DT, add_controls, stlsq
from sindy_wm.models.pipeline import OTHER, SIDES
from sindy_wm.models.reach_state import BLOW, ReachStatePipeline, bump, closure_share_fn
from sindy_wm.paths import DATA_DIR

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
OUT = DATA_DIR / "processed" / "generalisation" / "share_physical"
SCEN = {"marines": "exp06_intents_typed", "mixed": "exp08_mixed_typed"}
OLD_TERMS = ("relax", "follow", "deaths", "approach", "recede", "withdraw")
NEW_TERMS = ("fill", "saturate")  # amendment 2: (1 - x) and (1 - x) in contact
TERMS = OLD_TERMS + NEW_TERMS
SIGN = {
    "relax": 1,
    "follow": 1,
    "deaths": 1,
    "approach": 1,
    "recede": -1,
    "withdraw": -1,
    "fill": 1,
    "saturate": 1,
}
USE_NEW = False  # set by the `sat` command
EXCLUDE: set = set()  # amendment 2b: "nosat" drops follow and relax (no closure in the share equation)
THRESHOLDS = (0.05, 0.15)
HORIZONS = (1.0, 2.0, 4.0, 8.0)
SG = 9


def library(x, clo, dclo, dlnA, dd, depth, wd, side, contact=None) -> dict:
    """The candidate terms (plan 22); dd is the rate of the distance to the type's front. With USE_NEW,
    also fill = 1 - x and saturate = (1 - x) * contact, contact = 1 when any unit of either side has an
    enemy in reach (amendment 2)."""
    dep = np.maximum(depth, 0.5)
    L = {
        "relax": clo - x,
        "follow": dclo,
        "deaths": (1 - x) * dlnA,
        "approach": (1 - x) * np.maximum(-dd, 0) / dep,
        "recede": x * np.maximum(dd, 0) / dep,
    }
    if side == "blue":
        L["withdraw"] = x * wd
    if USE_NEW:
        L["fill"] = 1 - x
        L["saturate"] = (1 - x) * contact
    return L


def load(scen: str, split: str, max_battles: int | None = None) -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / "processed" / SCEN[scen] / f"{split}.parquet")
    if max_battles:
        d = d.loc[d.index.get_level_values(0).unique()[:max_battles]]
    return add_controls(d)


def rows(states, m, side, a, radius: int = 0, stride: int = 8):
    """Regression rows for one equation. radius 0: one row per step, SG-9 rates of x, clo, ln A and d.
    radius > 0 (weak form, amendment 1): one row per window of 2 radius + 1 steps every `stride` steps,
    target −∫φ′x, columns ∫φΘ, `follow` as −∫φ′clo; all steps of the window alive."""
    if radius:
        phi, dphi, w = bump(radius)
    L, clo_fn = m.L, closure_share_fn(m)
    Th, y = {k: [] for k in TERMS}, []
    for _, g in states.groupby(level=0, sort=False):
        if len(g) < SG + 2:
            continue
        A = g[f"alive_{side}_{a}"].to_numpy(float)
        x = np.divide(g[f"reach_{side}_{a}"].to_numpy(float), A, out=np.zeros_like(A), where=A > 0)
        clo = clo_fn(g, side, a)
        d = g["distance"].to_numpy(float) - g[f"front_{side}_{a}"].fillna(0).to_numpy(float)
        contact = (sum(g[f"reach_{s2}_{t}"].to_numpy(float) for s2 in SIDES for t in L.types) > 0).astype(
            float
        )
        rate = lambda v: savgol_filter(v, SG, 2, deriv=1, delta=DT)  # noqa: E731
        lib = library(
            x,
            clo,
            rate(clo),
            rate(np.log(np.maximum(A, 1e-3))),
            rate(d),
            g[f"depth_{side}"].fillna(0).to_numpy(float),
            g["withdraw"].to_numpy(float),
            side,
            contact,
        )
        opp = sum(g[f"alive_{OTHER[side]}_{t}"].to_numpy(float) for t in L.types)
        ok = (A > 0) & (opp > 0)
        ok[: SG // 2] = ok[-(SG // 2) :] = False
        ok &= np.convolve(~((A > 0) & (opp > 0)), np.ones(SG), "same") == 0  # whole window alive
        if not radius:
            for k, v in lib.items():
                Th[k].append(v[ok])
            y.append(rate(x)[ok])
            continue
        live = (A > 0) & (opp > 0)
        lib["follow"] = clo  # integrated by parts below
        for c in range(max(radius, SG // 2), len(x) - max(radius, SG // 2) - 1, stride):
            sl = slice(c - radius, c + radius + 1)
            if not live[sl].all():
                continue
            for k, v in lib.items():
                Th[k].append([-(w * dphi) @ v[sl] if k == "follow" else (w * phi) @ v[sl]])
            y.append([-(w * dphi) @ x[sl]])
    names = [k for k in TERMS if Th[k] and k not in EXCLUDE]
    return np.column_stack([np.concatenate(Th[k]) for k in names]), np.concatenate(y), names


def fit_signed(theta, y, names, thr):
    """STLSQ; a kept term with the wrong sign is dropped and the rest refitted (plan 22)."""
    cols = list(range(len(names)))
    dropped = []
    while True:
        c = np.zeros(len(names))
        c[cols] = stlsq(theta[:, cols], y, thr)
        bad = [j for j in cols if c[j] != 0 and np.sign(c[j]) != SIGN[names[j]]]
        if not bad:
            return c, dropped
        worst = max(bad, key=lambda j: abs(c[j]) * theta[:, j].std())
        dropped.append(names[worst])
        cols.remove(worst)


class SharePhysical(ReachStatePipeline):
    """The frozen pipeline with the share in reach as a state, x' from the physical library."""

    def _reach(self, n, X, S, F, u, sh, offset, B) -> dict:
        L = self.L
        if self.eqs is None:
            R = super(ReachStatePipeline, self)._reach(n, X, S, F, u, sh, offset, B)
        else:
            if n == 0:
                for k in sh:
                    A0 = X[:, L.idx("alive", *k)]
                    self._state[k] = np.clip(
                        np.divide(B["reach0"][:, L.pos(*k)], A0, out=np.zeros_like(A0), where=A0 > 1e-9), 0, 1
                    )
            else:
                Xp, Sp, Fp, up, shp = self._prev
                contact = (sum(Xp[:, L.idx("alive", *k)] * v for k, v in self._state.items()) > 1e-6).astype(
                    float
                )
                for k, (names, coef) in self.eqs.items():
                    side, a = k
                    j = L.pos(*k)
                    A_p, A_n = Xp[:, L.idx("alive", *k)], X[:, L.idx("alive", *k)]
                    x = self._state[k]
                    lib = library(
                        x,
                        sh[k],
                        (sh[k] - shp[k]) / DT,
                        (np.log(np.maximum(A_n, 1e-3)) - np.log(np.maximum(A_p, 1e-3))) / DT,
                        ((X[:, -1] - F[:, j]) - (Xp[:, -1] - Fp[:, j])) / DT,
                        S[:, 0 if side == "blue" else 1],
                        u[:, 2],
                        side,
                        contact,
                    )
                    new = x + DT * sum(c * lib[nm] for nm, c in zip(names, coef, strict=True) if c)
                    self.blown |= (A_n > 1e-9) & (~np.isfinite(new) | (new < -BLOW) | (new > 1 + BLOW))
                    self._state[k] = np.clip(np.nan_to_num(new), 0, 1)
            R = {k: X[:, L.idx("alive", *k)] * self._state[k] for k in sh}
        self._prev = (X.copy(), S.copy(), F.copy(), u.copy(), {k: v.copy() for k, v in sh.items()})
        self._log.append(np.column_stack([sum(R[(s, a)] for a in L.types) for s in SIDES]))
        return R


def true_reach(states, L, B, steps):
    cols = {s: [f"reach_{s}_{a}" for a in L.types] for s in SIDES}
    arr = {
        ep: np.column_stack([g[cols[s]].sum(axis=1).to_numpy(float) for s in SIDES])
        for ep, g in states.groupby(level=0, sort=False)
    }
    out = np.full((len(B["ep"]), steps + 1, 2), np.nan)
    for i, (ep, k0) in enumerate(zip(B["ep"], B["k0"], strict=True)):
        v = arr[ep][k0 : k0 + steps + 1]
        out[i, : len(v)] = v
    return out


def boot(x, grp, rng, n=1000):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((w * sums).sum() / (w * cnt).sum())
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def print_share_libraries() -> None:
    print("Share library x' (x = units in reach / alive), expected sign in brackets")
    expl = {
        "relax": "clo - x            relax to the closure",
        "follow": "clo'               follow the closure",
        "deaths": "(1 - x) (ln A)'    deaths happen in reach",
        "approach": "(1 - x) max(-d', 0) / depth   walking in",
        "recede": "x max(d', 0) / depth           walking out",
        "withdraw": "x wd               Blue withdraws (Blue only)",
        "fill": "1 - x              fill up, always",
        "saturate": "(1 - x) contact    fill up once in contact",
    }
    for k in TERMS:
        tag = "NEW" if k in NEW_TERMS else "old"
        print(f"  {tag}  {k:9s} [{'+' if SIGN[k] > 0 else '-'}]  {expl[k]}")


def run(max_battles=None, radii=(0,), tag="") -> None:
    out_dir = OUT / (tag or ("weak" if radii != (0,) else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = pickle.load(open(FROZEN, "rb"))
    rng = np.random.default_rng(0)
    fits, val_rows = {}, []
    for scen in SCEN:
        m = frozen[scen]
        t0 = time.time()
        st = load(scen, "train", max_battles)
        for side in SIDES:
            for a in m.L.types:
                for rad in radii:
                    theta, y, names = rows(st, m, side, a, rad)
                    for thr in THRESHOLDS:
                        raw = np.zeros(len(names))
                        raw[:] = stlsq(theta, y, thr)
                        c, dropped = fit_signed(theta, y, names, thr)
                        r2 = 1 - ((y - theta @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                        fits[(scen, side, a, rad, thr)] = {
                            "names": names,
                            "raw": raw.tolist(),
                            "coef": c.tolist(),
                            "dropped": dropped,
                            "r2": float(r2),
                            "n": len(y),
                        }
                        eq = " ".join(f"{v:+.3f}·{nm}" for nm, v in zip(names, c, strict=True) if v)
                        print(
                            f"{scen} {side} {a} r{rad} thr {thr}: {eq}  [dropped {dropped}, R² {r2:.3f}]",
                            flush=True,
                        )
        print(f"{scen} fitted in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        sv = load(scen, "val", max_battles)
        horizon = round(max(HORIZONS) / DT)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = m.L.batch(sv, horizon=horizon)
        TR, TY, ep = true_reach(sv, m.L, B, horizon), B["team"]["Y"], B["ep"]
        models = {"frozen": SharePhysical.from_frozen(m, None)}
        for rad in radii:
            for thr in THRESHOLDS:
                f = {(s, a): fits[(scen, s, a, rad, thr)] for s in SIDES for a in m.L.types}
                eqs = {k: (v["names"], v["coef"]) for k, v in f.items()}
                models[f"{'weak r' + str(rad) if rad else (tag or 'physical')} {thr}"] = (
                    SharePhysical.from_frozen(m, eqs)
                )
        out = {}
        for name, mod in models.items():
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                Y = mod.L.team_view(mod.rollout(B))
            out[name] = (Y, mod.R_log, mod.blown.copy())
        fY, fR, _ = out["frozen"]
        for name, (Y, R, blown) in out.items():
            for h in HORIZONS:
                n = round(h / DT)
                ok = ~np.isnan(TY[:, n, 0]) & ~np.isnan(TR[:, n - 1, 0])
                real = TY[ok, n, 0] - TY[ok, n, 1] - (TY[ok, 0, 0] - TY[ok, 0, 1])

                def errs(Y, R, ok=ok, n=n, real=real):
                    pred = Y[ok, n, 0] - Y[ok, n, 1] - (Y[ok, 0, 0] - Y[ok, 0, 1])
                    return (
                        10 * np.abs(R[ok, n - 1] - TR[ok, n - 1]).sum(1),
                        np.abs(Y[ok, n, :2] - TY[ok, n, :2]).mean(1),
                        np.abs(pred - real),
                        np.abs(Y[ok, n, 4] - TY[ok, n, 4]),  # distance (team view: H_B, H_R, A_B, A_R, d)
                        pred,
                    )

                re, he, me, de, pred = errs(Y, R)
                rf, hf, mf, _, _ = errs(fY, fR)
                row = {
                    "scenario": scen,
                    "model": name,
                    "h": h,
                    "starts": int(ok.sum()),
                    "reach error": re.mean(),
                    "health error": he.mean(),
                    "margin error": me.mean(),
                    "margin R²": 1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum(),
                    "distance error": de.mean(),
                    "blow-ups": float(blown.mean()),
                }
                if name != "frozen":
                    for lab, a_, b_ in (("reach", re, rf), ("margin", me, mf), ("health", he, hf)):
                        lo, hi = boot(a_ - b_, ep[ok], rng)
                        row[f"{lab} − frozen 95%"] = f"[{lo:+.4f}, {hi:+.4f}]"
                    row["health ratio"] = he.mean() / hf.mean()
                val_rows.append(row)
        print(f"{scen} rolled out in {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(val_rows)
    with pd.option_context("display.width", 250):
        print(t.round(4).to_string(index=False))
    if max_battles is None:
        pickle.dump(fits, open(out_dir / "fits.pkl", "wb"))
        json.dump(val_rows, open(out_dir / "val.json", "w"), indent=1, default=float)


def health() -> None:
    """Amendment 2: the frozen health library vs the same + own units alive of the target type (the
    exponential shape, A' proportional to A). STLSQ with the frozen threshold on the frozen rows (train)."""
    from sindy_wm.models.pipeline import prepare

    frozen = pickle.load(open(FROZEN, "rb"))
    print(
        "Health library H_s,t' (frozen): R[a]*s[t'] enemy units in reach x fire split [-], A[a] enemy alive [-], 1 [-]"
    )
    print("New: + A_own[t] own units alive of type t [-] (losses proportional to own numbers)\n")
    for scen in SCEN:
        m = frozen[scen]
        r = prepare(load(scen, "train"), m.L)
        wd = r["withdraw"].to_numpy()
        for s in SIDES:
            o = OTHER[s]
            R_o = {a: r[f"reach_{o}_{a}"].to_numpy() for a in m.L.types}
            A_o = {a: r[f"alive_{o}_{a}"].to_numpy() for a in m.L.types}
            sig = m._sigma_measured(r, o)
            for t in m.L.types:
                Lb = m._terms(R_o, sig, A_o, wd, o, t)
                y = r[f"d_health_{s}_{t}"].to_numpy()
                c_old = stlsq(np.column_stack(list(Lb.values())), y, m.thr_health)
                Ln = Lb | {"A_own": r[f"alive_{s}_{t}"].to_numpy()}
                th = np.column_stack(list(Ln.values()))
                c_new = stlsq(th, y, m.thr_health)
                ls = np.linalg.lstsq(th, y, rcond=None)[0]
                eff = ls * th.std(0) / y.std()

                def fmt(names, c):
                    return " ".join(f"{v:+.4f}·{k}" for k, v in zip(names, c, strict=True) if v) or "0"

                print(f"{scen} H_{s}^{t}'")
                print(f"   old library: {fmt(list(Lb), c_old)}")
                print(f"   new library: {fmt(list(Ln), c_new)}")
                print(
                    f"   A_own unthresholded: coef {ls[-1]:+.4f}, effect size {eff[-1]:+.3f} (threshold {m.thr_health})"
                )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "dry"
    if what == "health":
        health()
        sys.exit()
    if what.endswith("noclo"):
        USE_NEW, EXCLUDE = True, {"follow", "relax"}
        print_share_libraries()
        print("  (no closure: follow and relax left out)")
        run(200 if what.startswith("dry") else None, (0,), "noclo")
        sys.exit()
    if what.endswith("sat"):
        USE_NEW = True
        print_share_libraries()
        run(200 if what.startswith("dry") else None, (0,), "sat")
        sys.exit()
    radii = (8, 16) if "weak" in what else (0,)
    run(200 if what.startswith("dry") else None, radii)
