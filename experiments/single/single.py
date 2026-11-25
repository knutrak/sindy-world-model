# ruff: noqa: E501, E702, I001
"""One model with the scenario parameters as input (docs/single/00_plan.md). Exploratory, val.

uv run --extra explore experiments/single/single.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "experiments/mixed", "experiments/generalisation", "experiments/triple"):
    sys.path.insert(0, str(ROOT / p))
import evaluate_final as E  # noqa: E402
from triple import BONUS, ARMORED, config  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models import pipeline as PL  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import OTHER, SIDES, Layout, Pipeline, prepare  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402
from sindy_wm.states.typed import unit_stats  # noqa: E402

PROC = DATA_DIR / "processed"
ARMIES = {
    "marines": ("exp06_intents", ("MARINE",), 7.0),
    "marauders": ("exp09_marauders", ("MARAUDER",), 7 * 7.125 / 5.75),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER"), 8.0),
    "triple": ("exp12_triple", ("MARINE", "MARAUDER", "STALKER"), 8.0),
    "stalkers": ("exp07_stalkers", ("STALKER",), 7 * 7.25 / 5.75),
}
VARIANTS = ("DM", "DP", "CM", "CP", "HM", "A1", "SM", "SP", "R4", "R3")  # V0 = marines, A3 = HP = DM
BASES = ("R", "A", "Rfocus", "Rhold", "R2")


def props(a: str, t: str) -> dict:
    """Net damage, 1 / cooldown, 1 / hp, range, speed of attacker a against target t, relative to marine on marine."""
    a, t = a.upper(), t.upper()
    sa, st = unit_stats(a), unit_stats(t)
    dmg = sa["damage"] + (BONUS.get(a, 0.0) if t in ARMORED else 0.0)
    total = st["hp"] + st["shield"]
    net = total / (st["shield"] / dmg + st["hp"] / (dmg - st["armor"]))
    return {
        "dmg": net / 6,
        "icd": 10 / np.ceil(sa["cooldown"] * 16),
        "ihp": 45 / total,
        "range": sa["range"] / 5,
        "speed": sa["speed"] / 3.15,
    }


def monomials(a, t, large):
    p = props(a, t)
    rate = p["dmg"] * p["icd"] * p["ihp"]
    if not large:
        return {"rate": rate}
    out = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                n = "·".join([x for x, u in (("dmg", i), ("icd", j), ("ihp", k)) if u]) or "1"
                out["rate" if i and j and k else n] = p["dmg"] ** i * p["icd"] ** j * p["ihp"] ** k
    return out | {"rate·range": rate * p["range"], "rate·speed": rate * p["speed"]}


class Single(Pipeline):
    """The pipeline's rollout with coefficients from the shared model."""

    def _bases(self, R, s, A, focus, hold):
        return {"R": R * s, "A": A * s, "Rfocus": R * s * focus, "Rhold": R * s * hold, "R2": R * R * s}

    def _terms(self, R_o, sig, A_o, wd, o, t):
        u, L = self._u, {}
        for a in self.L.types:
            R = R_o[a] * (1 - wd) if o == "blue" else R_o[a]
            for n, v in self._bases(R, sig[(a, t)], A_o[a], u[:, 3], u[:, 1]).items():
                L[f"{n}[{a}]"] = v
        L["1"] = np.ones_like(wd)
        return L

    def _dist_library(self, X, U, S, reach):
        X = X.copy()
        X[:, 4] = X[:, 4] * self.dscale
        r, self.withdraw_radius = self.withdraw_radius, self.withdraw_radius * self.dscale
        try:
            F = super()._dist_library(X, U, S * self.dscale, reach)
        finally:
            self.withdraw_radius = r
        return {k: v * self.vscale for k, v in F.items()}


def blank(types, wr) -> Single:
    m = Single(tuple(types), withdraw_radius=wr, **config())
    m.dscale = 5.75 / np.mean(list(m.rho.values()))
    m.vscale = np.mean([unit_stats(t.upper())["speed"] for t in m.L.types]) / 3.15
    return m


def load(name, split):
    if name in ARMIES:
        e, types, wr = ARMIES[name]
        return pd.read_parquet(PROC / f"{e}_typed" / f"{split}.parquet"), types, wr
    t = pd.read_parquet(PROC / "exp13_properties_typed" / f"{name.lower()}.parquet")
    first = t.index.get_level_values(0) < 500
    return (t[first] if split == "train" else t[~first]), (f"MARINE_{name}",), 7.0


_cache = {}


def rows_of(name, n):
    if name not in _cache:
        t, types, wr = load(name, "train")
        r = prepare(t, Layout(tuple(x.lower() for x in types)))
        _cache[name] = (r.sample(min(60_000, len(r)), random_state=0), types, wr)
    r, types, wr = _cache[name]
    return r.head(n), types, wr


def fit(pool: dict, large: bool, thr: float) -> dict:
    """Shared parameters from a pool {name: rows}."""
    data = [(*rows_of(k, n), blank(*rows_of(k, n)[1:])) for k, n in pool.items()]
    out = {"health": {}, "p": {}, "share": {}, "large": large}
    for s in SIDES:
        o = OTHER[s]
        Th, Y, AL, HE, DA, FO, names = [], [], [], [], [], [], None
        X, y, rho = [], [], []
        for r, _, _, m in data:
            tt, wd = m.L.types, r["withdraw"].to_numpy()
            m._u = np.column_stack([r["advance"], r["hold"], wd, r["focus"]]).astype(float)
            sig = m._sigma_measured(r, o)
            for t in tt:
                T = m._terms(
                    {a: r[f"reach_{o}_{a}"].to_numpy() for a in tt},
                    sig,
                    {a: r[f"alive_{o}_{a}"].to_numpy() for a in tt},
                    wd,
                    o,
                    t,
                )
                F = {}
                for b in BASES if large else ("R", "A"):
                    for a in tt:
                        for g, v in monomials(a, t, large).items():
                            F[(b, g)] = F.get((b, g), 0) + T[f"{b}[{a}]"] * v
                F[("1", "1")] = T["1"]
                names = list(F)
                Th.append(np.column_stack(list(F.values())))
                Y.append(r[f"d_health_{s}_{t}"].to_numpy())
                AL.append(r[f"alive_{s}_{t}"].to_numpy())
                HE.append(r[f"health_{s}_{t}"].to_numpy())
                DA.append(r[f"d_alive_{s}_{t}"].to_numpy())
                FO.append(r["focus"].to_numpy())
                q = r[r[f"alive_{s}_{t}"] > 0]
                nopp = 10 * sum(q[f"alive_{o}_{a}"] for a in tt)
                X.append(
                    np.column_stack(
                        [
                            q["distance"] - q[f"front_{s}_{t}"].fillna(0),
                            q[f"depth_{s}"],
                            q[f"depth_{o}"],
                            q[f"width_{s}"],
                            q[f"width_{o}"],
                            nopp,
                        ]
                    )
                )
                y.append((q[f"reach_{s}_{t}"] / q[f"alive_{s}_{t}"]).to_numpy())
                rho.append(np.full(len(q), m.rho[t]))
        Th, Y, AL, HE, DA, FO = np.vstack(Th), *(np.concatenate(v) for v in (Y, AL, HE, DA, FO))
        c = PL.stlsq(Th, Y, thr)
        out["health"][s] = {n: v for n, v in zip(names, c, strict=True) if v}
        ok = (HE > 0.03) & (AL > 0)
        x = (Th @ c) * AL / np.maximum(HE, 1e-9)
        Z = np.column_stack([x, x * FO]) if s == "red" else x[:, None]
        pc, *_ = np.linalg.lstsq(Z[ok], DA[ok], rcond=None)
        out["p"][s] = (float(pc[0]), float(pc[1]) if s == "red" else 0.0)
        X, y, rho = np.vstack(X), np.concatenate(y), np.concatenate(rho)
        i = np.random.default_rng(0).choice(len(y), min(200_000, len(y)), replace=False)
        out["share"][s], _ = curve_fit(
            lambda Q, la, be, g: PL.derived_share(*Q[:6], la, be, g, rho=Q[6]),
            np.column_stack([X, rho])[i].T,
            y[i],
            p0=(1.0, 1.0, 1.0),
            maxfev=20000,
        )
    # distance in scaled units
    Ts = []
    for r, _, wr, m in data:
        T = PL.team_rows(r, m.L)
        T[["distance", *PL.SHAPE]] *= m.dscale
        T["d_distance"] /= m.vscale
        T["wr"] = wr * m.dscale
        keep = ["movement", "distance", *PL.SHAPE, "d_distance", "wr", *PL.CTRL] + [
            f"{k}_{s}" for k in ("health", "alive", "reach") for s in SIDES
        ]
        Ts.append(T[list(dict.fromkeys(keep))])
    T = pd.concat(Ts, ignore_index=True)
    P = {"gates": PL.fit_gates(T, 1.0, skip=("withdraw",)), "form": "derived"}
    P["gates"]["withdraw"] = np.array([0, 99, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], float)
    for s in SIDES:
        a, o = T[T[f"alive_{s}"] > 0], OTHER[s]
        a = a.sample(min(200_000, len(a)), random_state=0)
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
        m = Pipeline(("MARINE",), withdraw_radius=wr, **config())
        m.P_team = P
        F = m._dist_library(
            g[["health_blue", "health_red", "alive_blue", "alive_red", "distance"]].to_numpy(),
            g[PL.CTRL].to_numpy(float),
            g[PL.SHAPE].to_numpy(),
            g[["reach_blue", "reach_red"]].to_numpy(),
        )
        out["dist_names"] = list(F)
        parts.append(np.column_stack(list(F.values())))
        yd.append(g["d_distance"].to_numpy())
    N = out["dist_names"]
    cols = [
        j for j in PL._family_cols(N, PL.MOVE_FAMILY) if N[j] not in (*PL.DIST_EXCLUDE, "close*wd", "flee*wd")
    ] + [N.index("danger*wd"), N.index("walk_r*wd")]
    cd = np.zeros(len(N))
    cd[cols] = PL.stlsq(np.vstack(parts)[:, cols], np.concatenate(yd), config()["thr_distance"])
    out["dist_coef"], out["P_team"] = cd, P
    return out


def build(par: dict, types, wr) -> Single:
    m = blank(types, wr)
    m.coef, m.p, m.share, m.alloc_b = {}, {}, {}, {}
    for s in SIDES:
        for t in m.L.types:
            c = {}
            for (b, g), v in par["health"][s].items():
                if b == "1":
                    c["1"] = v
                    continue
                for a in m.L.types:
                    c[f"{b}[{a}]"] = c.get(f"{b}[{a}]", 0.0) + v * monomials(a, t, par["large"])[g]
            m.coef[(s, t)], m.p[(s, t)], m.share[(s, t)] = c, par["p"][s], par["share"][s]
    m.dist_names, m.dist_coef, m.P_team = par["dist_names"], par["dist_coef"], par["P_team"]
    return m


def describe(par):
    L = [
        f"H_{s}' = " + " ".join(f"{v:+.4f}·{b}·{g}" for (b, g), v in par["health"][s].items()) for s in SIDES
    ]
    L += [f"p {s}: {par['p'][s][0]:.3f} {par['p'][s][1]:+.3f}·focus" for s in SIDES]
    return "\n".join(
        L
        + [
            "d' / v = "
            + " ".join(f"{c:+.3f}·{n}" for n, c in zip(par["dist_names"], par["dist_coef"], strict=True) if c)
        ]
    )


def main():
    frozen = pickle.load(open(H.CACHE / "generalisation_frozen.pkl", "rb"))
    own = {k: frozen[k] for k in ("marines", "marauders", "mixed")}
    own["triple"] = pickle.load(open(PROC / "exp12_triple_typed" / "triple_model.pkl", "rb"))
    own["stalkers"] = pickle.load(open(PROC / "exp07_stalkers_typed" / "stalker_model.pkl", "rb"))
    refits = pickle.load(open(PROC / "exp13_properties_typed" / "refits.pkl", "rb"))
    own |= {v: refits[v] for v in VARIANTS}
    var = {v: 12_000 for v in VARIANTS}
    full = {k: 60_000 for k in ("marines", "marauders", "mixed", "triple")} | var
    lo = {"marines": 60_000} | var
    pars = {
        "small": fit(full, False, 0.15),
        "large": fit(full, True, 0.15),
        "large 0.05": fit(full, True, 0.05),
        "LO small": fit(lo, False, 0.15),
        "LO large": fit(lo, True, 0.15),
    }
    tab, verdict = [], {}
    for name in own:
        st, types, wr = load(name, "val")
        B = Layout(tuple(x.lower() for x in types)).batch(add_controls(st))
        T, Y = B["team"], {}
        Y["own"] = own[name].L.team_view(own[name].rollout(B))
        for k, par in pars.items():
            m = build(par, types, wr)
            Y[k] = m.L.team_view(m.rollout(B))
        hm = {k: H.horizon_metrics(v, T) for k, v in Y.items()}
        for k in Y:
            row = {
                "army": name,
                "model": k,
                "health 2s": hm[k]["health_2s"],
                "ratio": hm[k]["health_2s"] / hm["own"]["health_2s"],
                "health 4s": hm[k]["health_4s"],
                "R2 2s": hm[k]["margin_r2_2s"],
                "distance 2s": hm[k]["distance_2s"],
            }
            if k != "own":
                ci = E.ci_health_diff(Y["own"], Y[k], T, h=2)
                row["- own 95%"] = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                verdict[(name, k)] = (ci[1] < 0, row["ratio"])
            tab.append(row)
    A = list(ARMIES)
    L = [
        "# One model with the scenario parameters as input: results (exploratory, val)",
        "",
        "Plan: `00_plan.md`.",
        "",
        "## Rules",
        "",
    ]
    for k in ("small", "large", "large 0.05"):
        L.append(
            f"- {k}: E1 better in {sum(bool(verdict[(a, k)][0]) for a in A)} of 5 (needs 3); E2 worst ratio {max(verdict[(a, k)][1] for a in A):.3f} (limit 1.10)"
        )
    for k in ("LO small", "LO large"):
        L.append(
            f"- {k}: E3 ratios marauders {verdict[('marauders', k)][1]:.3f}, stalkers {verdict[('stalkers', k)][1]:.3f} (limit 1.15)"
        )
    L += ["", "```", pd.DataFrame(tab).round(4).to_string(), "```", "", "## Equations", ""]
    for k, par in pars.items():
        L += [f"{k}:", "", "```", describe(par), "```", ""]
    (ROOT / "docs" / "single" / "01_results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
