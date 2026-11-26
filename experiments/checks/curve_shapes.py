# ruff: noqa: E501, E731, E741, B023
"""Which time shapes fit health, alive, share in reach and distance per battle, and which battle features shift them.

Descriptive (each fit uses the battle's whole future), exploratory, 26 Nov 2026. Marine train battles (exp06) with one
movement all through, from first contact. Output: docs/checks/curve_shapes.md and figures in docs/checks/figures/.
uv run --no-sync python experiments/checks/curve_shapes.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

warnings.filterwarnings("ignore")
OUT = PROJECT_ROOT / "docs" / "checks"
FIG = OUT / "figures"


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


def fit(f, t, y, p0, bounds=(-np.inf, np.inf)):
    try:
        p, _ = curve_fit(f, t, y, p0=p0, bounds=bounds, maxfev=5000)
        return p, r2(y, f(t, *p))
    except Exception:
        return np.full(len(p0), np.nan), np.nan


SHAPES = {
    "health": {
        "linear": (lambda t, h0, b: h0 + b * t, lambda t, y: (y[0], -0.1)),
        "quadratic": (lambda t, h0, b, c: h0 + b * t + c * t**2, lambda t, y: (y[0], -0.1, 0.0)),
        "exponential": (lambda t, h0, l: h0 * np.exp(-l * t), lambda t, y: (y[0], 0.1)),
    },
    "alive": {
        "linear": (lambda t, a0, b: a0 + b * t, lambda t, y: (y[0], -1.0)),
        "quadratic": (lambda t, a0, b, c: a0 + b * t + c * t**2, lambda t, y: (y[0], -1.0, 0.0)),
        "exponential": (lambda t, a0, l: a0 * np.exp(-l * t), lambda t, y: (y[0], 0.1)),
        "delayed linear": (
            lambda t, a0, td, b: a0 - b * np.maximum(t - td, 0),
            lambda t, y: (y[0], 1.0, 1.0),
        ),
    },
    "share": {
        "saturating": (
            lambda t, xi, x0, tau: xi - (xi - x0) * np.exp(-t / np.maximum(tau, 1e-3)),
            lambda t, y: (1.0, y[0], 1.5),
        ),
        "logistic": (
            lambda t, xi, k, t0: xi / (1 + np.exp(-k * (t - t0))),
            lambda t, y: (1.0, 2.0, 0.5),
        ),
    },
    "distance": {
        "linear": (lambda t, d0, b: d0 + b * t, lambda t, y: (y[0], -1.0)),
        "quadratic": (lambda t, d0, b, c: d0 + b * t + c * t**2, lambda t, y: (y[0], -1.0, 0.0)),
        "settling": (
            lambda t, de, d0, tau: de + (d0 - de) * np.exp(-t / np.maximum(tau, 1e-3)),
            lambda t, y: (y[-1], y[0], 1.0),
        ),
    },
}


def lanchester(t, a, b, hb0, hr0):
    """Square law: H_B' = -a H_R, H_R' = -b H_B (health in ten full units)."""
    w = np.sqrt(max(a * b, 1e-12))
    hb = hb0 * np.cosh(w * t) - (a / w) * hr0 * np.sinh(w * t)
    hr = hr0 * np.cosh(w * t) - (b / w) * hb0 * np.sinh(w * t)
    return hb, hr


def battles():
    d = pd.read_parquet(DATA_DIR / "processed" / "exp06_intents_typed" / "train.parquet")
    mv = d.groupby(level=0)["movement"].agg(lambda m: m.iloc[0] if m.nunique() == 1 else None).dropna()
    out = []
    for ep, intent in mv.items():
        g = d.loc[ep]
        rb, rr = g["reach_blue_marine"].to_numpy(), g["reach_red_marine"].to_numpy()
        ab, ar = g["alive_blue_marine"].to_numpy() * 10, g["alive_red_marine"].to_numpy() * 10
        c = np.flatnonzero((rb > 0) | (rr > 0))
        if not len(c):
            continue
        k0 = c[0]
        both = (ab > 0) & (ar > 0)
        last = np.flatnonzero(both)[-1]
        sl = slice(k0, last + 1)
        t = g["t"].to_numpy()[sl] - g["t"].to_numpy()[k0]
        if len(t) < 16:
            continue
        out.append(
            {
                "ep": ep,
                "intent": intent,
                "target": g["target"].iloc[0],
                "n_blue": g["n_blue"].iloc[0],
                "n_red": g["n_red"].iloc[0],
                "t": t,
                "hb": g["health_blue_marine"].to_numpy()[sl],
                "hr": g["health_red_marine"].to_numpy()[sl],
                "ab": ab[sl],
                "ar": ar[sl],
                "sb": (rb[sl] * 10) / np.maximum(ab[sl], 1e-9),
                "sr": (rr[sl] * 10) / np.maximum(ar[sl], 1e-9),
                "d": g["distance"].to_numpy()[sl],
                "d_contact": g["distance"].iloc[k0],
                "depth_blue": g["depth_blue"].iloc[k0],
                "depth_red": g["depth_red"].iloc[k0],
                "width_blue": g["width_blue"].iloc[k0],
                "width_red": g["width_red"].iloc[k0],
            }
        )
    return out


def main():
    B = battles()
    print(len(B), "battles")
    shape_rows, par_rows = [], []
    for b in B:
        t = b["t"]
        row = {"ep": b["ep"]}
        series = {
            "health": (("blue", b["hb"]), ("red", b["hr"])),
            "alive": (("blue", b["ab"]), ("red", b["ar"])),
            "share": (("blue", b["sb"]), ("red", b["sr"])),
            "distance": (("team", b["d"]),),
        }
        for q, ss in series.items():
            for side, y in ss:
                for name, (f, p0) in SHAPES[q].items():
                    p, s = fit(f, t, y, p0(t, y))
                    shape_rows.append(
                        {"quantity": q, "side": side, "shape": name, "k": len(p0(t, y)), "R2": s}
                    )
                    row[f"{q}|{side}|{name}"] = p
        # Lanchester square law, both sides jointly (2 free numbers, start fixed)
        f = lambda tt, a, bb: np.concatenate(lanchester(tt[: len(t)], a, bb, b["hb"][0], b["hr"][0]))
        y = np.concatenate([b["hb"], b["hr"]])
        p, _ = fit(f, np.concatenate([t, t]), y, (0.1, 0.1), bounds=(0, 2))
        hb, hr = lanchester(t, *p, b["hb"][0], b["hr"][0])
        for side, yy, pp in (("blue", b["hb"], hb), ("red", b["hr"], hr)):
            shape_rows.append(
                {
                    "quantity": "health",
                    "side": side,
                    "shape": "Lanchester square (joint)",
                    "k": 1,
                    "R2": r2(yy, pp),
                }
            )
        # per-battle parameters for the shift question
        half = lambda h: t[np.argmax(h <= h[0] / 2)] if (h <= h[0] / 2).any() else np.nan
        par_rows.append(
            {
                **{
                    k: b[k]
                    for k in (
                        "ep",
                        "intent",
                        "target",
                        "n_blue",
                        "n_red",
                        "d_contact",
                        "depth_blue",
                        "depth_red",
                        "width_blue",
                        "width_red",
                    )
                },
                "lanchester a (Red's fire on Blue)": p[0],
                "lanchester b (Blue's fire on Red)": p[1],
                "a per Red unit": p[0],
                "alive slope Blue": row["alive|blue|linear"][1],
                "alive slope Red": row["alive|red|linear"][1],
                "share tau Blue": row["share|blue|saturating"][2],
                "share tau Red": row["share|red|saturating"][2],
                "distance tau": row["distance|team|settling"][2],
                "distance d_eq": row["distance|team|settling"][0],
                "half-life Blue": half(b["hb"]),
                "half-life Red": half(b["hr"]),
                "duration": t[-1],
            }
        )
    S = pd.DataFrame(shape_rows)
    shape_tab = (
        S.groupby(["quantity", "side", "shape", "k"])["R2"]
        .agg(["median", lambda r: r.quantile(0.1)])
        .round(3)
    )
    shape_tab.columns = ["median R²", "10% R²"]
    P = pd.DataFrame(par_rows)
    P["force ratio"] = P["n_blue"] / P["n_red"]
    P["total"] = P["n_blue"] + P["n_red"]
    P["focus"] = (P["target"] == "focus").astype(float)
    P["advance"] = (P["intent"] == "advance").astype(float)
    feats = [
        "n_blue",
        "n_red",
        "force ratio",
        "focus",
        "advance",
        "d_contact",
        "depth_blue",
        "depth_red",
        "width_blue",
        "width_red",
    ]
    targets = [
        "lanchester a (Red's fire on Blue)",
        "lanchester b (Blue's fire on Red)",
        "alive slope Blue",
        "alive slope Red",
        "share tau Blue",
        "share tau Red",
        "distance tau",
        "distance d_eq",
        "half-life Blue",
        "half-life Red",
        "duration",
    ]
    shift = []
    for tg in targets:
        q = P[[tg, *feats]].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = q[tg].quantile([0.02, 0.98])
        q = q[(q[tg] >= lo) & (q[tg] <= hi)]
        X, y = q[feats].to_numpy(float), q[tg].to_numpy(float)
        cv_all = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean()
        best = max(
            feats,
            key=lambda f: cross_val_score(
                LinearRegression(), q[[f]].to_numpy(float), y, cv=5, scoring="r2"
            ).mean(),
        )
        cv_best = cross_val_score(LinearRegression(), q[[best]].to_numpy(float), y, cv=5, scoring="r2").mean()
        coef = LinearRegression().fit(q[[best]].to_numpy(float), y).coef_[0]
        shift.append(
            {
                "parameter": tg,
                "median": q[tg].median(),
                "10-90%": f"{q[tg].quantile(0.1):.3g} to {q[tg].quantile(0.9):.3g}",
                "CV R² all features": cv_all,
                "best single feature": best,
                "its CV R²": cv_best,
                "slope": coef,
                "battles": len(q),
            }
        )
    Sh = pd.DataFrame(shift)
    # one time scale: health curves on t / half-life vs on t
    grid_t, grid_s = np.linspace(0, 12, 97), np.linspace(0, 3, 97)
    curves = {"t": [], "s": []}
    for b in B:
        h = (b["hb"] + b["hr"]) / (b["hb"][0] + b["hr"][0])
        if not (h <= 0.5).any():
            continue
        th = b["t"][np.argmax(h <= 0.5)]
        if th <= 0:
            continue
        curves["t"].append(np.interp(grid_t, b["t"], h, right=np.nan))
        curves["s"].append(np.interp(grid_s, b["t"] / th, h, right=np.nan))
    spread = {}
    for k, C in curves.items():
        C = np.array(C)
        med = np.nanmedian(C, 0)
        ok = ~np.isnan(C)
        spread[k] = 1 - np.nansum((C - med) ** 2) / np.nansum((C - np.nanmean(C[ok])) ** 2)
    th_tab = P[["half-life Blue", "half-life Red"]].describe().round(2)
    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a_, (k, g, lab) in zip(
        ax, (("t", grid_t, "time since contact (s)"), ("s", grid_s, "time / own half-life")), strict=True
    ):
        C = np.array(curves[k])
        for c in C[:40]:
            a_.plot(g, c, color="#888", lw=0.4, alpha=0.6)
        a_.plot(g, np.nanmedian(C, 0), color="k", lw=1.6)
        a_.set_xlabel(lab)
        a_.set_ylabel("health of both sides / start")
        a_.set_title(f"share of spread explained by the median: {spread[k]:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "curve_collapse.png", dpi=100)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for a_, (x, y, lab) in zip(
        ax,
        (
            ("total", "half-life Red", "units in total"),
            ("force ratio", "lanchester b (Blue's fire on Red)", "Blue / Red units"),
            ("n_red", "lanchester a (Red's fire on Blue)", "Red units"),
        ),
        strict=True,
    ):
        a_.scatter(P[x], P[y], s=5, alpha=0.4)
        a_.set_xlabel(lab)
        a_.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(FIG / "curve_shift.png", dpi=100)
    md = lambda t: "```\n" + t.to_string() + "\n```"
    text = [
        "# Curve shapes and what shifts them (descriptive, 26 Nov 2026)",
        "",
        "Script: `experiments/checks/curve_shapes.py`. Marine train battles (exp06) with one movement all through "
        f"({len(B)} battles), from first contact until one side has no units. Each fit uses the battle's whole future: "
        "this describes shapes, it is not a predictive model.",
        "",
        "## Which shape fits (R² per battle, k = free numbers per curve)",
        "",
        md(shape_tab),
        "",
        "Lanchester square (joint): two numbers a, b for both health curves together, start fixed.",
        "",
        "## What shifts the parameters between battles (5-fold CV R² across battles; 2% tails trimmed)",
        "",
        md(Sh.set_index("parameter").round(3)),
        "",
        "## One time scale",
        "",
        f"Health of both sides / start, against time and against time / the battle's own half-life: the median curve explains "
        f"{spread['t']:.2f} of the spread against time and {spread['s']:.2f} against scaled time. Half-lives:",
        "",
        md(th_tab),
        "",
        "Figures: `figures/curve_collapse.png`, `figures/curve_shift.png`.",
    ]
    (OUT / "curve_shapes.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
