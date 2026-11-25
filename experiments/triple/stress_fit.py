# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Stress tests of the pipeline's fitting (docs/triple/04_stress.md). Train data only.

uv run --extra explore experiments/triple/stress_fit.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]
from stress_core import SC, features, fit, head, load, spurious, sub, terms  # noqa: E402
from triple import table  # noqa: E402

from sindy_wm.models.intent_model import stlsq  # noqa: E402
from sindy_wm.models.pipeline import OTHER, SIDES, Layout, fit_gates, prepare, team_rows  # noqa: E402

# ---------------------------------------------------------------------------------------------------
head("1 constant term: can STLSQ keep it? (planted y = -0.18 R - c)")
rng = np.random.default_rng(0)
m = fit("marines", sub("marines", 300))
rows = sub("marines", 300)
names, T = features(m, rows, "blue", "marine")
for c in (0.0, 0.02, 0.1, 0.5, 5.0):
    y = -0.18 * T[:, 0] - c + rng.normal(0, 0.02, len(T))
    w = stlsq(T, y, 0.15)
    print(f"planted constant {-c:+.2f}: kept", {n: round(v, 3) for n, v in zip(names, w) if v})

head("2 planted truth: recover known laws from real features + residual-sized noise (mixed, 600 battles)")
rows = sub("mixed", 600)
m = fit("mixed", rows)
true = {
    "as fitted": lambda n: m.coef[("blue", "marine")].get(n, 0.0),
    "marauder term absent": lambda n: -0.178 if n == "R[marine]*s[marine]" else 0.0,
    "weak 5% cross term": lambda n: {
        "R[marine]*s[marine]": -0.178,
        "R[marauder]*s[marine]": -0.165,
        "R[marine]*s[marauder]": -0.009,
    }.get(n, 0.0),
    "strong cross term": lambda n: {
        "R[marine]*s[marine]": -0.178,
        "R[marauder]*s[marine]": -0.165,
        "R[marine]*s[marauder]": -0.08,
    }.get(n, 0.0),
    "alive instead of reach": lambda n: {"A[marine]": -0.1, "A[marauder]": -0.1}.get(n, 0.0),
    "weak firepower 0.03": lambda n: {"R[marine]*s[marine]": -0.178, "R[marauder]*s[marine]": -0.03}.get(
        n, 0.0
    ),
}
names, T = features(m, rows, "blue", "marine")
res = rows["d_health_blue_marine"].to_numpy() - T @ np.array([m.coef[("blue", "marine")][n] for n in names])
for lab, f in true.items():
    b = np.array([f(n) for n in names])
    ok, err = 0, []
    for i in range(20):
        y = T @ b + rng.permutation(res)
        w = stlsq(T, y, 0.15)
        ok += set(np.flatnonzero(w)) == set(np.flatnonzero(b))
        hit = (b != 0) & (w != 0)
        err.append(np.abs(w[hit] / b[hit] - 1).max() if hit.any() else np.nan)
    print(f"{lab:25s} exact support {ok}/20, worst relative coefficient error {np.nanmax(err):.3f}")

head("3 null: health rates shuffled across rows / battles -> terms kept (20 shuffles, all equations)")
for name in ("marines", "mixed", "triple"):
    rows = sub(name, 300)
    m = fit(name, rows)
    kept = 0
    for i in range(20):
        for s in SIDES:
            for t in m.L.types:
                _, T = features(m, rows, s, t)
                kept += (stlsq(T, rng.permutation(rows[f"d_health_{s}_{t}"].to_numpy()), 0.15) != 0).sum()
    print(name, "terms kept in 20 x all equations:", int(kept))

head("4 symmetries: swap sides (withdraw steps removed), swap type order")
rows = sub("mixed", 400)
nowd = rows[rows["withdraw"] == 0]
sw = nowd.rename(columns=lambda c: c.replace("blue", "#").replace("red", "blue").replace("#", "red"))
a, b = fit("mixed", nowd), fit("mixed", sw)
d = max(
    abs(a.coef[(s, t)][n] - b.coef[(OTHER[s], t)][n])
    for s in SIDES
    for t in a.L.types
    for n in a.coef[(s, t)]
)
print("side swap: max difference of health coefficients", f"{d:.2e}")
a, b = fit("mixed", rows), fit("mixed", rows, types=("MARAUDER", "MARINE"))
d = max(abs(a.coef[k][n] - b.coef[k][n]) for k in a.coef for n in a.coef[k])
dd = max(abs(x - y) for x, y in zip(a.dist_coef, b.dist_coef))
print(
    "type order: health",
    f"{d:.2e}",
    "distance",
    f"{dd:.2e}",
    "alive",
    f"{max(abs(a.p[k][0] - b.p[k][0]) for k in a.p):.2e}",
)
a, b = fit("mixed", rows), fit("mixed", rows)
print(
    "determinism (same data twice): identical",
    all(a.coef[k] == b.coef[k] for k in a.coef) and np.array_equal(a.dist_coef, b.dist_coef),
)
sh = rows.sample(frac=1, random_state=1)
b = fit("mixed", sh)
print(
    "row order shuffled: health",
    f"{max(abs(a.coef[k][n] - b.coef[k][n]) for k in a.coef for n in a.coef[k]):.2e}",
    "distance",
    np.round(np.abs(a.dist_coef - b.dist_coef).max(), 4),
)

head("5 threshold (health) and smoothing window: spurious / missing terms")
for name in ("marines", "mixed", "triple"):
    rows = sub(name, 500)
    print(
        name,
        "threshold:",
        {
            thr: spurious(fit(name, rows, thr_health=thr))
            for thr in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
        },
    )
for name in ("marines", "mixed", "triple"):
    e, ty, _ = SC[name]
    L = Layout(tuple(t.lower() for t in ty))
    tb = table(e, "train")
    tb = tb.loc[tb.index.get_level_values(0).unique()[:500]]
    out = {}
    for w in (3, 5, 9, 17, 33, 65):
        try:
            m = fit(name, prepare(tb, L, window=w))
            k = [abs(v) for v in terms(m).values()]
            out[w] = (*spurious(m), round(float(np.mean(k)), 3))
        except Exception as ex:
            out[w] = type(ex).__name__ + ": " + str(ex)[:60]
    print(name, "window -> (spurious, missing, mean |k|):", out)

head("6 how many battles are needed (5 draws each): draws with exact structure, coefficient spread, crashes")
for name in ("marines", "stalkers", "mixed", "triple"):
    full = fit(name, load(name))
    line = []
    for n in (3, 10, 30, 100, 300, 1000):
        ex, crash, rel = 0, [], []
        for i in range(5):
            try:
                m = fit(name, sub(name, n, seed=i))
                ex += spurious(m) == (0, 0)
                rel += [
                    abs(m.coef[(s, t)][k] / full.coef[(s, t)][k] - 1)
                    for (s, t, k) in terms(full)
                    if (s, t, k) in terms(m)
                ]
            except Exception as e2:
                crash.append(f"{type(e2).__name__}: {str(e2)[:50]}")
        line.append(
            f"{n}: {ex}/5 exact, max dev {max(rel) if rel else float('nan'):.2f}"
            + (f", crashes {len(crash)} ({crash[0]})" if crash else "")
        )
    print(name, "|", " | ".join(line), flush=True)

head("7 movement gates: 12 random starting points each (do they agree?)")
for name in ("marines", "stalkers", "mixed", "triple"):
    rows = sub(name, 600)
    L = Layout(tuple(t.lower() for t in SC[name][1]))
    Tt = team_rows(rows, L)
    m = fit(name, rows)
    base = fit_gates(Tt, m.gate_scale, skip=("withdraw",))
    from scipy.optimize import curve_fit

    from sindy_wm.models.intent_model import SHAPE, _close_gate

    for mv in ("advance", "hold"):
        a = Tt[Tt["movement"] == mv]
        a = a.sample(min(30_000, len(a)), random_state=0)
        X = tuple(a[c].to_numpy() for c in ["distance", *SHAPE])
        y = a["d_distance"].to_numpy()
        sse = lambda p: float(np.mean((_close_gate(X, *p) - y) ** 2))  # noqa: E731
        got = []
        for i in range(12):
            p0 = np.array(
                [
                    rng.uniform(2, 10),
                    rng.uniform(3, 14),
                    *rng.normal(0, 0.5, 4),
                    rng.uniform(0.1, 2),
                    rng.uniform(0, 1),
                ]
            )
            try:
                p, _ = curve_fit(_close_gate, X, y, p0=p0, maxfev=20000)
                got.append((round(sse(p), 3), round(p[0], 2), round(p[1], 1)))
            except Exception as e2:
                got.append((9.0, type(e2).__name__, 0))
        got.sort(key=lambda g: g[0])
        print(
            name,
            mv,
            "pipeline: mse",
            round(sse(base[mv]), 3),
            "v",
            round(base[mv][0], 2),
            "c",
            round(base[mv][1], 1),
            "| best random",
            got[0],
            "| median",
            got[6],
            "| worst",
            got[-1],
        )
    d = dict(zip(m.dist_names, m.dist_coef))
    print(
        "   distance:",
        {k: round(v, 2) for k, v in d.items() if v},
        "walk speeds:",
        [(t.lower()) for t in SC[name][1]],
        flush=True,
    )

head("8 missing or one-sided data")
rows = sub("mixed", 300)
cases = {
    "no withdraw": rows[rows["withdraw"] == 0],
    "no hold": rows[rows["hold"] == 0],
    "no advance": rows[rows["advance"] == 0],
    "only advance": rows[rows["advance"] == 1],
    "no focus": rows[rows["focus"] == 0],
    "always focus": rows[rows["focus"] == 1],
    "only far (no unit in reach)": rows[(rows[[c for c in rows if c.startswith("reach_")]].sum(axis=1) == 0)],
    "NaN in one health rate": rows.assign(
        d_health_blue_marine=rows["d_health_blue_marine"].where(np.arange(len(rows)) != 5)
    ),
    "inf in distance": rows.assign(distance=rows["distance"].where(np.arange(len(rows)) != 5, np.inf)),
    "NaN in one reach": rows.assign(
        reach_red_marine=rows["reach_red_marine"].where(np.arange(len(rows)) != 5)
    ),
    "every row twice": pd.concat([rows, rows]),
    "marauders absent on Red": rows.assign(
        **{
            c: 0.0
            for c in rows
            if c.endswith("red_marauder")
            or c == "fire_blue_marine_marauder"
            or c == "fire_blue_marauder_marauder"
        }
    ),
    "empty table": rows.iloc[:0],
    "float32": rows.astype({c: "float32" for c in rows.select_dtypes("float64").columns}),
}
ref = fit("mixed", rows)
for lab, rr in cases.items():
    try:
        m = fit("mixed", rr)
        bad = [k for k, v in terms(m).items() if not np.isfinite(v)]
        nan = not np.isfinite(list(m.coefficients().values())).all()
        dev = [
            abs(m.coef[(s, t)][k] / ref.coef[(s, t)][k] - 1)
            for (s, t, k) in terms(ref)
            if m.coef[(s, t)].get(k)
        ]
        print(
            f"{lab:30s} fits; spurious/missing {spurious(m)}, non-finite {nan}, max dev {max(dev) if dev else float('nan'):.3f}, p_focus {[round(v[1], 2) for v in m.p.values()]}, distance { {k: round(float(v), 1) for k, v in zip(m.dist_names, m.dist_coef) if v} }"
        )
    except Exception as e2:
        print(f"{lab:30s} RAISES {type(e2).__name__}: {str(e2)[:90]}")

head("9 measurement noise on the states before differentiation (mixed, 400 battles)")
L = Layout(("marine", "marauder"))
tb = table("exp08_mixed", "train")
tb = tb.loc[tb.index.get_level_values(0).unique()[:400]]
ref = fit("mixed", prepare(tb, L))
for sd in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
    t2 = tb.copy()
    for c in [c for c in tb if c.startswith(("health_",))]:
        t2[c] = np.maximum(t2[c] + rng.normal(0, sd, len(t2)), 0)
    m = fit("mixed", prepare(t2, L))
    k = [abs(v) / abs(ref.coef[(s, t)][n]) for (s, t, n), v in terms(m).items() if ref.coef[(s, t)].get(n)]
    print(
        f"health noise sd {sd}: spurious/missing {spurious(m)}, coefficient ratio {min(k) if k else 0:.2f}-{max(k) if k else 0:.2f}, p {[round(v[0], 2) for v in m.p.values()]}"
    )
for sd in (0.02, 0.05, 0.1):
    t2 = tb.copy()
    for c in [c for c in tb if c.startswith("reach_")]:
        t2[c] = np.maximum(t2[c] + rng.normal(0, sd, len(t2)), 0)
    m = fit("mixed", prepare(t2, L))
    k = [abs(v) / abs(ref.coef[(s, t)][n]) for (s, t, n), v in terms(m).items() if ref.coef[(s, t)].get(n)]
    print(
        f"reach noise sd {sd} (= {10 * sd:.1f} units): spurious/missing {spurious(m)}, coefficient ratio {min(k) if k else 0:.2f}-{max(k) if k else 0:.2f}"
    )

head("10 wrong time step: every second recorded step kept (dt 0.25 instead of 0.125)")
tb = table("exp06_intents", "train")
tb = tb.loc[tb.index.get_level_values(0).unique()[:400]]
L = Layout(("marine",))
from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402

half = tb[tb.index.get_level_values(1) % 2 == 0]
for lab, t2 in (("dt 0.125", tb), ("dt 0.25", half)):
    try:
        m = fit("marines", prepare(t2, L))
        B = L.batch(add_controls(t2))
        with np.errstate(all="ignore"):
            Y = L.team_view(m.rollout(B))
        h = H.horizon_metrics(Y, B["team"])
        print(
            lab,
            "k",
            [round(v, 3) for v in terms(m).values()],
            "health error labelled 1 s / 2 s:",
            round(h["health_1s"], 4),
            round(h["health_2s"], 4),
        )
    except ValueError as e2:
        print(lab, "REFUSED:", str(e2)[:120])

head("11 duplicate episode ids: two battles merged under one id")
two = tb.loc[tb.index.get_level_values(0).unique()[:200]]
a = two.reset_index()
a["episode_idx"] = a["episode_idx"] % 100
b = a.set_index(["episode_idx", "step"])
try:
    m0, m1 = fit("marines", prepare(two, L)), fit("marines", prepare(b, L))
    print("separate", terms(m0).values(), m0.p, "| merged", terms(m1).values(), m1.p)
except Exception as e2:
    print("RAISES", type(e2).__name__, str(e2)[:100])

head("12 battles dropped silently by the smoothing window, and health that rises")
for name in ("marines", "stalkers", "mixed", "triple"):
    e, ty, _ = SC[name]
    tb = table(e, "train")
    n = tb.groupby(level=0).size()
    line = f"{name}: battles {len(n)}, dropped (<= 9 steps) {(n <= 9).sum()}"
    for t in ty:
        for s in SIDES:
            dh = tb[f"health_{s}_{t.lower()}"].groupby(level=0).diff()
            line += f" | {s} {t.lower()} rises in {(dh > 1e-9).mean():.1%} of steps"
    print(line)
print("\nDONE")
