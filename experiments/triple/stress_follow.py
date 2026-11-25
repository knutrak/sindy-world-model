# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Follow-ups on the stress tests (docs/triple/04_stress.md): do the STLSQ defects touch the fitted models, how
stable is the triple structure, stalker closing speed, shield regeneration. Train data only.

    uv run --extra explore experiments/triple/stress_follow.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]
import stress_core as C  # noqa: E402
from triple import table  # noqa: E402

from sindy_wm.models.intent_model import stlsq  # noqa: E402
from sindy_wm.models.pipeline import SIDES  # noqa: E402


def stlsq_fixed(theta, y, threshold, alpha=1e-6, max_iter=25, trace=None):
    """As `stlsq`, with a constant column scaled by its size and zero returned when nothing survives."""
    sd = theta.std(axis=0)
    const = sd < 1e-9 * (np.abs(theta).max(axis=0) + 1e-300)
    scale = np.where(const, np.maximum(np.abs(theta).max(axis=0), 1e-12), sd + 1e-12)
    T, ys = theta / scale, y.std()
    keep, coef = np.ones(T.shape[1], bool), np.zeros(T.shape[1])
    for _ in range(max_iter):
        coef[:] = 0
        if not keep.any():
            break
        A = T[:, keep]
        coef[keep] = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ y)
        new = np.abs(coef) >= threshold * ys
        if trace is not None:
            trace.append((np.abs(coef) / ys, keep.copy()))
        if (new == keep).all():
            break
        keep = new
    return coef / scale


C.head("A do the two defects change the real fits? (all train battles)")
for name in ("marines", "marauders", "stalkers", "mixed", "triple"):
    rows = C.load(name)
    m = C.fit(name, rows)
    for s in SIDES:
        for t in m.L.types:
            names, T = C.features(m, rows, s, t)
            y = rows[f"d_health_{s}_{t}"].to_numpy()
            a, b = stlsq(T, y, 0.15), stlsq_fixed(T, y, 0.15)
            sup = [n for n, v in zip(names, b) if v]
            j = np.flatnonzero(a)
            X = np.column_stack([T[:, j], np.ones(len(T))])
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            print(
                f"{name:9s} {s:4s} {t:8s} same terms {set(np.flatnonzero(a)) == set(np.flatnonzero(b))}"
                f" | max rel diff {np.abs(b[j] / a[j] - 1).max():.4f} | intercept if added by hand {c[-1]:+.4f}"
                f" (effect size {abs(c[-1]) / y.std():.3f}, threshold 0.15), k changes "
                + ", ".join(f"{v:+.1%}" for v in c[:-1] / a[j] - 1)
            )

C.head("B bootstrap of the term selection by battle (60 resamples): share of resamples keeping each term")
rng = np.random.default_rng(0)
for name in ("marines", "marauders", "stalkers", "mixed", "triple"):
    rows = C.load(name)
    m = C.fit(name, rows)
    eps = rows.index.get_level_values(0).to_numpy()
    uniq, inv = np.unique(eps, return_inverse=True)
    exact, worst, extra = 0, [], 0
    keepcount, spur = {}, {}
    F = {(s, t): C.features(m, rows, s, t) for s in SIDES for t in m.L.types}
    # Gram matrices per battle make resampling cheap
    for (s, t), (names, T) in F.items():
        y = rows[f"d_health_{s}_{t}"].to_numpy()
        Z = np.column_stack([T, y, np.ones(len(T))])
        p = Z.shape[1]
        G = np.zeros((len(uniq), p, p))
        for i in range(p):
            for j in range(p):
                G[:, i, j] = np.bincount(inv, weights=Z[:, i] * Z[:, j], minlength=len(uniq))
        F[(s, t)] = (names, G)
    for _ in range(60):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        ok = True
        for (s, t), (names, G) in F.items():
            g = np.tensordot(w, G, 1)
            n = g[-1, -1]
            k = len(names)
            mean, sq = g[-1, : k + 1] / n, np.diag(g)[: k + 1] / n
            sd = np.sqrt(np.maximum(sq - mean**2, 0)) + 1e-12
            sd[:k] = np.where(np.array(names) == "1", 1.0, sd[:k])  # the constant, as in the fixed stlsq
            A, b = g[:k, :k] / np.outer(sd[:k], sd[:k]), g[:k, k] / sd[:k]
            keep, coef = np.ones(k, bool), np.zeros(k)
            for _ in range(25):
                coef[:] = 0
                if not keep.any():
                    break
                idx = np.flatnonzero(keep)
                c = np.linalg.solve(A[np.ix_(idx, idx)] + 1e-6 * n * np.eye(len(idx)), b[idx])
                coef[:] = 0
                coef[idx] = c
                new = np.abs(coef) >= 0.15 * sd[k]
                if (new == keep).all():
                    break
                keep = new
            got = {names[i] for i in np.flatnonzero(coef)}
            want = {f"R[{a}]*s[{t}]" for a in m.L.types}
            for x in want:
                keepcount[(s, t, x)] = keepcount.get((s, t, x), 0) + (x in got)
            for x in got - want:
                spur[(s, t, x)] = spur.get((s, t, x), 0) + 1
            ok &= got == want
        exact += ok
    low = {k: round(v / 60, 2) for k, v in keepcount.items() if v < 60}
    print(
        f"{name:9s} exact structure in {exact}/60; terms kept in < 100%: {low or 'none'}; spurious: {spur or 'none'}"
    )

C.head("C effect size of each kept term (|coef| sd(x) / sd(y); the threshold is 0.15)")
for name in ("mixed", "triple"):
    rows = C.load(name)
    m = C.fit(name, rows)
    out = {}
    for s in SIDES:
        for t in m.L.types:
            names, T = C.features(m, rows, s, t)
            y = rows[f"d_health_{s}_{t}"].to_numpy()
            for n, v in m.coef[(s, t)].items():
                if v:
                    out[f"{s} {n}"] = round(abs(v) * T[:, names.index(n)].std() / y.std(), 3)
    print(name, dict(sorted(out.items(), key=lambda z: z[1])))

C.head("D closing speed under advance by distance (mean d' of battles' steps, map units / s)")
for name, v in (("marines", 3.15), ("marauders", 3.15), ("stalkers", 4.13), ("triple", 0)):
    r = C.load(name)
    a = r[r["movement"] == "advance"]
    b = pd.cut(a["distance"], [0, 5, 6, 7, 8, 9, 10, 12, 14, 18])
    g = a.groupby(b, observed=True)["d_distance"].agg(["mean", "count"])
    print(
        name,
        "2 x walk speed",
        2 * v,
        {str(k): (round(x["mean"], 2), int(x["count"])) for k, x in g.iterrows()},
    )

C.head(
    "E the same at the unit level: do both sides walk at full speed? (exp07, 100 battles, distance > 10, advance)"
)
from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import RAW_DIR  # noqa: E402

for exp, sets, spd in (("exp06_intents", "train_0", 3.15), ("exp07_stalkers", "train", 4.13)):
    d = RAW_DIR / exp / sets
    if not d.exists():
        d = next(p for p in (RAW_DIR / exp).iterdir() if p.name.startswith("train"))
    sn = storage.load_snapshots(d, list(range(100)))
    ct = storage.load_controls(d).set_index(["episode_idx", "step"])["movement"]
    sn = sn[sn["alive"]].join(ct, on=["episode_idx", "step"])
    c = sn.groupby(["episode_idx", "step", "team"])[["x", "y"]].mean().unstack("team")
    dist = np.hypot(c[("x", "blue")] - c[("x", "red")], c[("y", "blue")] - c[("y", "red")]).rename("D")
    sn = sn.join(dist, on=["episode_idx", "step"])
    q = sn[(sn["D"] > 10) & (sn["movement"] == "advance") & (sn["step"] > 2)]
    q = q.assign(v=np.hypot(q["vx"], q["vy"]))
    print(
        exp,
        "walk speed",
        spd,
        {
            t: (round(g["v"].mean(), 2), round((g["v"] > 0.9 * spd).mean(), 2), len(g))
            for t, g in q.groupby("team")
        },
        "(mean speed, share at full speed, unit-steps)",
    )

C.head("F shield regeneration: when does stalker health rise, and an own-alive term")
for name, t in (("stalkers", "stalker"), ("triple", "stalker")):
    tb = table(C.SC[name][0], "train")
    for s in SIDES:
        h, al = tb[f"health_{s}_{t}"], tb[f"alive_{s}_{t}"]
        dh = h.groupby(level=0).diff().fillna(0)
        hit = (dh < -1e-9).groupby(level=0).cummax()
        up = dh > 1e-9
        print(
            name,
            s,
            f"rises: before the side's first damage {(up & ~hit).sum() / up.sum():.0%}; after {(up & hit).sum() / up.sum():.0%}",
            f"| health gained / lost over battles {dh[dh > 0].sum() / -dh[dh < 0].sum():.1%}",
            f"(of which after first damage {dh[(dh > 0) & hit].sum() / -dh[dh < 0].sum():.1%})",
        )
    rows = C.load(name)
    m = C.fit(name, rows)
    for s in SIDES:
        names, T = C.features(m, rows, s, t)
        X = np.column_stack([T, rows[f"alive_{s}_{t}"]])
        w = stlsq_fixed(X, rows[f"d_health_{s}_{t}"].to_numpy(), 0.15)
        j = [i for i in range(len(names)) if names[i].endswith(f"s[{t}]") and names[i].startswith("R[")] + [
            len(names)
        ]
        y = rows[f"d_health_{s}_{t}"].to_numpy()
        ls = np.linalg.lstsq(X[:, j], y, rcond=None)[0]
        print(
            "   own alive as candidate kept:",
            bool(w[-1]),
            "| by hand",
            round(float(ls[-1]), 4),
            "(game while regenerating 0.0125), firepower changes",
            np.round(ls[:-1] / np.array([m.coef[(s, t)][names[i]] for i in j[:-1]]) - 1, 3),
        )
print("DONE")
