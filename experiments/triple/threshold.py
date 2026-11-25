# ruff: noqa: E501, E731, B007, B023, B905, I001
"""What limits the stability of the triple model's structure: threshold, data or signal? Train only, exploratory.

uv run --extra explore experiments/triple/threshold.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

sys.argv = [sys.argv[0]]
import stress_core as C  # noqa: E402
from sindy_wm.models.pipeline import SIDES  # noqa: E402

NAMES = ("marines", "marauders", "stalkers", "mixed", "triple")
THR = (0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25)


def grams(name):
    """Per-battle Gram matrices of [candidates, y, 1] for every health equation."""
    rows = C.load(name)
    m = C.fit(name, rows)
    eps = rows.index.get_level_values(0).to_numpy()
    uniq, inv = np.unique(eps, return_inverse=True)
    out = {}
    for s in SIDES:
        for t in m.L.types:
            names, T = C.features(m, rows, s, t)
            Z = np.column_stack([T, rows[f"d_health_{s}_{t}"].to_numpy(), np.ones(len(T))])
            p = Z.shape[1]
            G = np.array(
                [
                    [np.bincount(inv, weights=Z[:, i] * Z[:, j], minlength=len(uniq)) for j in range(p)]
                    for i in range(p)
                ]
            )
            out[(s, t)] = (names, np.moveaxis(G, 2, 0))
    return m, len(uniq), out


def scaled(g, names):
    k, n = len(names), g[-1, -1]
    mean, sq = g[-1, : k + 1] / n, np.diag(g)[: k + 1] / n
    sd = np.sqrt(np.maximum(sq - mean**2, 0)) + 1e-12
    sd[:k] = np.where(np.array(names) == "1", 1.0, sd[:k])
    return g[:k, :k] / np.outer(sd[:k], sd[:k]), g[:k, k] / sd[:k], sd, n


def select(A, b, sd, n, thr):
    k = len(b)
    keep, coef = np.ones(k, bool), np.zeros(k)
    for _ in range(25):
        coef[:] = 0
        if not keep.any():
            break
        i = np.flatnonzero(keep)
        coef[i] = np.linalg.solve(A[np.ix_(i, i)] + 1e-6 * n * np.eye(len(i)), b[i])
        new = np.abs(coef) >= thr * sd[k]
        if (new == keep).all():
            break
        keep = new
    return coef


def main():
    data = {n: grams(n) for n in NAMES}
    print(
        "### 1 effect sizes: smallest real term (fitted alone with the other real terms) and largest spurious candidate (full library, step 1)"
    )
    for name, (m, nb, F) in data.items():
        real, spur = [], []
        for (s, t), (names, G) in F.items():
            A, b, sd, n = scaled(G.sum(0), names)
            want = np.array([x in {f"R[{a}]*s[{t}]" for a in m.L.types} for x in names])
            i = np.flatnonzero(want)
            c = np.linalg.solve(A[np.ix_(i, i)], b[i])
            real += [
                (abs(v) / sd[-1] if False else abs(v) / sd[len(names)], s, names[j]) for v, j in zip(c, i)
            ]
            full = np.linalg.solve(A + 1e-6 * n * np.eye(len(names)), b)
            spur += [(abs(full[j]) / sd[len(names)], s, t, names[j]) for j in np.flatnonzero(~want)]
        print(
            f"{name:9s} smallest real {min(real)[0]:.3f} ({min(real)[1]} {min(real)[2]}) | largest spurious {max(spur)[0]:.3f} {max(spur)[1:]}"
        )
    print(
        "\n### 2 exact structure in 60 battle bootstraps, by threshold: exact% (runs with a missing / with a spurious term)"
    )
    for name, (m, nb, F) in data.items():
        rng = np.random.default_rng(0)
        W = [np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(60)]
        line = []
        S = {k: [scaled(np.tensordot(w, G, 1), names) for w in W] for k, (names, G) in F.items()}
        for thr in THR:
            ex = mi = sp = 0
            for j in range(60):
                miss = extra = False
                for (s, t), (names, _) in F.items():
                    got = {names[i] for i in np.flatnonzero(select(*S[(s, t)][j], thr))}
                    want = {f"R[{a}]*s[{t}]" for a in m.L.types}
                    miss |= bool(want - got)
                    extra |= bool(got - want)
                ex, mi, sp = ex + (not miss and not extra), mi + miss, sp + extra
            line.append(f"{thr:.2f}: {ex / 60:.0%} ({mi}/{sp})")
        print(f"{name:9s}", "  ".join(line))
    print(
        "\n### 3 triple: the weakest term (blue stalker <- marine) by number of battles: effect size mean ± sd over 200 bootstraps, k mean ± sd, share kept at 0.15"
    )
    m, nb, F = data["triple"]
    for key, term in (
        (("blue", "stalker"), "R[marine]*s[stalker]"),
        (("red", "stalker"), "R[marine]*s[stalker]"),
    ):
        names, G = F[key]
        j = names.index(term)
        want = [names.index(f"R[{a}]*s[stalker]") for a in m.L.types]
        for size in (190, 375, 750, 1500, 3000, 6000):
            rng = np.random.default_rng(1)
            e, k, kept = [], [], 0
            for _ in range(200):
                A, b, sd, n = scaled(
                    np.tensordot(np.bincount(rng.integers(0, nb, size), minlength=nb), G, 1), names
                )
                c = np.linalg.solve(A[np.ix_(want, want)], b[want])
                v = c[want.index(j)]
                e.append(abs(v) / sd[len(names)])
                k.append(v / sd[j])
                kept += select(A, b, sd, n, 0.15)[j] != 0
            note = (
                ""
                if size <= nb
                else "  (resampled beyond the data: what more battles would give if these are typical)"
            )
            print(
                f"{key[0]} {size:5d} battles: effect {np.mean(e):.3f} ± {np.std(e):.3f}; k {np.mean(k):+.4f} ± {np.std(k):.4f} ({abs(np.mean(k) / np.std(k)):.0f} sd from 0); kept {kept / 200:.0%}{note}"
            )
    print("\n### 4 why the effect size is small: effect = |k| sd(x) / sd(y)")
    for name, key, term in (
        ("marines", ("blue", "marine"), "R[marine]*s[marine]"),
        ("mixed", ("blue", "marauder"), "R[marine]*s[marauder]"),
        ("triple", ("blue", "marauder"), "R[marine]*s[marauder]"),
        ("triple", ("blue", "stalker"), "R[marine]*s[stalker]"),
        ("triple", ("blue", "stalker"), "R[marauder]*s[stalker]"),
    ):
        m, nb, F = data[name]
        names, G = F[key]
        _, _, sd, n = scaled(G.sum(0), names)
        kk = m.coef[key][term]
        g = G.sum(0)
        print(
            f"{name:8s} {key[0]} {term:24s} |k| {abs(kk):.3f}  sd(x) {sd[names.index(term)]:.4f}  mean(x) {g[-1, names.index(term)] / n:.4f}  sd(y) {sd[len(names)]:.4f}  effect {abs(kk) * sd[names.index(term)] / sd[len(names)]:.3f}"
        )
    print(
        "\n### 5 out-of-sample cost of dropping the weakest term (triple, one-step R² on val, blue stalker)"
    )
    from triple import rates

    L = m.L if (m := data["triple"][0]) else None
    vr, tr = rates("exp12_triple", "val", L), C.load("triple")
    for s in SIDES:
        nm, T = C.features(m, tr, s, "stalker")
        _, V = C.features(m, vr, s, "stalker")
        yt, yv = tr[f"d_health_{s}_stalker"].to_numpy(), vr[f"d_health_{s}_stalker"].to_numpy()
        for lab, drop in (
            ("all three", ()),
            ("without marine", ("marine",)),
            ("without stalker", ("stalker",)),
        ):
            i = [nm.index(f"R[{a}]*s[stalker]") for a in m.L.types if a not in drop]
            c = np.linalg.lstsq(T[:, i], yt, rcond=None)[0]
            print(
                s,
                f"{lab:16s} val R² {1 - ((yv - V[:, i] @ c) ** 2).sum() / ((yv - yv.mean()) ** 2).sum():.4f}  k",
                np.round(c, 3),
            )


if __name__ == "__main__":
    main()
