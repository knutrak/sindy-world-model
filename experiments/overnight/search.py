# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 1 (docs/overnight/00_plan.md): variants scored on left-out unit types, train battles only.

python experiments/overnight/search.py <tag> '<json list of [name, opt]>' [base name]   appends to search.csv / errors
"""

import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit as F
import numpy as np
import pandas as pd

ALL = ("marines", "marauders", "stalkers", "mixed", "triple", *F.VARIANTS)
FOLDS = {
    "marauders": (("marines", "stalkers", *F.VARIANTS), ("marauders", "mixed")),
    "stalkers": (("marines", "marauders", "mixed", *F.VARIANTS), ("stalkers", "triple")),
    "range": (tuple(a for a in ALL if a not in ("R4", "R3")), ("R4", "R3")),
    "deadly": (tuple(a for a in ALL if a not in ("DP", "CM", "HM", "SP")), ("DP", "CM", "HM", "SP")),
}
OUT = F.CACHE / "search"


def errors(m, B, placebo=False):
    if placebo:
        B = dict(B)
        i = np.random.default_rng(0).permutation(len(B["X0"]))
        for k in ("mu0", "sd0"):
            if k in B and m.needs_margins:
                B[k] = B[k][i]
        if m.needs_vd:
            B["vd0"] = B["vd0"][i]
    with np.errstate(all="ignore"):
        Y = m.L.team_view(m.rollout({**B, "U": B["U"][:, :32]}))
    T, out = B["team"], {"ep": B["ep"]}
    for h in (1, 2, 4):
        n = h * 8
        out[f"e{h}"] = np.abs(Y[:, n, :2] - T["Y"][:, n, :2]).mean(1)
        real = (T["Y"][:, n, 0] - T["Y"][:, n, 1]) - (T["X0"][:, 0] - T["X0"][:, 1])
        out[f"m{h}"] = (real - ((Y[:, n, 0] - Y[:, n, 1]) - (T["X0"][:, 0] - T["X0"][:, 1]))) ** 2
        out[f"v{h}"] = real
    out["d2"] = np.abs(Y[:, 16, 4] - T["Y"][:, 16, 4])
    out["a2"] = 10 * np.abs(Y[:, 16, 2:4] - T["Y"][:, 16, 2:4]).mean(1)
    bad = ~np.isfinite(Y).all((1, 2)) | (np.abs(Y[:, :, 4]) > 60).any(1)
    out["bad"] = bad.astype(float)
    return out


def job(a):
    name, opt, fold, placebo = a
    t0 = time.time()
    par = F.fit(F.pool_of(FOLDS[fold][0]), opt)
    res = {arm: errors(F.model(par, arm), F.army(arm)["B"], placebo) for arm in FOLDS[fold][1]}
    return name, opt, fold, placebo, par, res, time.time() - t0


def compare(res, base):
    """Ratios of mean 2 s error per army to the base, per fold; pooled paired CI on base-normalised differences."""
    rng, folds, diffs = np.random.default_rng(0), {}, []
    for fold in FOLDS:
        r = []
        for arm in FOLDS[fold][1]:
            a, b = res[fold][arm], base[fold][arm]
            ok = np.isfinite(a["e2"]) & np.isfinite(b["e2"])
            s = b["e2"][ok].mean()
            r.append(a["e2"][ok].mean() / s)
            g = (
                pd.DataFrame({"d": (a["e2"][ok] - b["e2"][ok]) / s, "ep": a["ep"][ok]})
                .groupby("ep")["d"]
                .agg(["sum", "count"])
                .to_numpy()
            )
            i = rng.integers(0, len(g), (1000, len(g)))
            diffs.append(g[i, 0].sum(1) / g[i, 1].sum(1))
        folds[fold] = float(np.mean(r))
    d = np.mean(diffs, axis=0)
    return folds, float(d.mean()), [float(x) for x in np.percentile(d, [2.5, 97.5])]


def r2(res, arm_fold):
    f, arm = arm_fold
    x = res[f][arm]
    ok = np.isfinite(x["m2"]) & np.isfinite(x["v2"])
    return 1 - x["m2"][ok].sum() / ((x["v2"][ok] - x["v2"][ok].mean()) ** 2).sum()


def run(tag, variants, base_name):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, opt in variants:
        opt = {k: tuple(v) if isinstance(v, list) else v for k, v in opt.items()}
        for p in (False, True) if ("marg" in opt or "vd" in opt) else (False,):
            jobs += [(name, opt, f, p) for f in FOLDS]
    got = {}
    with Pool(8) as pool:
        for name, opt, fold, p, par, res, s in pool.imap_unordered(job, jobs):
            key = name + ("|placebo" if p else "")
            got.setdefault(key, {"opt": opt, "par": {}, "res": {}, "s": 0.0})
            got[key]["par"][fold], got[key]["res"][fold] = par, res
            got[key]["s"] += s
    for k, g in got.items():
        pickle.dump(g, open(OUT / f"{k.replace('|', '__')}.pkl", "wb"))
    base = pickle.load(open(OUT / f"{base_name}.pkl", "rb"))["res"]
    rows = []
    for k, g in got.items():
        folds, d, ci = compare(g["res"], base)
        bad = max(float(g["res"][f][a]["bad"].mean()) for f in FOLDS for a in FOLDS[f][1])
        mr = np.mean([r2(g["res"], (f, a)) for f in FOLDS for a in FOLDS[f][1]])
        e4 = np.mean(
            [
                np.nanmean(g["res"][f][a]["e4"]) / np.nanmean(base[f][a]["e4"])
                for f in FOLDS
                for a in FOLDS[f][1]
            ]
        )
        d2 = np.mean(
            [
                np.nanmean(g["res"][f][a]["d2"]) / np.nanmean(base[f][a]["d2"])
                for f in FOLDS
                for a in FOLDS[f][1]
            ]
        )
        rows.append(
            {
                "tag": tag,
                "variant": k,
                "base": base_name,
                **{f"{f} ratio": v for f, v in folds.items()},
                "pooled diff": d,
                "ci lo": ci[0],
                "ci hi": ci[1],
                "gate": sum(v < 1 for v in folds.values()) >= 3 and ci[1] < 0 and bad <= 0.005,
                "e4 ratio": e4,
                "mean margin R2": mr,
                "d2 ratio": d2,
                "max bad": bad,
                "terms": F.complexity(g["par"]["range"]),
                "cpu s": g["s"],
                "opt": json.dumps(g["opt"]),
            }
        )
    t = pd.DataFrame(rows)
    p = OUT / "search.csv"
    t.to_csv(p, mode="a", header=not p.exists(), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.drop(columns=["tag", "opt", "base"]).round(4).to_string())


if __name__ == "__main__":
    run(
        sys.argv[1],
        json.loads(sys.argv[2]),
        sys.argv[3] if len(sys.argv) > 3 else json.loads(sys.argv[2])[0][0],
    )
