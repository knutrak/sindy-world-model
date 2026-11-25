# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 3: the blind test (docs/overnight/00_plan.md). build, then analyse once -> docs/overnight/02_blind_test.md"""

import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
for p in ("experiments/overnight", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import fit as F
import numpy as np
import pandas as pd
import prep
import search as Se
from build_typed_states import build_one

from sindy_wm.evaluation import harness as H
from sindy_wm.models.intent_model import SHAPE, add_controls, add_rates
from sindy_wm.models.param import Param
from sindy_wm.models.pipeline import SIDES, Layout, team_rows

OUT, DOC = prep.PROC / "exp14_blind_typed", ROOT / "docs" / "overnight" / "02_blind_test.md"
TYPES = {
    "marines": ("MARINE",),
    "marauders": ("MARAUDER",),
    "stalkers": ("STALKER",),
    "mixed": ("MARINE", "MARAUDER"),
    "triple": ("MARINE", "MARAUDER", "STALKER"),
    "n1": ("MARINE_N1",),
    "n2": ("MARINE_N2",),
    "n3": ("MARINE_N3",),
}
WR = {
    "marines": 7.0,
    "marauders": 7 * 7.125 / 5.75,
    "stalkers": 7 * 7.25 / 5.75,
    "mixed": 8.0,
    "triple": 8.0,
    "n1": 7.0,
    "n2": 7.0,
    "n3": 7.0,
}
FIN = ("H5", "I7")


def build(n):
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / f"{n}.parquet").exists():
        build_one(("exp14_blind", n, TYPES[n])).to_parquet(OUT / f"{n}.parquet")
    return n


def ci(a, b, ep, n=1000):
    ok = np.isfinite(a) & np.isfinite(b)
    g = pd.DataFrame({"d": a[ok] - b[ok], "ep": ep[ok]}).groupby("ep")["d"].agg(["sum", "count"]).to_numpy()
    i = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
    return np.percentile(g[i, 0].sum(1) / g[i, 1].sum(1), [2.5, 97.5])


def r2(x, h=2):
    ok = np.isfinite(x[f"m{h}"]) & np.isfinite(x[f"v{h}"])
    return 1 - x[f"m{h}"][ok].sum() / ((x[f"v{h}"][ok] - x[f"v{h}"][ok].mean()) ** 2).sum()


def analyse():
    assert not DOC.exists(), "already run"
    fin = pickle.load(open(F.CACHE / "finalists.pkl", "rb"))
    frozen = pickle.load(open(H.CACHE / "generalisation_frozen.pkl", "rb"))
    own = {k: frozen[k] for k in ("marines", "marauders", "mixed")}
    own["triple"] = pickle.load(open(prep.PROC / "exp12_triple_typed" / "triple_model.pkl", "rb"))
    own["stalkers"] = pickle.load(open(prep.PROC / "exp07_stalkers_typed" / "stalker_model.pkl", "rb"))
    E, rows = {}, []
    for n, types in TYPES.items():
        st = pd.read_parquet(OUT / f"{n}.parquet")
        st["vd"] = st.groupby(level=0)["distance"].diff().fillna(0.0) / 0.125
        st = add_controls(st)
        L = Layout(tuple(t.lower() for t in types))
        models = {f"{k} {w}": Param(types, WR[n], par) for (k, w), par in fin.items()}
        if n in own:
            for a in ("needs_margins", "needs_vd"):
                setattr(own[n], a, False)
            models["B0 own"] = own[n]
        ids = st.index.get_level_values(0).unique()
        if n == "n2":  # T4: a refit on its first 500 battles, all models scored on the other 500
            r = add_rates(st.loc[ids[:500]], cols=(*L.x_cols, *SHAPE), window=9)
            al = {s: r[f"alive_{s}_{L.types[0]}"] for s in SIDES}
            r = team_rows(r[(al["blue"] > 0) & (al["red"] > 0)], L)
            F._mem["n2"] = {"rows": r, "types": types, "wr": WR[n]}
            for k in FIN:
                models[f"{k} refit"] = Param(types, WR[n], F.fit({"n2": 60_000}, fin[(k, "all")]["opt"]))
        B = prep.batch(L, st.loc[ids[500:]] if n == "n2" else st)
        E[n] = {k: Se.errors(m, B) for k, m in models.items()}
        X = np.repeat(B["team"]["X0"][:, None], 17, 1)
        pers = float(np.nanmean(np.abs(X[:, 16, :2] - B["team"]["Y"][:, 16, :2])))
        for k, x in E[n].items():
            rows.append(
                {
                    "army": n,
                    "model": k,
                    "e1": np.nanmean(x["e1"]),
                    "e2": np.nanmean(x["e2"]),
                    "e4": np.nanmean(x["e4"]),
                    "margin R2 2s": r2(x),
                    "margin R2 4s": r2(x, 4),
                    "d2": np.nanmean(x["d2"]),
                    "a2": np.nanmean(x["a2"]),
                    "bad": x["bad"].mean(),
                    "persistence e2": pers,
                }
            )
    t = pd.DataFrame(rows)
    ev = lambda n, k: np.nanmean(E[n][k]["e2"])  # noqa: E731
    better = lambda n, a, b: ci(E[n][a]["e2"], E[n][b]["e2"], E[n][a]["ep"])  # noqa: E731
    L = [
        "# Blind test (exp14, read once)",
        "",
        "Plan: `00_plan.md`. Script: `experiments/overnight/blind.py`.",
        "",
        "## Rules",
        "",
    ]
    for k in FIN:
        q = {n: better(n, f"{k} LO", "B1 LO") for n in ("marauders", "stalkers", "mixed", "triple")}
        v = {n: better(n, f"{k} all", "B1 all") for n in ("n1", "n2", "n3")}
        t3 = {n: ev(n, f"{k} all") / ev(n, "B0 own") for n in ("marines", "marauders", "mixed", "triple")}
        t4 = ev("n2", f"{k} all") / ev("n2", f"{k} refit")
        t5 = {n: r2(E[n][f"{k} all"]) - r2(E[n]["B1 all"]) for n in TYPES}
        s = lambda d: ", ".join(f"{n} [{x[0]:+.4f}, {x[1]:+.4f}]" for n, x in d.items())  # noqa: E731
        L += [
            f"- **{k} T1 (primary)**: {sum(x[1] < 0 for x in q.values())} of 4 (needs 3): {s(q)}",
            f"- {k} T2: {sum(x[1] < 0 for x in v.values())} of 3 (needs 2): {s(v)}",
            f"- {k} T3: worst {max(t3.values()):.3f} (limit 1.05): { ({n: round(x, 3) for n, x in t3.items()}) }",
            f"- {k} T4: {t4:.3f} (limit 1.10)",
            f"- {k} T5: lowest difference {min(t5.values()):+.3f} (needs >= 0): { ({n: round(x, 3) for n, x in t5.items()}) }",
        ]
    for n in ("marauders", "stalkers"):
        L.append(
            f"- B1 fitted on marines and variants, {n}: {ev(n, 'B1 LO') / ev(n, 'B0 own'):.3f} x own (limit 1.10)"
        )
    L += ["", "## All numbers", "", "```", t.round(4).to_string(), "```", ""]
    DOC.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        with Pool(6) as pool:
            for n in pool.imap_unordered(build, list(TYPES)):
                print("built", n, flush=True)
    else:
        analyse()
