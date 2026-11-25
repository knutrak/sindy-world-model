# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 0 data: prepared rows and scoring batches from TRAIN battles only, cached per army (docs/overnight/00_plan.md)."""

import os
import pickle
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sindy_wm.io import storage
from sindy_wm.models.intent_model import SHAPE, add_controls, add_rates
from sindy_wm.models.pipeline import SIDES, Layout, team_rows
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.margins import margins

PROC = DATA_DIR / "processed"
OUT = PROC / "overnight"
ARMIES = {
    "marines": ("exp06_intents", ("MARINE",), 7.0, 1500, 400),
    "marauders": ("exp09_marauders", ("MARAUDER",), 7 * 7.125 / 5.75, 1200, 300),
    "stalkers": ("exp07_stalkers", ("STALKER",), 7 * 7.25 / 5.75, 1200, 300),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER"), 8.0, 1500, 400),
    "triple": ("exp12_triple", ("MARINE", "MARAUDER", "STALKER"), 8.0, 1200, 300),
}
VARIANTS = ("DM", "DP", "CM", "CP", "HM", "A1", "SM", "SP", "R4", "R3")


def spec(name):
    if name in ARMIES:
        return ARMIES[name]
    return ("exp13_properties", (f"MARINE_{name}",), 7.0, 350, 150)


def raw_parts(exp, name, ids):
    if exp == "exp13_properties":
        return [(RAW_DIR / exp / name.lower(), list(ids), 0)]
    base = RAW_DIR / exp
    if (base / "train").exists():
        return [(base / "train", list(ids), 0)]
    parts = {}
    for i in ids:
        parts.setdefault(i // 1000, []).append(i % 1000)
    return [(base / f"train_{k}", v, 1000 * k) for k, v in parts.items()]


def extend(st, types, exp, name, marg=True):
    """Adds margins (from snapshots) and the backward difference of the distance."""
    if marg:
        parts = []
        for d, ids, off in raw_parts(exp, name, st.index.get_level_values(0).unique()):
            for c in np.array_split(np.asarray(ids), max(1, len(ids) // 100)):
                for ep, g in storage.load_snapshots(d, c.tolist()).groupby("episode_idx"):
                    m = margins(g, types)
                    m["episode_idx"] = ep + off
                    parts.append(m.reset_index().set_index(["episode_idx", "step"]))
        st = st.join(pd.concat(parts)[[c for c in parts[0] if c.startswith(("mu_", "sd_"))]])
    st["vd"] = st.groupby(level=0)["distance"].diff().fillna(0.0) / 0.125
    return st


def batch(L, st):
    """Layout.batch plus mu0, sd0, vd0 at the same start points."""
    B = L.batch(st)
    idx = pd.MultiIndex.from_arrays([B["ep"], B["k0"]])
    pos = st.groupby(level=0).cumcount().to_numpy()
    key = pd.MultiIndex.from_arrays([st.index.get_level_values(0), pos])
    for k in ("mu", "sd"):
        cols = [f"{k}_{s}_{t}" for s in SIDES for t in L.types]
        if cols[0] in st:
            B[f"{k}0"] = pd.DataFrame(st[cols].to_numpy(), index=key).loc[idx].to_numpy()
    B["vd0"] = pd.Series(st["vd"].to_numpy(), index=key).loc[idx].to_numpy()
    return B


def one(name):
    path = OUT / f"{name}.pkl"
    if path.exists():
        return name
    exp, types, wr, nf, ns = spec(name)
    if name in ARMIES:
        t = pd.read_parquet(PROC / f"{exp}_typed" / "train.parquet")
    else:
        t = pd.read_parquet(PROC / "exp13_properties_typed" / f"{name.lower()}.parquet")
    ids = t.index.get_level_values(0).unique()[: nf + ns]
    st = add_controls(extend(t.loc[ids].copy(), types, exp, name))
    L = Layout(tuple(x.lower() for x in types))
    mcols = [c for c in st if c.startswith(("mu_", "sd_"))]
    for c in mcols:  # margins of a wiped-out type: hold the last value (rows with the type alive are used)
        st[c] = st.groupby(level=0)[c].ffill().fillna(0.0)
    fit_ids, score_ids = ids[:nf], ids[nf:]
    r = add_rates(st.loc[fit_ids], cols=(*L.x_cols, *SHAPE, *mcols), window=9)
    al = {s: sum(r[f"alive_{s}_{a}"] for a in L.types) for s in SIDES}
    r = team_rows(r[(al["blue"] > 0) & (al["red"] > 0)], L)
    r = r.sample(min(60_000, len(r)), random_state=0)
    pickle.dump({"rows": r, "B": batch(L, st.loc[score_ids]), "types": types, "wr": wr}, open(path, "wb"))
    return name


if __name__ == "__main__":
    names = sys.argv[1:] or [*ARMIES, *VARIANTS]
    with Pool(6) as pool:
        for n in pool.imap_unordered(one, names):
            print("prepared", n, flush=True)
