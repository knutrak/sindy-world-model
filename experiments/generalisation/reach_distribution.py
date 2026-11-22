"""Feasibility check: do the mean and spread of each unit's nearest-enemy reach margin carry reach?
(docs/generalisation/09_reach_distribution_check.md; exploratory, criterion fixed before the run.)

    uv run --extra explore experiments/generalisation/reach_distribution.py
"""

import os
import pickle
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.models.intent_model import SHAPE  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402
from sindy_wm.states.margins import margins  # noqa: E402

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
DOC = PROJECT_ROOT / "docs" / "generalisation" / "09_reach_distribution_check.md"
SCEN = {
    "marines": ("exp06_intents", ("MARINE",)),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER")),
}
SIDES = ("blue", "red")
MAX_REACH = 6.0
N_TRAIN = 1000


def build_one(job) -> pd.DataFrame:
    d, ids, types, offset = job
    snaps = storage.load_snapshots(d, ids)
    parts = []
    for ep, g in snaps.groupby("episode_idx"):
        t = margins(g, types)
        t["episode_idx"] = ep + offset
        parts.append(t.reset_index().set_index(["episode_idx", "step"]))
    return pd.concat(parts)


def build(exp: str, name: str, types, n: int | None = None) -> pd.DataFrame:
    d = RAW_DIR / exp / name
    ids = storage.load_episode_index(d)["episode_idx"].tolist()[:n]
    jobs = [(d, c.tolist(), types, 0) for c in np.array_split(np.asarray(ids), 16) if len(c)]
    with Pool(8) as pool:
        return pd.concat(pool.map(build_one, jobs))


def predict(D: pd.DataFrame, side, t, c=0.0, k=1.0) -> np.ndarray:
    mu, sd, n = D[f"mu_{side}_{t}"].to_numpy(), D[f"sd_{side}_{t}"].to_numpy(), D[f"n_{side}_{t}"].to_numpy()
    with np.errstate(all="ignore"):
        s = ndtr((c - mu) / np.maximum(k * sd, 1e-3))
    return np.where(n > 0, np.nan_to_num(s), 0.0) * n / 10  # units in reach / 10


def main() -> None:
    models = pickle.load(open(FROZEN, "rb"))
    lines, rows = [], []
    for scen, (exp, types) in SCEN.items():
        m = models[scen]
        L = m.L
        tt = [t.lower() for t in types]
        tr = build(exp, "train_0", types, N_TRAIN)
        va = build(exp, "val", types)
        typed = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet")
        va = va.join(typed, how="inner")
        both = sum(va[f"alive_blue_{t}"] for t in tt).gt(0) & sum(va[f"alive_red_{t}"] for t in tt).gt(0)
        va = va[both]
        # consistency: counted units with margin <= 0 equal the typed reach
        check = max(
            float(np.abs(va[f"in_{s}_{t}"] / 10 - va[f"reach_{s}_{t}"]).max()) for s in SIDES for t in tt
        )
        # fit c, k on train (all sides and types pooled), on the measured share
        trn = tr[(sum(tr[f"n_blue_{t}"] for t in tt) > 0) & (sum(tr[f"n_red_{t}"] for t in tt) > 0)]

        def resid(p, D=trn, tt=tt):
            return np.concatenate(
                [predict(D, s, t, *p) - D[f"in_{s}_{t}"].to_numpy() / 10 for s in SIDES for t in tt]
            )

        fit = least_squares(resid, x0=(0.0, 1.0)).x
        X = va[L.x_cols].to_numpy(float)
        S = va[SHAPE].fillna(0).to_numpy(float)
        F = va[L.front_cols].fillna(0).to_numpy(float)
        with np.errstate(all="ignore"):
            sh = m._shares(X, S, F)
        err = {"closure": 0.0, "mu-sigma raw": 0.0, "mu-sigma fitted": 0.0}
        for s in SIDES:
            meas = sum(va[f"reach_{s}_{t}"].to_numpy() for t in tt)
            clo = sum(va[f"alive_{s}_{t}"].to_numpy() * np.nan_to_num(sh[(s, t)]) for t in tt)
            raw = sum(predict(va, s, t) for t in tt)
            fitd = sum(predict(va, s, t, *fit) for t in tt)
            for name, p in (("closure", clo), ("mu-sigma raw", raw), ("mu-sigma fitted", fitd)):
                err[name] += 10 * np.abs(p - meas).mean() / 2
        same = {
            k: float(
                np.mean(
                    [
                        (va[f"{k}_{s}_{t}"].groupby(level=0).diff().abs() < 1e-9).mean()
                        for s in SIDES
                        for t in tt
                    ]
                )
            )
            for k in ("mu", "sd")
        }
        rows.append(
            {
                "scenario": scen,
                "steps": len(va),
                "closure": err["closure"],
                "mu-sigma raw": err["mu-sigma raw"],
                "mu-sigma fitted": err["mu-sigma fitted"],
                "fitted / closure": err["mu-sigma fitted"] / err["closure"],
                "c": fit[0],
                "k": fit[1],
                "mu unchanged": same["mu"],
                "sd unchanged": same["sd"],
                "count check (max abs)": check,
            }
        )
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    passed = bool((t["fitted / closure"] <= 0.75).all())
    lines = [
        "",
        "**Run 22 Nov.** Reach error in units per side and step (val, measured states):",
        "",
        "```",
        t.round(4).to_string(index=False),
        "```",
        "",
        f"**Criterion (fitted ≤ 0.75 × closure in both scenarios): {'passes' if passed else 'fails'}.**",
        "'count check' = the largest difference between units counted with margin ≤ 0 and the typed reach "
        "column (should be 0).",
        "",
    ]
    text = DOC.read_text().replace("(filled in after the run)\n", "\n".join(lines))
    DOC.write_text(text)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
