"""A unit-level hybrid world model in the planner (plan: docs/generalisation/16_hybrid_model_plan.md).

uv run --extra explore experiments/generalisation/hybrid_model.py fit    switch classifiers
uv run --extra explore experiments/generalisation/hybrid_model.py h12    H1 geometry, H2 full model (val)
uv run --extra explore experiments/generalisation/hybrid_model.py h3     ranking on the counterfactual sets
"""

import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mu_sigma as MS  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.models.hybrid import (  # noqa: E402
    CTRL_BLUE,
    HybridWorld,
    SwitchModel,
    features,
    state_from_snapshot,
)
from sindy_wm.models.intent_model import DT, add_controls  # noqa: E402
from sindy_wm.models.kite import CTRL_K, add_kite  # noqa: E402
from sindy_wm.models.reach_state import ReachStatePipeline  # noqa: E402
from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.typed import unit_stats  # noqa: E402

ROWS = DATA_DIR / "processed" / "generalisation" / "unit_geometry"
OUT = DATA_DIR / "processed" / "planning" / "hybrid_models.pkl"
RES = DATA_DIR / "processed" / "generalisation" / "hybrid_model.json"
WD = {"marines": 7.0, "mixed": 8.0}
VAL = {"marines": ("exp06_intents", 300), "mixed": ("exp08_mixed", 300)}
HZ = 32
SEPARATION = os.environ.get("HYBRID_SEPARATION") == "1"  # docs/generalisation/18
N_BOOT = 2000


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    worlds = {}
    for scen in ("marines", "mixed"):
        tr = pd.read_parquet(ROWS / f"{scen}_train.parquet")
        sw = {}
        for side in ("blue", "red"):
            ctrls = [c for c in CTRL_BLUE if scen == "mixed" or c != "kite"] if side == "blue" else ["1"]
            for typ in sorted(tr.type.unique()):
                a = tr[(tr.team == side) & (tr.type == typ) & ((tr.speed > 2.9) | (tr.speed < 0.3))]
                X = features(
                    a.m.to_numpy(float),
                    a.ready.to_numpy(float),
                    a.danger.to_numpy(float),
                    a.tir.to_numpy(float),
                    a.contact.to_numpy(bool),
                    a.movement.to_numpy(object),
                    a.target.to_numpy(object),
                    side,
                    ctrls,
                )
                y = (a.speed > 2.9).to_numpy(int)
                sub = np.random.default_rng(0).choice(len(y), min(200_000, len(y)), replace=False)
                mu, sd = X[sub].mean(0), X[sub].std(0) + 1e-9
                best = None
                for C in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
                    clf = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=300, tol=1e-3)
                    clf.fit((X[sub] - mu) / sd, y[sub])
                    if int((clf.coef_ != 0).sum()) <= 8:
                        best = SwitchModel(clf, mu, sd)
                if best is None:  # docs/generalisation/16: the depth-3 tree where no <= 8-term rule exists
                    best = SwitchModel(
                        DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[sub], y[sub])
                    )
                sw[(side, typ)] = (best, ctrls)
                print(scen, side, typ, "fitted", flush=True)
        worlds[scen] = HybridWorld(sw, HybridWorld.k_from_pipeline(MS.frozen(scen)), WD[scen])
    pickle.dump(worlds, open(OUT, "wb"))


def _battle_job(args):
    scen, exp, ep, starts = args
    world = pickle.load(open(OUT, "rb"))[scen]
    world.separation = SEPARATION
    d = RAW_DIR / exp / "val"
    snap = storage.load_snapshots(d, [ep])
    steps = np.sort(snap.step.unique())
    ctrl = (
        storage.load_controls(d).query("episode_idx == @ep").set_index("step").reindex(steps).ffill().bfill()
    )
    by_step = {k: g for k, g in snap.groupby("step")}
    alive = np.stack([by_step[k].sort_values(["team", "unit_id"]).alive.to_numpy(bool) for k in steps])
    out = []
    for k0 in starts:
        st = state_from_snapshot(by_step[steps[k0]])
        intents = [
            tuple(ctrl.loc[steps[min(k0 + i, len(steps) - 1)], ["movement", "target"]]) for i in range(HZ)
        ]
        ag = np.concatenate(
            [alive[k0 : k0 + HZ + 1], np.repeat(alive[-1:], max(0, k0 + HZ + 1 - len(alive)), 0)]
        )
        g = world.rollout(st, intents, alive_given=ag)
        f = world.rollout(st, intents)
        out.append((ep, k0, g["reach_blue"], g["reach_red"], f["hp_blue"], f["hp_red"]))
    return out


def h12():
    rng = np.random.default_rng(0)
    rows = []
    for scen in ("marines", "mixed"):
        m = MS.frozen(scen)
        exp, nb = VAL[scen]
        st = add_kite(add_controls(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet")))
        st = st[st.index.get_level_values(0) < nb]
        with np.errstate(invalid="ignore", divide="ignore"):
            B = m.L.batch(st, horizon=HZ, ctrl=CTRL_K)
        fz = ReachStatePipeline.from_frozen(m, None)
        with np.errstate(all="ignore"):
            FY = fz.L.team_view(fz.rollout(B))
        FR = fz.R_log
        TY = B["team"]["Y"]
        from reach_state import true_reach

        TR = true_reach(st, m.L, B, HZ)
        jobs = {}
        for i, (ep, k0) in enumerate(zip(B["ep"], B["k0"], strict=True)):
            jobs.setdefault(int(ep), []).append((i, int(k0)))
        with Pool(8) as pool:
            res = pool.map(_battle_job, [(scen, exp, ep, [k for _, k in v]) for ep, v in jobs.items()])
        hyb = {}
        for lst in res:
            for ep, k0, rb, rr, hb, hr in lst:
                hyb[(ep, k0)] = (rb, rr, hb, hr)
        ref = unit_stats(m.L.types[0].upper())["hp"] * 10
        ep_all = B["ep"]
        for h in (1.0, 2.0, 4.0):
            n = round(h / DT)
            ok = ~np.isnan(TY[:, n, 0]) & ~np.isnan(TR[:, n - 1, 0])
            idx = np.flatnonzero(ok)
            hr_err = np.array(
                [
                    abs(hyb[(ep_all[i], B["k0"][i])][0][n - 1] - 10 * TR[i, n - 1, 0])
                    + abs(hyb[(ep_all[i], B["k0"][i])][1][n - 1] - 10 * TR[i, n - 1, 1])
                    for i in idx
                ]
            )
            fr_err = 10 * np.abs(FR[idx, n - 1] - TR[idx, n - 1]).sum(1)
            real = TY[idx, n, 0] - TY[idx, n, 1] - (TY[idx, 0, 0] - TY[idx, 0, 1])
            hm = np.array(
                [
                    (
                        hyb[(ep_all[i], B["k0"][i])][2][n]
                        - hyb[(ep_all[i], B["k0"][i])][3][n]
                        - hyb[(ep_all[i], B["k0"][i])][2][0]
                        + hyb[(ep_all[i], B["k0"][i])][3][0]
                    )
                    / ref
                    for i in idx
                ]
            )
            fm = FY[idx, n, 0] - FY[idx, n, 1] - (FY[idx, 0, 0] - FY[idx, 0, 1])
            for lab, a, b in (
                ("H1 reach error (units)", hr_err, fr_err),
                ("H2 margin error", np.abs(hm - real), np.abs(fm - real)),
            ):
                mean, lo, hi = MS.boot(a - b, ep_all[idx], rng)
                rows.append(
                    {
                        "scenario": scen,
                        "h": h,
                        "quantity": lab,
                        "hybrid": a.mean(),
                        "frozen": b.mean(),
                        "ratio": a.mean() / b.mean(),
                        "difference": mean,
                        "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                        "below 0": bool(hi < 0),
                    }
                )
        print(scen, "done", flush=True)
    t = pd.DataFrame(rows)
    res = RES.with_name("hybrid_model_sep.json") if SEPARATION else RES
    json.dump({"h12": rows}, open(res, "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["fit", "h12"])
    {"fit": fit, "h12": h12}[p.parse_args().what]()


if __name__ == "__main__":
    main()
