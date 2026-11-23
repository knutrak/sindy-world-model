"""Overnight: fixes S, R, T for the unit-level hybrid model (plan: docs/generalisation/20).

uv run --extra explore experiments/generalisation/overnight20.py all     every stage the rules allow
"""

import argparse
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "planning"))
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
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402
from sindy_wm.states.typed import unit_stats  # noqa: E402

ROWS = DATA_DIR / "processed" / "generalisation" / "unit_geometry"
PLAIN = DATA_DIR / "processed" / "planning" / "hybrid_models.pkl"
SEQ = DATA_DIR / "processed" / "planning" / "hybrid_models_seq.pkl"
SELECTED = DATA_DIR / "processed" / "planning" / "hybrid_selected.pkl"
RES = DATA_DIR / "processed" / "generalisation" / "overnight20.json"
DOC = PROJECT_ROOT / "docs" / "generalisation" / "21_overnight_hybrid.md"
VAL = {"marines": ("exp06_intents", 300), "mixed": ("exp08_mixed", 300)}
WD = {"marines": 7.0, "mixed": 8.0}
HZ = 32
VARIANTS = ["".join(v) for n in range(4) for v in itertools.combinations("SRT", n)]  # "", S, R, T, SR, ...
CF_NAMES = ["advance_nearest", "advance_focus", "hold_nearest", "hold_focus", "withdraw_nearest"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def doc(lines):
    DOC.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ fit S switches
def fit_seq():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    worlds = {}
    plain = pickle.load(open(PLAIN, "rb"))
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
                    prev=(a.speed_prev > 2.9).to_numpy(float),
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
                if best is None:
                    best = SwitchModel(
                        DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[sub], y[sub])
                    )
                sw[(side, typ)] = (best, ctrls)
                log(f"S switch {scen} {side} {typ}")
        w = HybridWorld(sw, plain[scen].k, WD[scen])
        worlds[scen] = w
    pickle.dump(worlds, open(SEQ, "wb"))


def world_for(scen: str, variant: str) -> HybridWorld:
    w = pickle.load(open(SEQ if "S" in variant else PLAIN, "rb"))[scen]
    w.sequence, w.steering, w.sticky = "S" in variant, "R" in variant, "T" in variant
    w.separation = "R" not in variant
    return w


# ------------------------------------------------------------------ H1 / H2
def _battle(args):
    scen, variant, exp, ep, starts = args
    w = world_for(scen, variant)
    d = RAW_DIR / exp / "val"
    snap = storage.load_snapshots(d, [ep])
    steps = np.sort(snap.step.unique())
    ctrl = (
        storage.load_controls(d).query("episode_idx == @ep").set_index("step").reindex(steps).ffill().bfill()
    )
    by = {k: g for k, g in snap.groupby("step")}
    alive = np.stack([by[k].sort_values(["team", "unit_id"]).alive.to_numpy(bool) for k in steps])
    out = []
    for k0 in starts:
        st = state_from_snapshot(by[steps[k0]])
        intents = [
            tuple(ctrl.loc[steps[min(k0 + i, len(steps) - 1)], ["movement", "target"]]) for i in range(HZ)
        ]
        ag = np.concatenate(
            [alive[k0 : k0 + HZ + 1], np.repeat(alive[-1:], max(0, k0 + HZ + 1 - len(alive)), 0)]
        )
        g, f = w.rollout(st, intents, alive_given=ag), w.rollout(st, intents)
        out.append((ep, k0, g["reach_blue"], g["reach_red"], f["hp_blue"], f["hp_red"]))
    return out


_frozen_cache = {}


def frozen_val(scen):
    if scen not in _frozen_cache:
        from reach_state import true_reach

        m = MS.frozen(scen)
        exp, nb = VAL[scen]
        st = add_kite(add_controls(pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "val.parquet")))
        st = st[st.index.get_level_values(0) < nb]
        with np.errstate(invalid="ignore", divide="ignore"):
            B = m.L.batch(st, horizon=HZ, ctrl=CTRL_K)
        fz = ReachStatePipeline.from_frozen(m, None)
        with np.errstate(all="ignore"):
            FY = fz.L.team_view(fz.rollout(B))
        _frozen_cache[scen] = (m, B, FY, fz.R_log, true_reach(st, m.L, B, HZ))
    return _frozen_cache[scen]


def h12(scen, variant, rng):
    m, B, FY, FR, TR = frozen_val(scen)
    exp, _ = VAL[scen]
    TY = B["team"]["Y"]
    jobs = {}
    for ep, k0 in zip(B["ep"], B["k0"], strict=True):
        jobs.setdefault(int(ep), []).append(int(k0))
    with Pool(8) as pool:
        res = pool.map(_battle, [(scen, variant, exp, ep, ks) for ep, ks in jobs.items()])
    hyb = {(ep, k0): v for lst in res for ep, k0, *v in lst}
    ref = unit_stats(m.L.types[0].upper())["hp"] * 10
    rows = []
    for h in (1.0, 2.0, 4.0):
        n = round(h / DT)
        idx = np.flatnonzero(~np.isnan(TY[:, n, 0]) & ~np.isnan(TR[:, n - 1, 0]))
        H = [hyb[(B["ep"][i], B["k0"][i])] for i in idx]
        hr = np.array(
            [
                abs(x[0][n - 1] - 10 * TR[i, n - 1, 0]) + abs(x[1][n - 1] - 10 * TR[i, n - 1, 1])
                for x, i in zip(H, idx, strict=True)
            ]
        )
        fr = 10 * np.abs(FR[idx, n - 1] - TR[idx, n - 1]).sum(1)
        real = TY[idx, n, 0] - TY[idx, n, 1] - (TY[idx, 0, 0] - TY[idx, 0, 1])
        hm = np.array([(x[2][n] - x[3][n] - x[2][0] + x[3][0]) / ref for x in H])
        fm = FY[idx, n, 0] - FY[idx, n, 1] - (FY[idx, 0, 0] - FY[idx, 0, 1])
        for lab, a, b in (("reach", hr, fr), ("margin", np.abs(hm - real), np.abs(fm - real))):
            mean, lo, hi = MS.boot(a - b, B["ep"][idx], rng)
            rows.append(
                {
                    "scenario": scen,
                    "variant": variant or "base",
                    "h": h,
                    "quantity": lab,
                    "hybrid": a.mean(),
                    "frozen": b.mean(),
                    "ratio": a.mean() / b.mean(),
                    "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                    "below 0": bool(hi < 0),
                }
            )
    return rows


# ------------------------------------------------------------------ H3
def _cf_job(args):
    variant, items = args
    w = world_for("marines", variant)
    out = []
    for raw_dir, ep, k, n_steps in items:
        snap = storage.load_snapshots(Path(raw_dir), [ep])
        steps = np.sort(snap.step.unique())
        ctrl = (
            storage.load_controls(Path(raw_dir)).query("episode_idx == @ep").set_index("step").reindex(steps)
        )
        ctrl = ctrl.ffill().bfill()
        st = state_from_snapshot(snap[snap.step == steps[k]])
        intents = [
            tuple(ctrl.loc[steps[min(k + i, len(steps) - 1)], ["movement", "target"]]) for i in range(n_steps)
        ]
        r = w.rollout(st, intents)
        out.append((raw_dir, ep, r["hp_blue"] - r["hp_red"]))
    return out


def h3(variant, rng):
    from model_regret import margin_change

    m = MS.frozen("marines")
    names = [n for _, n in MS.SCEN["marines"]["cf"]]
    st = MS.from_branch(MS.load("marines", "cf", branch=True), m.L, len(names))
    with np.errstate(invalid="ignore", divide="ignore"):
        B = m.L.batch(st, horizon=64, every=10**9, history=0, min_future=1, ctrl=CTRL_K)
    fz = ReachStatePipeline.from_frozen(m, None)
    with np.errstate(all="ignore"):
        FY = fz.L.team_view(fz.rollout(B))
    TY = B["team"]["Y"]
    g = st.groupby(level=0, sort=False)
    seed = g["seed"].first().loc[B["ep"]].to_numpy()
    branch = g["branch"].first().loc[B["ep"]].to_numpy()
    k_at = {ep: int(round(gg["branch_at"].iloc[0] / DT)) for ep, gg in g}
    items = []
    for ep, b in zip(B["ep"], branch, strict=True):
        exp, name = MS.SCEN["marines"]["cf"][b]
        items.append((str(RAW_DIR / exp / name), int(ep) - 100_000 * int(b), k_at[ep], 64))
    chunks = [items[i::32] for i in range(32)]
    with Pool(8) as pool:
        res = pool.map(_cf_job, [(variant, c) for c in chunks])
    pred = {(d, e): mm for lst in res for d, e, mm in lst}
    ref = unit_stats("MARINE")["hp"] * 10
    HM = np.full((len(items), 65), np.nan)
    for i, (d, e, _, _) in enumerate(items):
        x = pred[(d, e)] / ref
        HM[i, : len(x)] = x
    HY = np.stack([HM, np.zeros_like(HM), np.zeros_like(HM), np.zeros_like(HM), np.zeros_like(HM)], -1)

    def wide(v):
        return (
            pd.DataFrame({"s": seed, "b": branch, "v": v})
            .pivot(index="s", columns="b", values="v")
            .to_numpy()
        )

    rows = []
    for h in (1.0, 2.0, 4.0):
        T = wide(margin_change(TY, TY, h))
        idx = np.arange(len(T))
        pf = wide(margin_change(FY, TY, h)).argmax(1)
        ph = wide(margin_change(HY, TY, h)).argmax(1)
        rf, rh = T.max(1) - T[idx, pf], T.max(1) - T[idx, ph]
        ra = T.max(1) - T[idx, 0]
        d = rh - rf
        mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.quantile(mm, [0.025, 0.975])
        rows.append(
            {
                "h": h,
                "regret hybrid": rh.mean(),
                "regret frozen": rf.mean(),
                "always advance/nearest": ra.mean(),
                "hybrid − frozen": d.mean(),
                "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                "below 0": bool(hi < 0),
                "states": len(T),
            }
        )
    return rows


# ------------------------------------------------------------------ the chain
def all_():
    rng = np.random.default_rng(0)
    lines = [
        "# Overnight: fixes S, R, T for the unit-level hybrid model: results",
        "",
        "Plan: `20_overnight_hybrid_plan.md` (approved 23 Nov).",
        "Script: `experiments/generalisation/overnight20.py`.",
        "Tables are appended as each stage finishes.",
        "",
    ]
    doc(lines)
    t0 = time.time()
    subprocess.run([sys.executable, str(HERE / "unit_geometry.py"), "rows"], check=True)
    log("rows rebuilt with speed_prev")
    fit_seq()
    log("S switches fitted")
    rows = []
    for variant in VARIANTS:
        for scen in ("marines", "mixed"):
            rows += h12(scen, variant, rng)
            log(f"H1/H2 {scen} {variant or 'base'} done")
        json.dump({"h12": rows}, open(RES, "w"), indent=1, default=float)
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["scenario", "variant"], columns=["quantity", "h"], values="ratio")
    lines += [
        "## H1 / H2: error ratio to the frozen model (lower is better; H1 passes at ≤ 0.7 for reach)",
        "",
        "```",
        piv.round(3).to_string(),
        "```",
        "",
    ]
    mar = t[(t.scenario == "marines") & (t.quantity == "reach")].pivot_table(
        index="variant", columns="h", values="ratio"
    )
    passing = mar[(mar[2.0] <= 0.7) & (mar[4.0] <= 0.7)]
    if passing.empty:
        lines += [
            "**Selection:** no variant passes H1 in marines; the run stops here (plan `20`).",
            "",
            f"Run time: {(time.time() - t0) / 60:.0f} min.",
            "",
        ]
        doc(lines)
        return
    sel = passing[2.0].idxmin()
    sel_key = "" if sel == "base" else sel
    mm = t[(t.scenario == "marines") & (t.quantity == "margin") & (t.variant == sel)].set_index("h")
    h2_ok = bool(mm.loc[2.0, "below 0"] and mm.loc[4.0, "below 0"])
    lines += [
        f"**Selection:** variant `{sel}` (lowest marines reach ratio at 2 s among those passing H1). "
        f"H2 (margin below frozen at 2 s and 4 s): {'passes' if h2_ok else 'fails'}.",
        "",
    ]
    doc(lines)
    if not h2_ok:
        lines += [
            "The run stops after H2 (plan `20`).",
            "",
            f"Run time: {(time.time() - t0) / 60:.0f} min.",
            "",
        ]
        doc(lines)
        return
    r3 = h3(sel_key, rng)
    h3_ok = all(r["below 0"] for r in r3 if r["h"] in (2.0, 4.0))
    lines += [
        "## H3: ranking on exp06's counterfactual sets (five team-level candidates)",
        "",
        "```",
        pd.DataFrame(r3).round(4).to_string(index=False),
        "```",
        "",
        f"**H3: {'passes' if h3_ok else 'fails'}.**",
        "",
    ]
    doc(lines)
    log("H3 done")
    if not h3_ok:
        lines += [f"Run time: {(time.time() - t0) / 60:.0f} min.", ""]
        doc(lines)
        return
    pickle.dump(
        {"marines": world_for("marines", sel_key), "mixed": world_for("mixed", sel_key)}, open(SELECTED, "wb")
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "planning" / "closed34.py"),
            "run",
            "S1",
            "--n",
            "200",
            "--hybrid",
        ],
        check=True,
    )
    d = pd.read_csv(DATA_DIR / "processed" / "planning" / "closed34_S1.csv")
    d = d[d.seed < 21_000_000 + 200]
    W = d.pivot_table(index="seed", columns="policy", values="won", aggfunc="first").astype(float)
    res = []
    for ref in ("musigma5:h3", "fixed:hold/focus", "oracle5-h3", "model:h3"):
        x = (W["hybrid:h3"] - W[ref]).dropna().to_numpy()
        mm_ = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
        lo, hi = np.quantile(mm_, [0.025, 0.975])
        res.append(
            {
                "against": ref,
                "hybrid wins": W["hybrid:h3"].mean(),
                "its wins": W[ref].mean(),
                "difference": x.mean(),
                "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                "read-out": "beats" if lo > 0 else ("loses" if hi < 0 else "ties"),
            }
        )
    share = (W["hybrid:h3"].mean() - W["fixed:hold/focus"].mean()) / (
        W["oracle5-h3"].mean() - W["fixed:hold/focus"].mean()
    )
    helps = res[0]["read-out"] == "beats"
    lines += [
        "## H4: closed loop, S1, `34`'s first 200 seeds",
        "",
        "```",
        pd.DataFrame(res).round(3).to_string(index=False),
        "```",
        "",
        f"Share of the matched ceiling captured: {share:.2f}.",
        "",
        f'**"Helps the planner" (beats the μ, σ planner): {"yes" if helps else "no"}.**',
        "",
        f"Run time: {(time.time() - t0) / 60:.0f} min.",
        "",
    ]
    doc(lines)
    log("H4 done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["all"])
    p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    all_()


if __name__ == "__main__":
    main()
