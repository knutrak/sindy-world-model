"""Unit-level geometry with team-level damage (approved plan: docs/generalisation/13_unit_geometry_plan.md).

uv run --extra explore experiments/generalisation/unit_geometry.py rows    per-unit rows (train subsets, val)
uv run --extra explore experiments/generalisation/unit_geometry.py u1      fit movement laws, U1 criteria
"""

import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from sindy_wm.io import storage  # noqa: E402
from sindy_wm.models.reach_state import gram, two_level_gram  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402
from sindy_wm.states.typed import pair_reach  # noqa: E402

OUT = DATA_DIR / "processed" / "generalisation" / "unit_geometry"
DOC = PROJECT_ROOT / "docs" / "generalisation" / "14_unit_geometry.md"
SETS = {
    "marines": {
        "train": [("exp06_intents", "train_0", 1000)],
        "val": [("exp06_intents", "val", 300)],
        "wd": 7.0,
    },
    "mixed": {
        "train": [("exp08_mixed", "train_0", 1000), ("exp11_kite", "train_0", 500)],
        "val": [("exp08_mixed", "val", 300), ("exp11_kite", "val", 300)],
        "wd": 8.0,
    },
}
MAX_CD = {"MARINE": 0.61, "MARAUDER": 1.07}
SPEED = 3.15
GATES = (-1.0, -0.5, 0.0, 0.5, 1.0)
GROUPS = ["1", "m", "m2", *[f"g{c:g}" for c in GATES], "ready", "ready*g0", "danger", "tir"]
CTRL_BLUE = ["1", "adv", "hold", "hold*contact", "wd", "focus", "kite"]
THR = 0.05


def episode_rows(snap: pd.DataFrame, ctrl: pd.DataFrame, wd: float) -> pd.DataFrame:
    steps = np.sort(snap.step.unique())
    units = snap.drop_duplicates("unit_key")[["unit_key", "team", "unit_type"]].set_index("unit_key")
    keys = units.index.to_numpy()
    piv = {
        c: snap.pivot(index="step", columns="unit_key", values=c).reindex(index=steps, columns=keys)
        for c in ("x", "y", "vx", "vy", "cooldown", "alive", "target_in_range", "tkey")
    }
    pos = np.stack([piv["x"].to_numpy(float), piv["y"].to_numpy(float)], -1)
    vel = np.stack([piv["vx"].to_numpy(float), piv["vy"].to_numpy(float)], -1)
    alive = piv["alive"].fillna(False).to_numpy(bool)
    cd = piv["cooldown"].to_numpy(float)
    tir = piv["target_in_range"].fillna(False).to_numpy(bool)
    tkey = piv["tkey"].fillna("").to_numpy(object)
    col = {k: n for n, k in enumerate(keys)}
    team = units.team.to_numpy()
    utype = units.unit_type.to_numpy()
    blue, red = team == "blue", team == "red"
    R = np.array([[min(pair_reach(a, b), 6.0) for b in utype] for a in utype])
    c = ctrl.reindex(steps)
    out = []
    for t in range(len(steps) - 1):
        al = alive[t]
        if not (al & blue).any() or not (al & red).any():
            continue
        d = np.linalg.norm(pos[t][:, None, :] - pos[t][None, :, :], axis=-1)
        enemy = (team[:, None] != team[None, :]) & al[None, :]
        marg = np.where(enemy, d - R, np.inf)
        j = marg.argmin(1)
        m = marg[np.arange(len(keys)), j]
        dist_near = np.where(enemy, d, np.inf).min(1)
        contact = bool((m[blue & al] <= 0).any())
        mv, tg = c.movement.iloc[t], c.target.iloc[t]
        for i in np.flatnonzero(al & alive[t + 1]):
            e = pos[t, j[i]] - pos[t, i]
            n = np.linalg.norm(e)
            e = e / n if n > 0 else np.array([1.0, 0.0])
            v = vel[t + 1, i]
            vpar = float(v @ e)
            sp = float(np.linalg.norm(v))
            k = col.get(
                tkey[t, i], -1
            )  # own target if alive, else the nearest enemy (docs/generalisation/15)
            h = pos[t, k] - pos[t, i] if k >= 0 and al[k] else pos[t, j[i]] - pos[t, i]
            hn = np.linalg.norm(h)
            cos_head = float(v @ h / (sp * hn)) if sp > 1e-6 and hn > 1e-6 else np.nan
            out.append(
                (
                    team[i],
                    utype[i],
                    m[i],
                    cd[t, i] / MAX_CD[utype[i]],
                    float(dist_near[i] < wd),
                    float(tir[t, i]),
                    contact,
                    mv,
                    tg,
                    vpar,
                    float(np.linalg.norm(v - vpar * e)),
                    sp,
                    cos_head,
                )
            )
    return pd.DataFrame(
        out,
        columns=[
            "team",
            "type",
            "m",
            "ready",
            "danger",
            "tir",
            "contact",
            "movement",
            "target",
            "vpar",
            "vperp",
            "speed",
            "cos_head",
        ],
    )


def job(args) -> pd.DataFrame:
    exp, name, ids, wd = args
    d = RAW_DIR / exp / name
    snaps = storage.load_snapshots(d, ids)
    ctrl = storage.load_controls(d)
    parts = []
    for ep, g in snaps.groupby("episode_idx"):
        g = g.assign(unit_key=g.team + "_" + g.unit_id.astype(str))
        g = g.assign(
            tkey=np.where(g.target_id >= 0, g.target_team.astype(str) + "_" + g.target_id.astype(str), "")
        )
        cc = ctrl[ctrl.episode_idx == ep].set_index("step")[["movement", "target"]]
        r = episode_rows(g, cc, wd)
        r["battle"] = f"{exp}/{name}/{ep}"
        parts.append(r)
    return pd.concat(parts)


def rows():
    OUT.mkdir(parents=True, exist_ok=True)
    for scen, spec in SETS.items():
        for split in ("train", "val"):
            jobs = []
            for exp, name, n in spec[split]:
                ids = storage.load_episode_index(RAW_DIR / exp / name)["episode_idx"].tolist()[:n]
                jobs += [
                    (exp, name, c.tolist(), spec["wd"]) for c in np.array_split(np.asarray(ids), 40) if len(c)
                ]
            with Pool(8) as pool:
                t = pd.concat(pool.map(job, jobs))
            t.to_parquet(OUT / f"{scen}_{split}.parquet")
            print(scen, split, len(t), "rows", flush=True)


def design(df: pd.DataFrame, side: str, ctrls) -> np.ndarray:
    m = df.m.to_numpy(float)
    G = {"1": np.ones(len(m)), "m": np.clip(m, -3, 10), "m2": np.clip(m, -3, 10) ** 2}
    for c in GATES:
        G[f"g{c:g}"] = ndtr((m - c) / 0.3)  # 1 when out of range by more than c
    G["ready"] = df.ready.to_numpy(float)
    G["ready*g0"] = G["ready"] * G["g0"]
    G["danger"], G["tir"] = df.danger.to_numpy(float), df.tir.to_numpy(float)
    C = {"1": np.ones(len(m))}
    if side == "blue":
        mv = df.movement.to_numpy()
        C |= {
            "adv": (mv == "advance") * 1.0,
            "hold": (mv == "hold") * 1.0,
            "hold*contact": (mv == "hold") * df.contact.to_numpy(float),
            "wd": (mv == "withdraw") * 1.0,
            "focus": (df.target.to_numpy() == "focus") * 1.0,
            "kite": (mv == "kite") * 1.0,
        }
    Gm = np.column_stack([G[g] for g in GROUPS])
    Cm = np.column_stack([C[c] for c in ctrls])
    return (Gm[:, :, None] * Cm[:, None, :]).reshape(len(m), -1)


def u1():
    res, rows_out = {}, []
    for scen in SETS:
        tr = pd.read_parquet(OUT / f"{scen}_train.parquet")
        va = pd.read_parquet(OUT / f"{scen}_val.parquet")
        for side in ("blue", "red"):
            ctrls = [c for c in CTRL_BLUE if scen == "mixed" or c != "kite"] if side == "blue" else ["1"]
            for typ in sorted(tr.type.unique()):
                a, b = tr[(tr.team == side) & (tr.type == typ)], va[(va.team == side) & (va.type == typ)]
                Th, Tv = design(a, side, ctrls), design(b, side, ctrls)
                groups = np.repeat(np.arange(len(GROUPS)), len(ctrls))
                base = np.tile([c == "1" for c in ctrls], len(GROUPS))
                coef = two_level_gram(*gram(Th, a.vpar.to_numpy()), groups, base, THR)
                yv = b.vpar.to_numpy()
                r2 = 1 - ((yv - Tv @ coef) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
                probe = pd.DataFrame(
                    {
                        "m": [3.0],
                        "ready": [0.0],
                        "danger": [0.0],
                        "tir": [0.0],
                        "contact": [False],
                        "movement": ["advance"],
                        "target": ["nearest"],
                    }
                )
                walk = float((design(probe, side, ctrls) @ coef)[0])
                terms = {
                    f"{GROUPS[k // len(ctrls)]}·{ctrls[k % len(ctrls)]}": float(v)
                    for k, v in enumerate(coef)
                    if v
                }
                ok = len(terms) <= 6 and r2 >= 0.8 and abs(walk - SPEED) <= 0.15 * SPEED
                res[(scen, side, typ)] = {"coef": coef, "ctrls": ctrls}
                rows_out.append(
                    {
                        "scenario": scen,
                        "side": side,
                        "type": typ,
                        "rows (train)": len(a),
                        "terms": len(terms),
                        "val R²": r2,
                        "walking v∥ (m=3, advance)": walk,
                        "U1": bool(ok),
                        "equation": " ".join(f"{v:+.3g}·{k}" for k, v in terms.items()),
                    }
                )
                print(rows_out[-1], flush=True)
    pickle.dump(res, open(OUT / "u1_laws.pkl", "wb"))
    json.dump(rows_out, open(OUT / "u1.json", "w"), indent=1, default=float)


def hybrid():
    """docs/generalisation/15: walk/stop switch classifier (L1 logistic, <= 8 terms; depth-3 tree alongside),
    walk speed and heading, per side and type."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    rows_out = []
    for scen in SETS:
        tr = pd.read_parquet(OUT / f"{scen}_train.parquet")
        va = pd.read_parquet(OUT / f"{scen}_val.parquet")
        for side in ("blue", "red"):
            ctrls = [c for c in CTRL_BLUE if scen == "mixed" or c != "kite"] if side == "blue" else ["1"]
            names = [f"{g}·{c}" for g in GROUPS for c in ctrls]
            for typ in sorted(tr.type.unique()):
                a, b = tr[(tr.team == side) & (tr.type == typ)], va[(va.team == side) & (va.type == typ)]
                share = {"walk": (b.speed > 2.9).mean(), "stop": (b.speed < 0.3).mean()}
                a2, b2 = a[(a.speed > 2.9) | (a.speed < 0.3)], b[(b.speed > 2.9) | (b.speed < 0.3)]
                Xa, Xb = design(a2, side, ctrls), design(b2, side, ctrls)
                ya, yb = (a2.speed > 2.9).to_numpy(int), (b2.speed > 2.9).to_numpy(int)
                sub = np.random.default_rng(0).choice(len(ya), min(200_000, len(ya)), replace=False)
                mu, sd = Xa[sub].mean(0), Xa[sub].std(0) + 1e-9
                best = None
                for C in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
                    clf = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=300, tol=1e-3)
                    clf.fit((Xa[sub] - mu) / sd, ya[sub])
                    nz = int((clf.coef_ != 0).sum())
                    if nz <= 8:
                        acc = float((clf.predict((Xb - mu) / sd) == yb).mean())
                        best = (C, nz, acc, clf)
                tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xa[sub], ya[sub])
                tree_acc = float((tree.predict(Xb) == yb).mean())
                walk = b[b.speed > 2.9]
                med_speed, med_cos = float(walk.speed.median()), float(walk.cos_head.median())
                C, nz, acc, clf = best if best else (None, 0, float("nan"), None)
                terms = (
                    {names[k]: float(v) for k, v in enumerate(clf.coef_[0]) if v} if clf is not None else {}
                )
                ok = acc >= 0.9 and abs(med_speed - SPEED) <= 0.05 * SPEED and med_cos >= 0.95
                rows_out.append(
                    {
                        "scenario": scen,
                        "side": side,
                        "type": typ,
                        "walk share": share["walk"],
                        "stop share": share["stop"],
                        "in between": 1 - share["walk"] - share["stop"],
                        "switch accuracy (L1, <= 8 terms)": acc,
                        "terms": nz,
                        "C": C,
                        "tree depth 3 accuracy": tree_acc,
                        "majority class": max(yb.mean(), 1 - yb.mean()),
                        "walk speed (median)": med_speed,
                        "walk heading cos (median)": med_cos,
                        "passes": bool(ok),
                        "switch terms (standardised)": " ".join(f"{v:+.2f}·{k}" for k, v in terms.items()),
                    }
                )
                print(rows_out[-1], flush=True)
    json.dump(rows_out, open(OUT / "hybrid.json", "w"), indent=1, default=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["rows", "u1", "hybrid"])
    {"rows": rows, "u1": u1, "hybrid": hybrid}[p.parse_args().what]()


if __name__ == "__main__":
    main()
