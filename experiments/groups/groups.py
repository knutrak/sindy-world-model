"""Front / rear group states (docs/groups/00_plan.md). Exploratory: train and val only.

uv run --extra explore python experiments/groups/groups.py marines 1500
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sindy_wm.models.pipeline as PL  # noqa: E402
import sindy_wm.states.typed as TY  # noqa: E402
from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402

warnings.filterwarnings("ignore")
_stats = TY.unit_stats


def stats(name: str) -> dict:  # "MARINE_F" -> marine
    return _stats(name.split("_")[0])


TY.unit_stats = PL.unit_stats = stats
DT, WIN, HORIZON = 0.125, 32, 32
EXP = {"marines": ("exp06_intents", "MARINE"), "marauders": ("exp09_marauders", "MARAUDER")}
RADIUS = {"marines": 7.0, "marauders": 8.674}
RATE = {"marines": 0.213, "marauders": 0.135}
OUT = DATA_DIR / "processed" / "groups"
DOC = PROJECT_ROOT / "docs" / "groups"
LAY = {"T": ("{u}",), "G": ("{u}_f", "{u}_r")}


def relabel(w: pd.DataFrame, how: str, base: str, seed: int) -> pd.DataFrame:
    """unit_type by group, from the window's first step. Units dead at the start are labelled front."""
    if how == "team":
        return w
    s0 = w[w["step"] == w["step"].min()]
    al = s0[s0["alive"]]
    c = al.groupby("team")[["x", "y"]].mean()
    if {"blue", "red"} - set(c.index):
        return None
    axis = (c.loc["red"] - c.loc["blue"]).to_numpy()
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    rng = np.random.default_rng(seed)
    rear = {}
    for side, sign in (("blue", 1.0), ("red", -1.0)):
        a = al[al["team"] == side]
        key = sign * (a[["x", "y"]].to_numpy() @ axis) if how == "fr" else rng.random(len(a))
        order = np.argsort(-key, kind="stable")  # first = nearest the enemy
        rear[side] = set(a["unit_id"].to_numpy()[order[(len(a) + 1) // 2 :]])
    out = w.copy()
    lab = [f"{base}_{'R' if u in rear[t] else 'F'}" for u, t in zip(out["unit_id"], out["team"], strict=True)]
    out["unit_type"] = lab
    return out


def windows(job):
    """One shard: {variant: list of (key, states)} for each window, plus the controls."""
    exp, name, eps, base, mode = job
    d = RAW_DIR / exp / name
    ctrl = storage.load_controls(d).set_index(["episode_idx", "step"])
    cols = [c for c in ("movement", "target") if c in ctrl]
    ctrl = ctrl[cols].groupby(level=0).ffill()
    snaps = storage.load_snapshots(d, eps)
    out = {"team": [], "fr": [], "rand": []}
    for ep, g in snaps.groupby("episode_idx"):
        g = g.drop(columns="episode_idx")
        steps = np.sort(g["step"].unique())
        alive = g[g["alive"]].groupby(["step", "team"]).size().unstack().reindex(steps).fillna(0)
        both = (alive["blue"] > 0) & (alive["red"] > 0)
        last = int(np.nonzero(both.to_numpy())[0].max()) if both.any() else 0
        starts = range(0, last, WIN) if mode == "train" else range(8, len(steps) - 8, 8)
        for k in starts:
            if k >= last or not both.iloc[k]:
                continue
            idx = steps[k : k + WIN + 1]
            if len(idx) < 10:
                continue
            w = g[g["step"].isin(idx)]
            for how in out:
                lw = relabel(w, how, base, 1_000_003 * ep + int(k))
                if lw is None:
                    continue
                tt = (base,) if how == "team" else (f"{base}_F", f"{base}_R")
                st = TY.typed_states(lw, tt, max_reach=6.0)
                st["t"] = st.index.to_numpy() * DT
                st = st.join(ctrl.loc[ep].reindex(st.index).ffill())
                st["seed"] = 0
                out[how].append((ep * 1000 + k, st))
    return {h: pd.concat({k: s for k, s in v}, names=["episode_idx", "step"]) for h, v in out.items() if v}


def build(scen: str, n_train: int, n_val: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    exp, base = EXP[scen]
    raw = RAW_DIR / exp
    jobs, offs = {"train": [], "val": []}, {"train": [], "val": []}
    shards = sorted(p.name for p in raw.glob("train_*") if p.name.split("_")[-1].isdigit()) or ["train"]
    left = n_train
    for s in shards:
        ids = storage.load_episode_index(raw / s)["episode_idx"].tolist()[:left]
        left -= len(ids)
        for c in np.array_split(ids, max(1, len(ids) // 125)):
            if len(c):
                jobs["train"].append((exp, s, c.tolist(), base, "train"))
                offs["train"].append((int(s.split("_")[-1]) if "_" in s else 0) * 1_000_000)
        if left <= 0:
            break
    ids = storage.load_episode_index(raw / "val")["episode_idx"].tolist()[:n_val]
    for c in np.array_split(ids, max(1, len(ids) // 60)):
        jobs["val"].append((exp, "val", c.tolist(), base, "val"))
        offs["val"].append(0)
    res = {}
    for part in ("train", "val"):
        with mp.get_context("fork").Pool(8) as pool:
            got = pool.map(windows, jobs[part], chunksize=1)
        for h in ("team", "fr", "rand"):
            parts = []
            for o, g in zip(offs[part], got, strict=True):
                t = g[h]
                t.index = t.index.set_levels(t.index.levels[0] + o, level=0)
                parts.append(t)
            res[(part, h)] = pd.concat(parts)
        print(part, "windows", res[(part, "team")].index.get_level_values(0).nunique(), flush=True)
    pickle.dump(res, open(OUT / f"{scen}_{n_train}.pkl", "wb"))
    return res


def paired(ea, eb, group, rng, n=2000):
    """a - b of per-window errors, battle bootstrap. Returns mean, lo, hi."""
    df = pd.DataFrame({"g": group, "d": ea - eb}).groupby("g")["d"].agg(["sum", "size"])
    S, N = df["sum"].to_numpy(), df["size"].to_numpy()
    i = rng.integers(0, len(S), (n, len(S)))
    m = S[i].sum(1) / N[i].sum(1)
    return S.sum() / N.sum(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    scen, n_train = sys.argv[1], int(sys.argv[2])
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t0 = time.time()
    f = OUT / f"{scen}_{n_train}.pkl"
    res = pickle.load(open(f, "rb")) if f.exists() else build(scen, n_train, n_val)
    print("built", round(time.time() - t0), "s", flush=True)
    u = EXP[scen][1].lower()
    specs = {
        "T": ("team", (u,), "rule"),
        "FR-c": ("fr", (f"{u}_f", f"{u}_r"), "closure"),
        "FR-r": ("fr", (f"{u}_f", f"{u}_r"), "rule"),
        "P-c": ("rand", (f"{u}_f", f"{u}_r"), "closure"),
        "P-r": ("rand", (f"{u}_f", f"{u}_r"), "rule"),  # amendment 1 (post hoc)
    }
    models, batches, Ys, Yg = {}, {}, {}, {}
    for name, (h, types, alloc) in specs.items():
        L = PL.Layout(types)
        rows = PL.prepare(res[("train", h)], L)
        m = PL.Pipeline(types, alloc=alloc, movement_wd="rule", withdraw_radius=RADIUS[scen]).fit(rows)
        models[name] = m
        print(f"\n### {name}: {len(rows)} rows")
        for (s, t), c in m.coef.items():
            print(f"  H_{s}^{t}' =", "  ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v))
        print("  p", {k: (round(v[0], 3), round(v[1], 3)) for k, v in m.p.items()})
        if alloc == "closure":
            print("  target choice b0,b1,b2", {k: np.round(v, 2).tolist() for k, v in m.alloc_b.items()})
        print("  reach closure R2", {k: round(v, 3) for k, v in m.share_r2.items()})
        with np.errstate(all="ignore"):
            st = PL.add_controls(res[("val", h)])
            B = L.batch(st, horizon=HORIZON, every=10**9, history=0, min_future=8)
            Yt = m.rollout(B)
            Y = L.team_view(Yt)
            Yg[name] = Yt
        batches[name], Ys[name] = B, Y
    # frozen marines / marauders model on the same windows
    FZ = pickle.load(open(DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl", "rb"))
    B = batches["T"]
    with np.errstate(all="ignore"):
        Ys["frozen"] = FZ[scen].L.team_view(FZ[scen].rollout(B))
    truth = B["team"]["Y"]
    group = B["ep"] // 1000
    for n in batches:
        assert np.allclose(batches[n]["team"]["X0"][:, :4], B["team"]["X0"][:, :4]), n
    rng = np.random.default_rng(0)
    rows, pr = [], []
    for name, Y in Ys.items():
        r = {"model": name}
        for h in (1, 2, 4):
            i = int(h / DT)
            ok = ~np.isnan(truth[:, i, 0])
            e = np.abs(Y[ok, i, :2] - truth[ok, i, :2]).mean(1)
            r[f"health_{h}s"] = e.mean()
            r[f"alive_{h}s"] = np.abs(10 * (Y[ok, i, 2:4] - truth[ok, i, 2:4])).mean()
            dt = (Y[ok, i, 0] - Y[ok, i, 1]) - (B["team"]["X0"][ok, 0] - B["team"]["X0"][ok, 1])
            dr = (truth[ok, i, 0] - truth[ok, i, 1]) - (B["team"]["X0"][ok, 0] - B["team"]["X0"][ok, 1])
            r[f"mR2_{h}s"] = 1 - ((dt - dr) ** 2).sum() / ((dr - dr.mean()) ** 2).sum()
            r[f"n_{h}s"] = ok.sum()
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("model")
    print("\n", tab.round(4).to_string())
    for a, b in (
        ("FR-c", "T"),
        ("FR-c", "P-c"),
        ("FR-r", "T"),
        ("P-c", "T"),
        ("FR-c", "frozen"),
        ("T", "frozen"),
        ("FR-r", "P-r"),
        ("P-r", "T"),
        ("FR-r", "frozen"),
    ):
        for h in (1, 2, 4):
            i = int(h / DT)
            ok = ~np.isnan(truth[:, i, 0])
            f = lambda Y: np.abs(Y[ok, i, :2] - truth[ok, i, :2]).mean(1)  # noqa: B023, E731
            pr.append(
                {
                    "a": a,
                    "b": b,
                    "h": h,
                    **dict(
                        zip(("mean", "lo", "hi"), paired(f(Ys[a]), f(Ys[b]), group[ok], rng), strict=True)
                    ),
                }
            )
    pr = pd.DataFrame(pr)
    print("\nhealth error, a - b (95% battle bootstrap)\n", pr.round(4).to_string(index=False))
    # share of damage taken by the front, by distance at the window start (val, true data)
    L = PL.Layout(specs["FR-c"][1])
    Bf = batches["FR-c"]
    dmg = {}
    for s in PL.SIDES:
        hf, hr = (Bf["X0"][:, L.idx("health", s, t)] for t in L.types)
        i = int(2 / DT)
        yf, yr = (Bf["Y"][:, i, L.idx("health", s, t)] for t in L.types)
        ok = ~np.isnan(yf)
        lf, lr = (hf - yf)[ok], (hr - yr)[ok]
        dmg[s] = (float(lf.sum() / (lf + lr).sum()), float((hf[ok] / (hf + hr)[ok]).mean()))
    print(
        "\nfront's share of damage in 2 s (and its share of health at the start):",
        {k: tuple(np.round(v, 3)) for k, v in dmg.items()},
    )
    grp = {}
    for n in ("FR-c", "FR-r"):
        for s in PL.SIDES:
            for t in L.types:
                j = L.idx("health", s, t)
                for h in (2, 4):
                    i = int(h / DT)
                    ok = ~np.isnan(batches[n]["Y"][:, i, j])
                    grp[(n, s, t, h)] = (
                        np.abs(Yg[n][ok, i, j] - batches[n]["Y"][ok, i, j]).mean(),
                        (Yg[n][ok, i, j] - batches[n]["Y"][ok, i, j]).mean(),
                    )
    print("\nper-group health error (mean abs, signed), ten marines:")
    for k, v in grp.items():
        print("  ", *k, f"abs {v[0]:.4f} signed {v[1]:+.4f}")
    ratio = {
        n: {
            f"{s}:{t}:{k}": v / -RATE[scen]
            for (s, t), c in m.coef.items()
            for k, v in c.items()
            if v and k.startswith("R[")
        }
        for n, m in models.items()
    }
    print("\nk / game:", {n: [round(x, 2) for x in r.values()] for n, r in ratio.items()})
    json.dump(
        {
            "table": tab.to_dict(),
            "paired": pr.to_dict("records"),
            "damage": dmg,
            "group": {"/".join(map(str, k)): v for k, v in grp.items()},
            "ratio": ratio,
        },
        open(DOC / f"{scen}_{n_train}.json", "w"),
        default=float,
        indent=1,
    )
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
