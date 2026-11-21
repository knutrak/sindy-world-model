"""Train-only check of health compartments (healthy / wounded / dead), after 10_spread_state.md.

From the raw snapshots of each scenario's TRAIN set (exp06 marines, exp08 mixed): every 0.5 s, per side and
unit
type, the alive units are split into healthy (h ≥ 0.5) and wounded (h < 0.5), h = (hp + shield) / full.
Over the
next 1 s: kills, kills of units that were wounded at the start, damage (in full units of the type) and
damage on
the units wounded at the start. Windows where Blue's intent changes are left out.

    uv run --extra explore experiments/planning/compartments.py build   ->
    data/processed/planning/compartments_<S>.parquet
    uv run --extra explore experiments/planning/compartments.py check   -> docs/planning/11_compartments.md
"""

import argparse
import multiprocessing
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.models.intent_model import FAR_GATE  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR  # noqa: E402

SETS = {
    "S1": ("exp06_intents", ("MARINE",), "marines"),
    "S2": ("exp08_mixed", ("MARINE", "MARAUDER"), "mixed"),
}
OUT = DATA_DIR / "processed" / "planning"
DOC = PROJECT_ROOT / "docs" / "planning" / "11_compartments.md"
MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
W, EVERY, CUT = 8, 4, 0.5  # 1 s window, every 0.5 s, wounded below half health
N_BOOT = 2000


def windows(snaps: pd.DataFrame, types, relative: bool = False) -> list[dict]:
    """relative: a unit is wounded below CUT x its own start health (step 0), not CUT x full health."""
    rows = []
    for ep, g in snaps.groupby("episode_idx", sort=False):
        full = g["max_hp"] + g["max_shield"]
        g = g.assign(h=np.where(g["alive"], (g["hp"] + g["shield"]) / full, 0.0))
        h = g.pivot(index="step", columns=["team", "unit_id"], values="h").sort_index()
        al = g.pivot(index="step", columns=["team", "unit_id"], values="alive").sort_index().fillna(False)
        kind = g.groupby(["team", "unit_id"])["unit_type"].first()
        H, AL = h.to_numpy(), al.to_numpy().astype(bool)
        cols = list(h.columns)
        cut = CUT * (H[0] if relative else np.ones(H.shape[1]))
        for k in range(0, len(H) - W, EVERY):
            for side in ("blue", "red"):
                for t in types:
                    j = np.array([c[0] == side and kind[c] == t for c in cols])
                    a0, a1 = AL[k, j], AL[k + W, j]
                    if not a0.any():
                        continue
                    h0, h1, c = H[k, j], H[k + W, j], cut[j]
                    wd = a0 & (h0 < c)
                    dead = a0 & ~a1
                    dmg = np.clip(h0 - h1, 0, None) * a0
                    prev = max(k - W, 0)
                    rows.append(
                        {
                            "episode_idx": ep,
                            "step": int(h.index[k]),
                            "side": side,
                            "type": t.lower(),
                            "A0": a0.sum(),
                            "W0": wd.sum(),
                            "H0": h0[a0].sum(),
                            "kills": dead.sum(),
                            "kills_w": (dead & wd).sum(),
                            "dmg": dmg.sum(),
                            "dmg_w": dmg[wd].sum(),
                            "W1": (a1 & (h1 < c)).sum(),
                            "new_w": (a0 & ~wd & a1 & (h1 < c)).sum(),
                            "W_prev": (AL[prev, j] & (H[prev, j] < c)).sum(),
                            "HW0": h0[wd].sum(),
                        }
                    )
    return rows


def build_one(job) -> pd.DataFrame:
    exp, name, types, relative = job
    d = RAW_DIR / exp / name
    ids = storage.load_episode_index(d)["episode_idx"].tolist()
    parts = []
    for c in np.array_split(np.asarray(ids), 20):
        parts += windows(storage.load_snapshots(d, c.tolist()), types, relative)
    out = pd.DataFrame(parts)
    out["episode_idx"] += int(name.split("_")[-1]) * 1000  # as the per-type tables
    return out


def build(relative: bool = False) -> None:
    for scen, (exp, types, _) in SETS.items():
        shards = sorted(p.name for p in (RAW_DIR / exp).glob("train_*") if p.name.split("_")[-1].isdigit())
        with multiprocessing.get_context("spawn").Pool(len(shards)) as pool:
            parts = pool.map(build_one, [(exp, s, types, relative) for s in shards])
        t = pd.concat(parts)
        st = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "train.parquet")
        st = st[["movement", "target", "segment", "distance"]].reset_index()
        nxt = st.assign(step=st["step"] - W)[["episode_idx", "step", "segment"]].rename(
            columns={"segment": "seg1"}
        )
        prv = st.assign(step=st["step"] + W)[["episode_idx", "step", "target"]].rename(
            columns={"target": "target_prev"}
        )
        t = t.merge(st, on=["episode_idx", "step"], how="left").merge(
            nxt, on=["episode_idx", "step"], how="left"
        )
        t = t.merge(prv, on=["episode_idx", "step"], how="left")
        t["steady"] = t["segment"] == t["seg1"]
        t.to_parquet(OUT / f"compartments{'_rel' if relative else ''}_{scen}.parquet")
        print(f"{scen}: {len(t)} windows, {t['steady'].mean():.0%} with a steady intent", flush=True)


# ------------------------------------------------------------------ check
def cv_r2(X: np.ndarray, y: np.ndarray, ep: np.ndarray, fit, predict, folds: int = 5) -> float:
    pred = np.zeros_like(y, dtype=float)
    for k in range(folds):
        te = ep % folds == k
        pred[te] = predict(fit(X[~te], y[~te]), X[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def lin_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def lin_pred(c, X):
    return X @ c


def power_fit(X, y):
    """kills = A0 (1 − (1 − dmg/H0)^(p + pf·focus)), p and pf by a grid then refinement."""
    from scipy.optimize import least_squares

    def res(q):
        return power_pred(q, X) - y

    return least_squares(res, x0=[0.6, 0.2], bounds=([0, -2], [5, 5])).x


def power_pred(q, X):
    A0, frac, focus = X[:, 0], np.clip(X[:, 1], 0, 1), X[:, 2]
    return A0 * (1 - (1 - frac) ** np.clip(q[0] + q[1] * focus, 0, None))


def boot_mean_diff(a: pd.Series, b: pd.Series, ga, gb, rng) -> str:
    ua, ub = np.unique(ga), np.unique(gb)
    m = []
    for _ in range(N_BOOT):
        sa = rng.choice(ua, len(ua))
        sb = rng.choice(ub, len(ub))
        m.append(a[np.isin(ga, sa)].mean() - b[np.isin(gb, sb)].mean())
    lo, hi = np.quantile(m, [0.025, 0.975])
    return f"{a.mean() - b.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"


def check(min_start: float = 0.0) -> None:
    """min_start > 0: only sides whose units all started with at least that fraction of full health (the
    samplers start most battles damaged, every unit of a side at the same fraction)."""
    rng = np.random.default_rng(0)
    lines = [
        "# Health compartments: a train-only check",
        "",
        "Script: `experiments/planning/compartments.py` (set-up in its docstring). Train sets only, "
        "windows of 1 s",
        "every 0.5 s with a steady Blue intent, both sides alive. Wounded: below half health. Damage "
        "in full units.",
        "Exploratory, like `05_intent_residuals.md`: no model is changed and nothing is tested on val or cf.",
        "",
    ]
    for scen in SETS:
        t = pd.read_parquet(OUT / f"compartments_{scen}.parquet")
        sh = pd.read_parquet(OUT / f"start_health_{SETS[scen][0]}.parquet")
        t = t.merge(sh, on="episode_idx", how="left")
        t["start_health"] = np.where(t["side"] == "blue", t["start_health_blue"], t["start_health_red"])
        t = t[t["start_health"] >= min_start]
        t = t[t["steady"] & (t["A0"] > 0)].copy()
        t["focus"] = (t["target"] == "focus").astype(float)
        t["contact"] = t["distance"] < FAR_GATE[0]
        t = t[t["movement"] != "withdraw"]
        lines += [f"## {scen}", ""]
        # 1. where kills come from
        rows = []
        for (side, typ), g in t.groupby(["side", "type"]):
            for f in (0.0, 1.0):
                x = g[g["focus"] == f]
                rows.append(
                    {
                        "side": side,
                        "type": typ,
                        "Blue target": "focus" if f else "nearest",
                        "windows": len(x),
                        "kills per window": x["kills"].mean(),
                        "share of kills from units wounded at the start": x["kills_w"].sum()
                        / max(x["kills"].sum(), 1),
                        "share of alive units wounded": x["W0"].sum() / x["A0"].sum(),
                    }
                )
        lines += [
            "### 1. Where kills come from",
            "",
            "```",
            pd.DataFrame(rows).set_index(["side", "type", "Blue target"]).round(3).to_string(),
            "```",
            "",
        ]
        # 2. kill laws (CV R² of kills in the next 1 s, 5 folds by battle)
        rows = []
        for (side, typ), g in t[t["contact"]].groupby(["side", "type"]):
            y, ep = g["kills"].to_numpy(float), g["episode_idx"].to_numpy()
            frac = (g["dmg"] / g["H0"].clip(lower=1e-6)).to_numpy()
            Xp = np.column_stack([g["A0"], frac, g["focus"]])
            wsh = (g["W0"] / g["A0"]).to_numpy()
            d, dw, fo = g["dmg"].to_numpy(), g["dmg_w"].to_numpy(), g["focus"].to_numpy()
            X_state = np.column_stack([d, d * fo, d * wsh, d * wsh * fo, g["W0"], g["W0"] * fo])
            X_split = np.column_stack([dw, dw * fo, d - dw, (d - dw) * fo])
            rows.append(
                {
                    "side": side,
                    "type": typ,
                    "power law (A0, H loss, focus)": cv_r2(Xp, y, ep, power_fit, power_pred),
                    "compartment: damage × wounded share, W0 (model can know)": cv_r2(
                        X_state, y, ep, lin_fit, lin_pred
                    ),
                    "where damage lands, linear (needs per-unit fire)": cv_r2(
                        X_split, y, ep, lin_fit, lin_pred
                    ),
                    "both: power law + wounded share": cv_r2(np.column_stack([Xp, wsh]), y, ep, *power_w()),
                    "windows": len(g),
                }
            )
        lines += [
            "### 2. Kill laws in contact (CV R² of kills in the next 1 s)",
            "",
            "```",
            pd.DataFrame(rows).set_index(["side", "type"]).round(4).to_string(),
            "```",
            "",
        ]
        # 3. focus and the wounded compartment (Red: the side Blue's target rule acts on), in contact
        rows = []
        for (side, typ), g in t[t["contact"]].groupby(["side", "type"]):
            a, b = g[g["focus"] == 1], g[g["focus"] == 0]
            for col, label in (
                ("W0", "wounded at the start"),
                ("new_w", "healthy → wounded in 1 s"),
                ("kills_w", "wounded → dead in 1 s"),
                ("kills", "kills in 1 s"),
            ):
                rows.append(
                    {
                        "side": side,
                        "type": typ,
                        "quantity": label,
                        "focus − nearest [95%, bootstrap by battle]": boot_mean_diff(
                            a[col], b[col], a["episode_idx"].to_numpy(), b["episode_idx"].to_numpy(), rng
                        ),
                    }
                )
        lines += [
            "### 3. Focus and the wounded compartment (in contact; focus − nearest)",
            "",
            "```",
            pd.DataFrame(rows).set_index(["side", "type", "quantity"]).to_string(),
            "```",
            "",
        ]
        # 4. memory: does the previous second's target rule still matter, without / with W0?
        rows = []
        for (side, typ), g in t[t["contact"] & t["target_prev"].notna()].groupby(["side", "type"]):
            y = g["kills"].to_numpy(float)
            d, fo = g["dmg"].to_numpy(), g["focus"].to_numpy()
            fp = (g["target_prev"] == "focus").to_numpy(float)
            base = np.column_stack([d, d * fo, d * fp, np.ones_like(d)])
            withw = np.column_stack([base, d * g["W0"] / g["A0"], g["W0"]])
            c0, c1 = lin_fit(base, y), lin_fit(withw, y)
            rows.append(
                {
                    "side": side,
                    "type": typ,
                    "previous focus × damage, without W0": c0[2],
                    "with W0": c1[2],
                    "current focus × damage, without W0": c0[1],
                    "with W0 ": c1[1],
                    "windows": len(g),
                }
            )
        lines += [
            "### 4. Memory: the previous second's focus, without and with the wounded count",
            "",
            "```",
            pd.DataFrame(rows).set_index(["side", "type"]).round(4).to_string(),
            "```",
            "",
        ]
    if min_start > 0:
        lines.insert(5, f"**Only sides that started with at least {min_start:.0%} of full health.**")
    (
        DOC if min_start == 0 else DOC.with_name(f"11_compartments_start{round(100 * min_start)}.md")
    ).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def flows(min_start: float = 0.0) -> None:
    """Can the flows healthy -> wounded and wounded -> dead (relative wounded) be predicted from team-level
    quantities? CV R² over 1 s windows in contact, with the measured damage d or the frozen model's d-hat."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spread_state import health_rate

    frozen = pickle.load(open(MODELS, "rb"))
    lines = [
        "# Health compartments: the flows (relative wounded, train only)",
        "",
        "Script: `experiments/planning/compartments.py flows`. A unit is wounded once below half of "
        "its own start",
        "health. Windows of 1 s every 0.5 s, steady Blue intent, in contact, not withdrawing; CV R² "
        "over 5 folds",
        "by battle. d = measured damage on the side's type in the window (full units); d-hat = the "
        "frozen health",
        "equation's rate at the window's start × 1 s (what a rollout would have). w = W0 / A0 "
        "(wounded share).",
        "Exploratory: no model is changed and nothing is tested on val or cf.",
        "",
    ]
    if min_start > 0:
        lines.insert(7, f"**Only sides that started with at least {min_start:.0%} of full health.**")
    for scen, (exp, _types, name) in SETS.items():
        m = frozen[name]
        t = pd.read_parquet(OUT / f"compartments_rel_{scen}.parquet")
        sh = pd.read_parquet(OUT / f"start_health_{exp}.parquet")
        t = t.merge(sh, on="episode_idx", how="left")
        t["start_health"] = np.where(t["side"] == "blue", t["start_health_blue"], t["start_health_red"])
        t = t[t["steady"] & (t["A0"] > 0) & (t["movement"] != "withdraw") & (t["distance"] < FAR_GATE[0])]
        t = t[t["start_health"] >= min_start].copy()
        t["focus"] = (t["target"] == "focus").astype(float)
        rows = pd.read_parquet(DATA_DIR / "processed" / f"{exp}_typed" / "rates_train_w9.parquet")
        dh = []
        for side in ("blue", "red"):
            for typ in m.L.types:
                rate = health_rate(m, rows, side, typ)
                dh.append(
                    pd.DataFrame(
                        {
                            "episode_idx": rows.index.get_level_values(0),
                            "step": rows.index.get_level_values(1),
                            "side": side,
                            "type": typ,
                            "dhat": np.maximum(-rate, 0) * 10 * 1.0,
                        }
                    )
                )
        t = t.merge(pd.concat(dh), on=["episode_idx", "step", "side", "type"], how="inner")
        res, eff = [], []
        for (side, typ), g in t.groupby(["side", "type"]):
            ep, fo = g["episode_idx"].to_numpy(), g["focus"].to_numpy()
            A0, W0 = g["A0"].to_numpy(float), g["W0"].to_numpy(float)
            w = W0 / A0
            one = np.ones(len(g))
            ys = {
                "healthy → wounded": g["new_w"].to_numpy(float),
                "wounded → dead": g["kills_w"].to_numpy(float),
                "healthy → dead": (g["kills"] - g["kills_w"]).to_numpy(float),
                "all kills": g["kills"].to_numpy(float),
            }
            row = {"side": side, "type": typ, "windows": len(g)}
            for dn, d in (("d", g["dmg"].to_numpy()), ("d-hat", g["dhat"].to_numpy())):
                Xn = np.column_stack([d, d * fo, one])
                Xc = np.column_stack([d * (1 - w), d * (1 - w) * fo, d * w, d * w * fo, W0, W0 * fo, one])
                for yn, y in ys.items():
                    row[f"{yn} | {dn}, no state"] = cv_r2(Xn, y, ep, lin_fit, lin_pred)
                    row[f"{yn} | {dn}, compartments"] = cv_r2(Xc, y, ep, lin_fit, lin_pred)
                if dn == "d-hat":
                    wm = float(np.median(w))
                    for yn in ("healthy → wounded", "wounded → dead"):
                        c = lin_fit(Xc, ys[yn])
                        near = c[0] * (1 - wm) + c[2] * wm
                        foc = near + c[1] * (1 - wm) + c[3] * wm
                        eff.append(
                            {
                                "side": side,
                                "type": typ,
                                "flow": yn,
                                "median w": wm,
                                "per unit of d-hat, nearest": near,
                                "focus": foc,
                                "per wounded unit, nearest": c[4],
                                "focus ": c[4] + c[5],
                            }
                        )
            dw, dhl = g["dmg_w"].to_numpy(), (g["dmg"] - g["dmg_w"]).to_numpy()
            row["wounded → dead | where damage lands (per-unit)"] = cv_r2(
                np.column_stack([dw, dw * fo, one]), ys["wounded → dead"], ep, lin_fit, lin_pred
            )
            row["healthy → wounded | where damage lands (per-unit)"] = cv_r2(
                np.column_stack([dhl, dhl * fo, one]), ys["healthy → wounded"], ep, lin_fit, lin_pred
            )
            frac = g["dmg"].to_numpy() / g["H0"].clip(lower=1e-6).to_numpy()
            row["all kills | d, power law"] = cv_r2(
                np.column_stack([A0, frac, fo]), ys["all kills"], ep, power_fit, power_pred
            )
            res.append(row)
        R = pd.DataFrame(res).set_index(["side", "type"]).T
        lines += [
            f"## {scen}",
            "",
            "CV R² per flow and predictor set:",
            "",
            "```",
            R.round(3).to_string(),
            "```",
            "",
            "Fitted flows (compartment set, d-hat), at the median wounded share:",
            "",
            "```",
            pd.DataFrame(eff).set_index(["side", "type", "flow"]).round(4).to_string(),
            "```",
            "",
        ]
    out = DOC.with_name(
        "12_compartment_flows" + (f"_start{round(100 * min_start)}" if min_start else "") + ".md"
    )
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def power_w():
    """Power law with the exponent p + pf·focus + q·wounded share."""
    from scipy.optimize import least_squares

    def pred(q, X):
        A0, frac, focus, w = X[:, 0], np.clip(X[:, 1], 0, 1), X[:, 2], X[:, 3]
        return A0 * (1 - (1 - frac) ** np.clip(q[0] + q[1] * focus + q[2] * w, 0, None))

    def fit(X, y):
        return least_squares(lambda q: pred(q, X) - y, x0=[0.6, 0.2, 0.0]).x

    return fit, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["build", "build-rel", "check", "flows"])
    parser.add_argument("--min-start", type=float, default=0.0)
    a = parser.parse_args()
    if a.what.startswith("build"):
        build(relative=a.what == "build-rel")
    else:
        (check if a.what == "check" else flows)(a.min_start)


if __name__ == "__main__":
    main()
