"""The kite variant (approved plan: docs/planning/32_kite_refit_plan.md): fit and checks K1-K3.

uv run --extra explore experiments/planning/kite_refit.py fit     exp08 train + exp11 train -> kite_model.pkl
uv run --extra explore experiments/planning/kite_refit.py k1      exp08 val: kite variant vs frozen
uv run --extra explore experiments/planning/kite_refit.py k2      exp11 val: kite vs advance windows; reach
uv run --extra explore experiments/planning/kite_refit.py k3      counterfactual sets, seven candidates
uv run --extra explore experiments/planning/kite_refit.py report  -> docs/planning/33_kite_refit.md
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_regret import margin_change  # noqa: E402

from sindy_wm.models.intent_model import DT, SHAPE, add_controls  # noqa: E402
from sindy_wm.models.kite import CTRL_K, KitePipeline, add_kite  # noqa: E402
from sindy_wm.models.pipeline import SIDES, prepare  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

P = DATA_DIR / "processed" / "planning"
FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
OUT = P / "kite_model.pkl"
RES = P / "kite_results.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "33_kite_refit.md"
E08 = DATA_DIR / "processed" / "exp08_mixed_typed"
E11 = DATA_DIR / "processed" / "exp11_kite_typed"
CONFIG = PROJECT_ROOT / "configs" / "generalisation_pipeline.yaml"
SPEC = ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd")
OFFSET = 100_000  # exp11 episode ids, so they never clash with exp08's
BRANCHES = (
    "advance_nearest",
    "advance_focus",
    "hold_nearest",
    "hold_focus",
    "withdraw_nearest",  # withdraw/focus plays the same (Blue doesn't fire); not built, as in `02`-`06`
    "kite_nearest",
    "kite_focus",
)
HORIZONS = (1.0, 2.0, 4.0)
N_BOOT = 2000


def _res() -> dict:
    return json.load(open(RES)) if RES.exists() else {}


def _save(key, value) -> None:
    r = _res()
    r[key] = value
    json.dump(r, open(RES, "w"), indent=1, default=float)


def models() -> dict:
    return {"frozen": pickle.load(open(FROZEN, "rb"))["mixed"], "kite": pickle.load(open(OUT, "rb"))}


def batch(m, states, **kw):
    with np.errstate(invalid="ignore", divide="ignore"):
        return m.L.batch(states, ctrl=CTRL_K if isinstance(m, KitePipeline) else None, **kw)


def boot(x, grp, rng):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((w * sums).sum() / (w * cnt).sum())
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def equations(m) -> str:
    side = {"blue": "B", "red": "R"}
    lines = []
    for (s, t), c in m.coef.items():
        lines.append(f"H_{side[s]}^{t}' = " + " ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v))
    for (s, t), (p, pf) in m.p.items():
        f = f" {pf:+.3f}·focus" if pf else ""
        lines.append(f"(ln A_{side[s]}^{t})' = ({p:.3f}{f}) · (ln H_{side[s]}^{t})'")
    d = " ".join(f"{c:+.3f}·{n}" for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c)
    lines.append(f"d' = {d}")
    lines.append(
        "kite gate (v, c, bb, br, eb, er, w, a): " + ", ".join(f"{x:.3f}" for x in m.P_team["gates"]["kite"])
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ fit
def fit() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    kw = {k: cfg["pipeline"][k] for k in SPEC}
    s = cfg["scenarios"]["mixed"]
    m = KitePipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **kw)
    r08 = pd.read_parquet(E08 / "rates_train_w9.parquet")
    r08["kite"] = 0.0
    r11 = add_kite(prepare(pd.read_parquet(E11 / "train.parquet"), m.L)).reset_index()
    r11["episode_idx"] += OFFSET
    rows = pd.concat([r08, r11.set_index(["episode_idx", "step"])])
    assert (rows.loc[rows["movement"] == "kite", ["advance", "hold", "withdraw"]] == 0).all().all()
    m.fit(rows)
    pickle.dump(m, open(OUT, "wb"))
    share = {mv: float((rows["movement"] == mv).mean()) for mv in ("advance", "hold", "withdraw", "kite")}
    _save("fit", {"equations": equations(m), "step shares": share, "rows": len(rows)})
    print(equations(m))
    print(share)


def diag() -> None:
    """Why the kite terms were dropped: least squares of each health equation on its kept terms plus the kite
    terms (no threshold); kite coefficient, its size relative to the matching base term, and its standardised
    effect size against the threshold (0.075). Train rows, as in the fit."""
    m = pickle.load(open(OUT, "rb"))
    r08 = pd.read_parquet(E08 / "rates_train_w9.parquet")
    r08["kite"] = 0.0
    r11 = add_kite(prepare(pd.read_parquet(E11 / "train.parquet"), m.L)).reset_index()
    r11["episode_idx"] += OFFSET
    rows = pd.concat([r08, r11.set_index(["episode_idx", "step"])])
    m._u = rows[CTRL_K].to_numpy(float)
    tt, out = m.L.types, []
    for s in SIDES:
        o = "red" if s == "blue" else "blue"
        R_o = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in tt}
        A_o = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in tt}
        sig = m._sigma_measured(rows, o)
        for t in tt:
            L = m._terms(R_o, sig, A_o, rows["withdraw"].to_numpy(), o, t)
            base = [k for k, c in m.coef[(s, t)].items() if c]
            names = base + [f"{k}*kite" for k in base]
            X = np.column_stack([L[k] for k in names])
            y = rows[f"d_health_{s}_{t}"].to_numpy()
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            eff = c * X.std(0) / y.std()
            for j, k in enumerate(names[len(base) :]):
                b = c[base.index(k.removesuffix("*kite"))]
                out.append(
                    {
                        "equation": f"H_{s}^{t}",
                        "term": k,
                        "coef": c[len(base) + j],
                        "relative to base": c[len(base) + j] / b,
                        "standardised": eff[len(base) + j],
                    }
                )
    m._u = None
    _save("diag", out)
    print(pd.DataFrame(out).round(4).to_string(index=False))


# ------------------------------------------------------------------ K1, K2
def _errors(Y, T, n):
    """team health error (both sides, ten full units) and distance error at step n; valid rows."""
    ok = ~np.isnan(T[:, n, 0])
    h = np.abs(Y[:, n, :2] - T[:, n, :2]).sum(1)
    d = np.abs(Y[:, n, 4] - T[:, n, 4])
    return ok, h, d


def k1() -> None:
    rng = np.random.default_rng(0)
    M = models()
    st = add_kite(add_controls(pd.read_parquet(E08 / "val.parquet")))
    B = {k: batch(m, st) for k, m in M.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = {k: m.L.team_view(m.rollout(B[k])) for k, m in M.items()}
    T, ep = B["frozen"]["team"]["Y"], B["frozen"]["ep"]
    rows = []
    for h in HORIZONS:
        n = round(h / DT)
        ok, hf, df = _errors(Y["frozen"], T, n)
        _, hk, dk = _errors(Y["kite"], T, n)
        for name, a, b in (("team health", hf, hk), ("distance", df, dk)):
            mean, lo, hi = boot(b[ok] - a[ok], ep[ok], rng)
            ratio = b[ok].mean() / a[ok].mean()
            rows.append(
                {
                    "h": h,
                    "error": name,
                    "frozen": a[ok].mean(),
                    "kite": b[ok].mean(),
                    "ratio": ratio,
                    "difference": mean,
                    "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                    "passes (ratio <= 1.05)": bool(ratio <= 1.05),
                }
            )
    _save("k1", rows)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def _windows(B, mv: int, n: int) -> np.ndarray:
    """Start points whose controls keep one movement for the whole n steps."""
    return (B["U"][:, :n, mv] == 1).all(1)


def k2() -> None:
    rng = np.random.default_rng(0)
    M = models()
    st = add_kite(add_controls(pd.read_parquet(E11 / "val.parquet")))
    B = {k: batch(m, st, every=4) for k, m in M.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = {k: m.L.team_view(m.rollout(B[k])) for k, m in M.items()}
    T, ep = B["kite"]["team"]["Y"], B["kite"]["ep"]
    rows = []
    for h in HORIZONS:
        n = round(h / DT)
        for name in ("kite", "frozen"):
            ok, he, de = _errors(Y[name], T, n)
            for mv, j in (("advance", 0), ("kite", 4)):
                w = ok & _windows(B["kite"], j, n)
                rows.append(
                    {
                        "h": h,
                        "model": name,
                        "window": mv,
                        "starts": int(w.sum()),
                        "team health error": he[w].mean(),
                        "distance error": de[w].mean(),
                    }
                )
    t = pd.DataFrame(rows)
    k = t[(t.model == "kite") & (t.h == 2.0)].set_index("window")["team health error"]
    ratio = float(k["kite"] / k["advance"])
    # the reach closure on the steps themselves, by movement
    m = M["kite"]
    L = m.L
    r = st.dropna(subset=L.x_cols)
    X, S, F = (r[c].to_numpy(float) for c in (L.x_cols, SHAPE, L.front_cols))
    sh = m._shares(X, S, np.nan_to_num(F))
    reach = []
    for mv in ("advance", "kite"):
        sel = (r["movement"] == mv).to_numpy()
        for s in SIDES:
            pred = sum(X[sel, L.idx("alive", s, a)] * sh[(s, a)][sel] for a in L.types)
            meas = sum(r[f"reach_{s}_{a}"].to_numpy()[sel] for a in L.types)
            reach.append(
                {
                    "movement": mv,
                    "side": s,
                    "measured": meas.mean(),
                    "closure": pred.mean(),
                    "mean abs error": np.abs(pred - meas).mean(),
                }
            )
    _save("k2", {"table": rows, "ratio at 2 s": ratio, "passes (<= 1.5)": ratio <= 1.5, "reach": reach})
    print(t.round(4).to_string(index=False))
    print(f"kite / advance team health error at 2 s: {ratio:.3f} (passes if <= 1.5)")
    print(pd.DataFrame(reach).round(4).to_string(index=False))
    del rng, ep


# ------------------------------------------------------------------ K3
def load_cf() -> pd.DataFrame:
    parts = []
    for b, name in enumerate(BRANCHES):
        folder = E11 if name.startswith("kite") else E08
        d = pd.read_parquet(folder / f"cf_{name}.parquet").reset_index()
        d["branch"] = b
        d["episode_idx"] = d["episode_idx"] + 10_000 * (b + 1)
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_kite(add_controls(pd.concat(parts)))


def from_branch(states, L) -> tuple[pd.DataFrame, float]:
    blue = [L.x_cols[L.idx("alive", "blue", t)] for t in L.types]
    red = [L.x_cols[L.idx("alive", "red", t)] for t in L.types]
    parts, first = [], []
    for _, g in states.groupby(level=0, sort=False):
        k = int(round(g["branch_at"].iloc[0] / DT))
        if k < len(g) - 8 and g[blue].iloc[k].sum() > 0 and g[red].iloc[k].sum() > 0:
            parts.append(g.iloc[k:])
            first.append(g.iloc[[k]])
    out, f = pd.concat(parts), pd.concat(first)
    n = out.groupby("seed")["branch"].nunique()
    keep = n[n == len(BRANCHES)].index
    f = f[f["seed"].isin(keep)]
    gap = float(f.groupby("seed")[L.x_cols].agg(lambda x: x.max() - x.min()).to_numpy().max())
    return out[out["seed"].isin(keep)], gap


def k3() -> None:
    rng = np.random.default_rng(0)
    M = models()
    states, gap = from_branch(load_cf(), M["kite"].L)
    B = {k: batch(m, states, horizon=64, every=10**9, history=0, min_future=1) for k, m in M.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = {k: m.L.team_view(m.rollout(B[k])) for k, m in M.items()}
    TY = B["kite"]["team"]["Y"]
    g = states.groupby(level=0, sort=False)
    seed = g["seed"].first().loc[B["kite"]["ep"]].to_numpy()
    branch = g["branch"].first().loc[B["kite"]["ep"]].to_numpy()

    def wide(v):
        return (
            pd.DataFrame({"s": seed, "b": branch, "v": v})
            .pivot(index="s", columns="b", values="v")
            .to_numpy()
        )

    rows, paired = [], []
    for h in HORIZONS:
        T = wide(margin_change(TY, TY, h))
        idx = np.arange(len(T))
        picks = {n: wide(margin_change(Yn, TY, h)).argmax(1) for n, Yn in Y.items()}
        picks |= {f"always {BRANCHES[j]}": np.full(len(T), j) for j in (0, 1, 6)}
        fixed = int(T.mean(0).argmax())
        picks[f"best fixed in hindsight ({BRANCHES[fixed]})"] = np.full(len(T), fixed)
        reg = {n: T.max(1) - T[idx, p] for n, p in picks.items()}
        for n, p in picks.items():
            rows.append(
                {
                    "h": h,
                    "ranker": n,
                    "regret": reg[n].mean(),
                    "picks kite": float(np.isin(p, [5, 6]).mean()),
                    "picks the true best": float((reg[n] <= 1e-9).mean()),
                }
            )
        for a, b in (("kite", "always kite_focus"), ("kite", "frozen")):
            d = reg[a] - reg[b]
            mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            lo, hi = np.quantile(mm, [0.025, 0.975])
            paired.append(
                {"h": h, "pair": f"{a} − {b}", "difference": d.mean(), "95%": f"[{lo:+.4f}, {hi:+.4f}]"}
            )
        rows[-1]["true best is kite"] = float(np.isin(T.argmax(1), [5, 6]).mean())
    t, pt = pd.DataFrame(rows), pd.DataFrame(paired)
    k = t.set_index(["h", "ranker"])["regret"]
    passes = any(k[(h, "kite")] < k[(h, "always kite_focus")] for h in (2.0, 4.0))
    _save("k3", {"table": rows, "paired": paired, "states": int(len(T)), "branch gap": gap, "passes": passes})
    print(f"states: {len(T)}, largest state difference at the branch: {gap:.2e}")
    print(t.round(4).to_string(index=False))
    print(pt.round(4).to_string(index=False))
    print("K3 passes:", passes)


# ------------------------------------------------------------------ report
def report() -> None:
    r = _res()
    md = lambda t: "```\n" + pd.DataFrame(t).round(4).to_string(index=False) + "\n```"  # noqa: E731
    lines = [
        "# The kite variant: fit and checks K1-K3",
        "",
        "Plan: `32` (approved 22 Nov, rules fixed before collection).",
        "Script: `experiments/planning/kite_refit.py`.",
        "Fitted on exp08 train + exp11 train (kite in the schedule).",
        "",
    ]
    if "fit" in r:
        lines += [
            "## Equations",
            "",
            "```",
            r["fit"]["equations"],
            "```",
            "",
            f"Share of training steps by movement: {r['fit']['step shares']}; rows {r['fit']['rows']}.",
            "",
        ]
    if "diag" in r:
        lines += [
            "## Why the kite terms were dropped (diagnostic, after the fit)",
            "",
            "Least squares on each equation's kept terms plus their kite terms, no threshold. "
            "'relative to base': kite coefficient / the matching base coefficient (−0.2 = 20% less "
            "fire under kite); 'standardised': the effect size STLSQ compares with 0.075.",
            "",
            md(r["diag"]),
            "",
        ]
    if "k1" in r:
        lines += ["## K1: exp08 val, kite variant vs frozen (error at the horizon)", "", md(r["k1"]), ""]
    if "k2" in r:
        lines += [
            "## K2: exp11 val, kite and advance windows",
            "",
            md(r["k2"]["table"]),
            "",
            f"Kite / advance team health error at 2 s (kite variant): {r['k2']['ratio at 2 s']:.3f}; "
            f"passes (<= 1.5): {r['k2']['passes (<= 1.5)']}.",
            "",
            "Reach closure on the steps:",
            "",
            md(r["k2"]["reach"]),
            "",
        ]
    if "k3" in r:
        lines += [
            "## K3: counterfactual sets, seven candidates (exp08 cf seeds)",
            "",
            f"States: {r['k3']['states']}; largest state difference at the branch "
            f"{r['k3']['branch gap']:.1e}.",
            "",
            md(r["k3"]["table"]),
            "",
            md(r["k3"]["paired"]),
            "",
            f"K3 passes: {r['k3']['passes']}.",
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["fit", "diag", "k1", "k2", "k3", "report"])
    {"fit": fit, "diag": diag, "k1": k1, "k2": k2, "k3": k3, "report": report}[p.parse_args().what]()


if __name__ == "__main__":
    main()
