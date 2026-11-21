"""Exploratory: the frozen model with more intent terms (after 05_intent_residuals.md), S1 and S2.

Three additions, each fitted on the scenario's TRAIN set only, with the frozen model's own terms left as
they are:

  focus cost   Blue's reach in Red's health loss × (1 − c·focus)/(1 − c·f̄), c and f̄ measured on train
               (docs/planning/04_focus_variant.md, "compensated")
  distance     the frozen d' plus focus·(1 − wd)·contact and focus·(1 − wd)·(1 − contact), STLSQ (0.05)
               on the frozen d' residual; contact as in the movement library
  shape        depth and width per side evolve in the rollout (the frozen model holds them): STLSQ (0.05)
               on the older team model's shape library (drift and relaxation per zone far/contact and
               movement); rates by Savitzky-Golay (window 9), as the other rates

Variants: frozen, each addition alone, and all three. Evaluated on val (prediction, as the generalisation
report) and on the counterfactual sets (regret, as 02 and 03).

    uv run --extra explore experiments/planning/intent_variant.py --fit  -> equations, saved variants
    uv run --extra explore experiments/planning/intent_variant.py  -> docs/planning/06_intent_variant.md
"""

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_checks import check2  # noqa: E402
from focus_variant import focus_cost, regret_rows  # noqa: E402
from model_regret import MODELS, SETS, from_branch, load  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import (  # noqa: E402
    CTRL,
    SHAPE,
    _shape_library,
    add_controls,
    add_rates,
    r2,
    stlsq,
)
from sindy_wm.models.pipeline import intent_features, team_rows  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "06_intent_variant.md"
OUT = DATA_DIR / "processed" / "planning" / "intent_variants.pkl"
THR = 0.05


def train_rows(folder: str) -> pd.DataFrame:
    rows = pd.read_parquet(DATA_DIR / "processed" / folder / "rates_train_w9.parquet")
    states = add_controls(pd.read_parquet(DATA_DIR / "processed" / folder / "train.parquet"))
    rates = add_rates(states[SHAPE + ["distance", "t"]], cols=tuple(SHAPE), window=9)[
        [f"d_{c}" for c in SHAPE]
    ]
    out = rows.join(rates, how="inner")
    assert len(out) == len(rows), "rates rows and states must share the (episode, step) index"
    return out


def fit_extra(m, rows: pd.DataFrame) -> tuple[dict, dict]:
    T = team_rows(rows, m.L)
    Xt = np.column_stack(
        [T[c].to_numpy() for c in ("health_blue", "health_red", "alive_blue", "alive_red", "distance")]
    )
    U, S = T[CTRL].to_numpy(), T[SHAPE].to_numpy()
    F = m._dist_library(Xt, U, S, T[["reach_blue", "reach_red"]].to_numpy())
    resid = T["d_distance"].to_numpy() - np.column_stack([F[k] for k in m.dist_names]) @ m.dist_coef
    G = intent_features(F, Xt, U)
    names = list(G)
    theta = np.column_stack([G[k] for k in names])
    c = stlsq(theta, resid, THR)
    fit = {"distance residual R²": r2(resid, theta @ c)}
    extra = {"dist": (names, c), "shape": {}}
    for j, col in enumerate(SHAPE):
        Fs = _shape_library(Xt[:, -1], S[:, j], U)
        n = list(Fs)
        th = np.column_stack([Fs[k] for k in n])
        y = T[f"d_{col}"].to_numpy()
        cs = stlsq(th, y, THR)
        extra["shape"][j] = (n, cs)
        fit[f"{col} R²"] = r2(y, th @ cs)
    return extra, fit


def equations(extra: dict) -> str:
    def terms(names, c):
        return " ".join(f"{v:+.4f}·{k}" for k, v in zip(names, c, strict=True) if v) or "0"

    lines = [f"d' = (frozen d') {terms(*extra['dist'])}"]
    for j, (n, c) in extra["shape"].items():
        lines.append(f"{SHAPE[j]}' = {terms([k.replace('a*', SHAPE[j] + '·') for k in n], c)}")
    return "\n".join(lines)


def variants(m, extra: dict, c: float, fbar: float) -> dict:
    def make(focus=False, dist=False, shape=False):
        v = copy.deepcopy(m)
        if focus:
            v.focus_cost, v.focus_norm = c, 1 - c * fbar
        if dist or shape:
            v.intent_extra = {
                "dist": extra["dist"] if dist else None,
                "shape": extra["shape"] if shape else None,
            }
        return v

    return {
        "frozen": m,
        "+focus cost": make(focus=True),
        "+distance focus": make(dist=True),
        "+shape": make(shape=True),
        "all three": make(True, True, True),
    }


def fit_all() -> dict:
    frozen = pickle.load(open(MODELS, "rb"))
    out = {}
    for scen, (folder, name) in SETS.items():
        m = frozen[name]
        extra, fit = fit_extra(m, train_rows(folder))
        c, fbar = focus_cost(folder, m.L.types)
        out[scen] = {"extra": extra, "fit": fit, "c": c, "fbar": fbar}
        print(f"## {scen} ({name}): focus cost c = {c:.4f}, f̄ = {fbar:.3f}")
        print(equations(extra))
        print({k: round(v, 3) for k, v in fit.items()}, "\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT, "wb"))
    return out


def val_table(models: dict, folder: str) -> pd.DataFrame:
    L = next(iter(models.values())).L
    val = add_controls(pd.read_parquet(DATA_DIR / "processed" / folder / "val.parquet"))
    with np.errstate(invalid="ignore", divide="ignore"):
        Bh = L.batch(val)
        rows = {n: H.horizon_metrics(L.team_view(m.rollout(Bh)), Bh["team"]) for n, m in models.items()}
    t = pd.DataFrame(rows).T
    keep = [
        c
        for c in t
        if any(c.startswith(p) for p in ("health_", "margin_r2", "distance_")) and "step" not in c
    ]
    return t[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    if parser.parse_args().fit:
        fit_all()
        return
    frozen = pickle.load(open(MODELS, "rb"))
    fitted = pickle.load(open(OUT, "rb"))
    rng = np.random.default_rng(0)
    lines = [
        "# Exploratory: the frozen model with more intent terms",
        "",
        "Script: `experiments/planning/intent_variant.py` (set-up in its docstring). Not the frozen "
        "pipeline:",
        "fitted on train, evaluated on val and the counterfactual sets only (all held-out sets are spent).",
        "",
    ]
    for scen, (folder, name) in SETS.items():
        f = fitted[scen]
        models = variants(frozen[name], f["extra"], f["c"], f["fbar"])
        lines += [f"## {scen} ({name} model)", "", f"Focus cost c = {f['c']:.4f}, f̄ = {f['fbar']:.3f}.", ""]
        lines += [
            "```",
            equations(f["extra"]),
            "",
            str({k: round(v, 3) for k, v in f["fit"].items()}),
            "```",
            "",
        ]
        lines += [
            "### Prediction on val",
            "",
            "```",
            val_table(models, folder).round(4).to_string(),
            "```",
            "",
        ]
        states, _ = from_branch(load(folder), frozen[name].L)
        lines += ["### Regret on the margin change at h (counterfactual set)", ""]
        lines += ["```", regret_rows(states, models, rng).round(4).to_string(), "```", ""]
        parts = []
        for n, mm in models.items():
            t = check2(states, mm, rng)
            if not parts:
                base = t[
                    [
                        not i[0].startswith(("alive", "health +", "square", "health margin: model"))
                        for i in t.index
                    ]
                ]
            t = t[[i[0].startswith("health margin: model") for i in t.index]]
            t.index = pd.MultiIndex.from_tuples([(n, i[1]) for i in t.index], names=["model", "h"])
            parts.append(t)
        lines += ["### Picks at h judged by the eventual outcome", ""]
        lines += ["```", base.round(4).to_string(), "", pd.concat(parts).round(4).to_string(), "```", ""]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
