"""Headroom follow-up, step 1: how well does the frozen pipeline model rank the five intents at short
horizons?

Data: the counterfactual sets (exp06_intents cf_* for S1 marines, exp08_mixed cf_* for S2 mixed), per-type
states from `scripts/build_typed_states.py`. Each seed is one battle replayed five times: identical up to a
drawn branch moment, then one intent for 4 s, then advance/nearest to the end. (withdraw/focus is left out:
withdraw ignores the target rule, so it repeats withdraw/nearest, as in the planner's five candidates.)

Score of a branch at horizon h: the change in the team health margin (Blue - Red, ten units of the first type)
from the branch moment to min(h, battle end). The model is rolled out from the branch state under each
branch's real controls. Per state: regret = true best score - true score of the ranker's pick. Rankers: the
frozen model (marines model on S1, mixed model on S2), always advance/nearest, the best fixed intent in
hindsight (chosen on these same states, so optimistic), and a random pick (its expected regret: best - mean
of the five).

    uv run --extra explore experiments/planning/model_regret.py  -> docs/planning/02_model_regret.md
"""

import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sindy_wm.models.intent_model import DT, add_controls
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

DOC = PROJECT_ROOT / "docs" / "planning" / "02_model_regret.md"
MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
BRANCHES = ("advance_nearest", "advance_focus", "hold_nearest", "hold_focus", "withdraw_nearest")
SETS = {"S1": ("exp06_intents_typed", "marines"), "S2": ("exp08_mixed_typed", "mixed")}
HORIZONS = (1.0, 2.0, 4.0, 8.0)
COSTLY = 0.05
N_BOOT = 2000


def load(folder: str) -> pd.DataFrame:
    parts = []
    for b, name in enumerate(BRANCHES):
        d = pd.read_parquet(DATA_DIR / "processed" / folder / f"cf_{name}.parquet").reset_index()
        d["branch"] = b
        d["episode_idx"] = d["episode_idx"] + 10_000 * (b + 1)
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_controls(pd.concat(parts))


def from_branch(states: pd.DataFrame, L) -> tuple[pd.DataFrame, float]:
    """Each branch from its branch moment on; keeps seeds with all five branches and both sides alive.
    Also returns the largest state difference across a seed's branches at the branch moment (should be 0)."""
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


def margin_change(Y: np.ndarray, truth: np.ndarray, h: float) -> np.ndarray:
    last = (~np.isnan(truth[:, :, 0])).sum(1) - 1
    i = np.minimum(round(h / DT), last)
    rows = np.arange(len(Y))
    return (Y[rows, i, 0] - Y[rows, i, 1]) - (Y[:, 0, 0] - Y[:, 0, 1])


def boot(x: np.ndarray, rng) -> str:
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return f"[{np.quantile(m, 0.025):+.4f}, {np.quantile(m, 0.975):+.4f}]"


def rank_table(T: np.ndarray, M: np.ndarray, rng) -> tuple[pd.DataFrame, dict]:
    """T, M: (states, 5) true and model scores. Returns the rankers' table and the paired comparison."""
    rows = np.arange(len(T))
    best = T.max(1)
    fixed = int(T.mean(0).argmax())
    picks = {
        "model": M.argmax(1),
        "always advance/nearest": np.zeros(len(T), int),
        f"best fixed in hindsight ({BRANCHES[fixed]})": np.full(len(T), fixed),
    }
    table, regret = {}, {}
    for name, p in picks.items():
        r = best - T[rows, p]
        regret[name] = r
        table[name] = {
            "regret": r.mean(),
            "regret 95%": boot(r, rng),
            f"costly (> {COSTLY})": (r > COSTLY).mean(),
            "picks the true best": (r <= 1e-9).mean(),
        }
    with np.errstate(invalid="ignore"):
        rho = [spearmanr(M[i], T[i]).statistic for i in rows]
    table["model"]["rank correlation"] = np.nanmean(rho)
    r = best - T.mean(1)
    table["random (expected)"] = {"regret": r.mean(), "regret 95%": boot(r, rng), "rank correlation": 0.0}
    d = regret["model"] - regret["always advance/nearest"]
    info = {
        "states": len(T),
        "spread": float((T.max(1) - T.min(1)).mean()),
        "ties": float((T.max(1) - T.min(1) < 1e-9).mean()),
        "model − always advance/nearest": f"{d.mean():+.4f} {boot(d, rng)}",
        "model's picks": dict(zip(BRANCHES, np.bincount(M.argmax(1), minlength=5) / len(T), strict=True)),
    }
    return pd.DataFrame(table).T, info


def main() -> None:
    models = pickle.load(open(MODELS, "rb"))
    rng = np.random.default_rng(0)
    lines = [
        "# Headroom follow-up, step 1: the frozen model's regret on the counterfactual sets",
        "",
        "Script: `experiments/planning/model_regret.py` (see its docstring for the set-up). Score: "
        "change in the",
        "team health margin (Blue − Red, in ten units of the first type) from the branch moment to h "
        "s (or the",
        "battle's end). Regret = true best − true score of the pick, per state; 95% bootstrap over "
        "states. The",
        "true branches play the intent for 4 s, then advance/nearest, so up to 4 s the truth is the "
        "intent alone.",
        "",
    ]
    diag = []
    for scen, (folder, model_name) in SETS.items():
        m = models[model_name]
        L = m.L
        states, gap = from_branch(load(folder), L)
        with np.errstate(invalid="ignore", divide="ignore"):
            B = L.batch(states, horizon=round(max(HORIZONS) / DT), every=10**9, history=0, min_future=1)
            Y = L.team_view(m.rollout(B))
        TY = B["team"]["Y"]
        seed = states.groupby(level=0, sort=False)["seed"].first().loc[B["ep"]].to_numpy()
        branch = states.groupby(level=0, sort=False)["branch"].first().loc[B["ep"]].to_numpy()
        lines += [f"## {scen} ({model_name} model, {folder}/cf_*)", ""]
        lines += [f"Largest state difference across a seed's branches at the branch moment: {gap:.2e}.", ""]
        for h in HORIZONS:
            frame = pd.DataFrame(
                {
                    "seed": seed,
                    "branch": branch,
                    "truth": margin_change(TY, TY, h),
                    "model": margin_change(Y, TY, h),
                }
            )
            T = frame.pivot(index="seed", columns="branch", values="truth").to_numpy()
            M = frame.pivot(index="seed", columns="branch", values="model").to_numpy()
            table, info = rank_table(T, M, rng)
            diag.append(diagnostics(scen, h, T, M))
            picks = ", ".join(f"{k} {v:.2f}" for k, v in info["model's picks"].items())
            lines += [
                f"### {scen} at {h:g} s",
                "",
                f"{info['states']} states; mean spread (best − worst branch) {info['spread']:.4f}; "
                f"all five tied in {info['ties']:.0%}.",
                f"Model − always advance/nearest, regret: {info['model − always advance/nearest']}.",
                f"Model's picks: {picks}.",
                "",
                "```",
                table.round(4).to_string(),
                "```",
                "",
            ]
    lines += [
        "## Target rule and movement, separately",
        "",
        "Mean score differences between branches, true and as the model predicts them; and the "
        "model's regret",
        "when it may only pick among the nearest variants and withdraw (regret still against the "
        "best of all five).",
        "",
        "```",
        pd.DataFrame(diag).set_index(["scenario", "h"]).round(4).to_string(),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


def diagnostics(scen: str, h: float, T: np.ndarray, M: np.ndarray) -> dict:
    """Columns follow BRANCHES: 0 advance/nearest, 1 advance/focus, 2 hold/nearest, 3 hold/focus, 4 "
    "withdraw."""
    near = np.array([0, 2, 4])
    pick = near[M[:, near].argmax(1)]
    return {
        "scenario": scen,
        "h": h,
        "focus − nearest (advance), true": (T[:, 1] - T[:, 0]).mean(),
        "model": (M[:, 1] - M[:, 0]).mean(),
        "hold − advance (nearest), true": (T[:, 2] - T[:, 0]).mean(),
        "model ": (M[:, 2] - M[:, 0]).mean(),
        "model regret, nearest-only picks": (T.max(1) - T[np.arange(len(T)), pick]).mean(),
        "always advance/nearest": (T.max(1) - T[:, 0]).mean(),
    }


if __name__ == "__main__":
    main()
