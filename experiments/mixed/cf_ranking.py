"""Step 4: does the mixed model rank the six intents like the simulator? (exp08 cf_*, exploratory)

In cf_<intent> the same battle is replayed six times: identical up to a drawn branch moment, then one
intent for 4 s, then advance/nearest. Every model is rolled out from the branch moment under each
branch's real controls; the outcome is the change in the team health margin (ten marines' hp) at 4 s
and 8 s, or at the battle's end if earlier. Metrics per seed across the six branches: Spearman rank
correlation, picks the true best, regret; against the best fixed intent (chosen in hindsight) and random.

    uv run --extra explore experiments/mixed/cf_ranking.py      -> docs/mixed/05_cf_ranking.md
"""

import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sindy_wm.evaluation import harness as H
from sindy_wm.models.intent_model import DT, add_controls
from sindy_wm.models.typed_model import team_horizon_batch, team_view, typed_batch
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

P = DATA_DIR / "processed" / "exp08_mixed"
DOC = PROJECT_ROOT / "docs" / "mixed" / "05_cf_ranking.md"


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def load_cf() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(P.glob("cf_*.parquet"))
    parts = []
    for k, f in enumerate(files):
        d = pd.read_parquet(f).reset_index()
        d["episode_idx"] += 10_000 * (k + 1)
        parts.append(d.set_index(["episode_idx", "step"]))
    return add_controls(pd.concat(parts)), [f.stem.removeprefix("cf_") for f in files]


def branch_batch(states: pd.DataFrame, horizon: int = 64) -> dict:
    parts = []
    for _, g in states.groupby(level=0, sort=False):
        k = int(round(g["branch_at"].iloc[0] / DT))
        if k < len(g) - 8 and g[[c for c in g if c.startswith("alive_blue_")]].iloc[k].sum() > 0:
            if g[[c for c in g if c.startswith("alive_red_")]].iloc[k].sum() > 0:
                parts.append(g.iloc[k:])
    with np.errstate(invalid="ignore", divide="ignore"):
        B = typed_batch(pd.concat(parts), horizon=horizon, every=10**9, history=0, min_future=1)
    B["seed"], B["branch"] = B["ep"] % 10_000, B["ep"] // 10_000 - 1
    return B


def margin_change(Y_team: np.ndarray, truth_team: np.ndarray, h: float) -> np.ndarray:
    n = round(h / DT)
    last = (~np.isnan(truth_team[:, :, 0])).sum(1) - 1
    i = np.minimum(n, last)
    rows = np.arange(len(Y_team))
    return (Y_team[rows, i, 0] - Y_team[rows, i, 1]) - (Y_team[:, 0, 0] - Y_team[:, 0, 1])


def ranking(B: dict, preds: dict, labels: list[str], h: float) -> tuple[dict, pd.DataFrame]:
    frame = pd.DataFrame(
        {"seed": B["seed"], "branch": B["branch"], "truth": margin_change(B["TY"], B["TY"], h)}
    )
    for n, Y in preds.items():
        frame[n] = margin_change(Y, B["TY"], h)
    wide = {c: frame.pivot(index="seed", columns="branch", values=c) for c in ["truth", *preds]}
    full = wide["truth"].dropna().index
    T = wide["truth"].loc[full].to_numpy()
    rows = np.arange(len(T))
    table = {}
    for n in preds:
        M = wide[n].loc[full].to_numpy()
        pick = M.argmax(1)
        table[n] = {
            "rank correlation": np.nanmean([spearmanr(M[i], T[i]).statistic for i in rows]),
            "picks the true best": float((T[rows, pick] >= T.max(1) - 1e-9).mean()),
            "regret": float((T.max(1) - T[rows, pick]).mean()),
        }
    best = int(T.mean(0).argmax())
    table[f"always {labels[best]}"] = {
        "rank correlation": np.nan,
        "picks the true best": float((T[:, best] >= T.max(1) - 1e-9).mean()),
        "regret": float((T.max(1) - T[:, best]).mean()),
    }
    rnd = np.random.default_rng(0).integers(0, T.shape[1], len(T))
    table["random"] = {
        "rank correlation": 0.0,
        "picks the true best": float((T[rows, rnd] >= T.max(1) - 1e-9).mean()),
        "regret": float((T.max(1) - T[rows, rnd]).mean()),
    }
    info = {"seeds": len(full), "spread": float(T.std(1).mean()), "best fixed": labels[best]}
    return info, pd.DataFrame(table).T


def predictions(B: dict, models: dict) -> dict:
    out = {}
    for name, m in models.items():
        out[name] = team_view(m.rollout(B)) if name.startswith("typed") else m.rollout(team_horizon_batch(B))
    return out


def main() -> None:
    states, labels = load_cf()
    B = branch_batch(states)
    models = pickle.load(open(H.CACHE / "mixed_models.pkl", "rb"))
    preds = predictions(B, models)
    lines = ["# Step 4: counterfactual intent ranking on mixed battles (exp08 cf_*, 19 Nov)", ""]
    lines += [f"Script: `experiments/mixed/cf_ranking.py`. Branches: {labels}.", ""]
    for h in (4, 8):
        info, table = ranking(B, preds, labels, h)
        lines += [
            f"## At {h} s ({info['seeds']} seeds; mean spread of the true margin change across branches "
            f"{info['spread']:.3f}; best fixed intent in hindsight: {info['best fixed']})",
            "",
            md(table.round(3)),
            "",
        ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
