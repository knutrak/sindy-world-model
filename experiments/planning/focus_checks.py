"""Two checks after step 1 (docs/planning/02_model_regret.md), on the same counterfactual sets (S1, S2).

Check 1, where focus's short-term cost sits: from the branch moment, over the first h s, focus vs nearest
(same movement, same seed): Blue units with an enemy in reach (reach_blue_*, what the model's R counts),
Blue units whose own target is in reach (fire_blue_*), Red health lost, Red units lost, and Red health lost
per firing Blue unit.

Check 2, a score at a short horizon h that ranks the branches by their eventual outcome. Scores (team view,
ten units of the first type): health margin, alive margin, health + alive margin, square law A_B² − A_R².
For each score and h, the branch picked by (a) its true value at h and (b) the frozen model's prediction at
h, judged by the branch's eventual outcome: regret in the final health margin and the share of battles won.

    uv run --extra explore experiments/planning/focus_checks.py  -> docs/planning/03_focus_checks.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_regret import BRANCHES, MODELS, SETS, boot, from_branch, load  # noqa: E402

from sindy_wm.models.intent_model import DT  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "03_focus_checks.md"
HORIZONS = (1.0, 2.0, 3.0, 4.0)
PAIRS = {"advance": (0, 1), "hold": (2, 3)}  # (nearest, focus) columns in BRANCHES


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def score(Z: np.ndarray, kind: str) -> np.ndarray:
    """Z: (..., 5) team view [H_B, H_R, A_B, A_R, d]."""
    hb, hr, ab, ar = Z[..., 0], Z[..., 1], Z[..., 2], Z[..., 3]
    return {
        "health margin": hb - hr,
        "alive margin": ab - ar,
        "health + alive": hb - hr + ab - ar,
        "square law": ab**2 - ar**2,
    }[kind]


def check1(states: pd.DataFrame, L, rng) -> pd.DataFrame:
    hp = {t: L.hp[t] / L.hp[L.types[0]] for t in L.types}
    rows = []
    for h in (1.0, 2.0):
        n = round(h / DT)
        per = []
        for _, g in states.groupby(level=0, sort=False):
            w = g.iloc[: n + 1]
            if len(w) < n + 1:
                continue
            reach = w[[f"reach_blue_{t}" for t in L.types]].sum(1).iloc[:n].mean()
            fire = w.filter(regex="^fire_blue_").sum(1).iloc[:n].mean()
            hr = sum(w[f"health_red_{t}"] * hp[t] for t in L.types)
            ar = sum(w[f"alive_red_{t}"] for t in L.types)
            per.append(
                {
                    "seed": g["seed"].iloc[0],
                    "branch": g["branch"].iloc[0],
                    "Blue in reach": reach * 10,
                    "Blue firing at own target": fire * 10,
                    "Red health lost": hr.iloc[0] - hr.iloc[-1],
                    "Red units lost": (ar.iloc[0] - ar.iloc[-1]) * 10,
                }
            )
        f = pd.DataFrame(per)
        f["Red health lost per firing unit·s"] = f["Red health lost"] / (f["Blue firing at own target"] * h)
        f.loc[f["Blue firing at own target"] < 0.5, "Red health lost per firing unit·s"] = np.nan
        for move, (bn, bf) in PAIRS.items():
            a = f[f.branch == bn].set_index("seed")
            b = f[f.branch == bf].set_index("seed")
            common = a.index.intersection(b.index)
            for col in [c for c in f if c not in ("seed", "branch")]:
                x, y = a.loc[common, col], b.loc[common, col]
                ok = x.notna() & y.notna()
                d = (y - x)[ok].to_numpy()
                rows.append(
                    {
                        "h": h,
                        "movement": move,
                        "quantity": col,
                        "nearest": x[ok].mean(),
                        "focus": y[ok].mean(),
                        "focus − nearest": d.mean(),
                        "95%": boot(d, rng),
                    }
                )
    return pd.DataFrame(rows).set_index(["h", "movement", "quantity"])


def check2(states: pd.DataFrame, m, rng) -> pd.DataFrame:
    L = m.L
    with np.errstate(invalid="ignore", divide="ignore"):
        B = L.batch(states, horizon=round(max(HORIZONS) / DT), every=10**9, history=0, min_future=1)
        Y = L.team_view(m.rollout(B))
        Bf = L.batch(states, horizon=720, every=10**9, history=0, min_future=1)
    TY, TF = B["team"]["Y"], Bf["team"]["Y"]
    assert (Bf["ep"] == B["ep"]).all()
    g = states.groupby(level=0, sort=False)
    seed = g["seed"].first().loc[B["ep"]].to_numpy()
    branch = g["branch"].first().loc[B["ep"]].to_numpy()
    won = g["battle_won"].first().loc[B["ep"]].to_numpy().astype(float)
    end = (~np.isnan(TF[:, :, 0])).sum(1) - 1
    final = TF[np.arange(len(TF)), end, 0] - TF[np.arange(len(TF)), end, 1]

    def wide(v):
        return (
            pd.DataFrame({"s": seed, "b": branch, "v": v})
            .pivot(index="s", columns="b", values="v")
            .to_numpy()
        )

    Fm, W = wide(final), wide(won)
    idx = np.arange(len(Fm))
    best = Fm.max(1)
    out = []

    def judge(name, h, pick):
        r = best - Fm[idx, pick]
        out.append(
            {
                "score": name,
                "h": h,
                "regret (final margin)": r.mean(),
                "95%": boot(r, rng),
                "won": W[idx, pick].mean(),
            }
        )

    judge("always advance/nearest", np.nan, np.zeros(len(Fm), int))
    fixed = int(Fm.mean(0).argmax())
    judge(f"best fixed in hindsight ({BRANCHES[fixed]})", np.nan, np.full(len(Fm), fixed))
    judge("true best by final margin", np.nan, Fm.argmax(1))
    for h in HORIZONS:
        last = (~np.isnan(TY[:, :, 0])).sum(1) - 1
        i = np.minimum(round(h / DT), last)
        r = np.arange(len(TY))
        for kind in ("health margin", "alive margin", "health + alive", "square law"):
            T = wide(score(TY[r, i], kind))
            M = wide(score(Y[r, i], kind))
            judge(f"{kind}: true at h", h, T.argmax(1))
            judge(f"{kind}: model at h", h, M.argmax(1))
    return pd.DataFrame(out).set_index(["score", "h"])


def main() -> None:
    models = pickle.load(open(MODELS, "rb"))
    rng = np.random.default_rng(0)
    lines = [
        "# Two checks on the focus term and the score at the horizon",
        "",
        "Script: `experiments/planning/focus_checks.py` (set-up in its docstring). Same "
        "counterfactual states as",
        "`02_model_regret.md`. Health in ten units of the first type; unit counts in units. 95%: "
        "bootstrap over states.",
        "",
    ]
    for scen, (folder, name) in SETS.items():
        m = models[name]
        states, _ = from_branch(load(folder), m.L)
        lines += [f"## {scen} ({name} model)", "", "### Check 1: focus vs nearest over the first h s", ""]
        lines += [md(check1(states, m.L, rng).round(4)), ""]
        lines += ["### Check 2: scores at h, judged by the eventual outcome", ""]
        lines += [md(check2(states, m, rng).round(4)), ""]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
