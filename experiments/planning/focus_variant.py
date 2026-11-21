"""Exploratory: the frozen model plus a focus cost on Blue's firing, on the counterfactual sets (S1, S2).

Check 1 (docs/planning/03_focus_checks.md) found that under focus, as many Blue units have an enemy in reach
but fewer have their own target in reach. Variant, no refit: in Red's health loss, Blue's reach R becomes

    R · (1 − c·focus) / (1 − c·f̄)

c = 1 − (firing / in reach | focus) / (firing / in reach | nearest), measured on each scenario's TRAIN set
(steps with Blue in reach, not withdrawing); f̄ = train's focus share of those steps (0.5), so the
train-mean rate is kept ("compensated"). "scale only" leaves out the division. The frozen models themselves
are not changed.

    uv run --extra explore experiments/planning/focus_variant.py  -> docs/planning/04_focus_variant.md
"""

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focus_checks import check2  # noqa: E402
from model_regret import MODELS, SETS, from_branch, load, margin_change, rank_table  # noqa: E402

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "04_focus_variant.md"
TRAIN = {"S1": "exp06_intents_typed", "S2": "exp08_mixed_typed"}


def focus_cost(folder: str, types) -> tuple[float, float]:
    d = pd.read_parquet(DATA_DIR / "processed" / folder / "train.parquet")
    reach = d[[f"reach_blue_{t}" for t in types]].sum(1)
    fire = d.filter(regex="^fire_blue_").sum(1)
    ok = (reach > 0) & (d["movement"] != "withdraw")
    g = pd.DataFrame({"r": reach[ok], "f": fire[ok], "target": d.loc[ok, "target"]}).groupby("target").sum()
    ratio = g["f"] / g["r"]
    return float(1 - ratio["focus"] / ratio["nearest"]), float((d.loc[ok, "target"] == "focus").mean())


def variants(m, c: float, fbar: float) -> dict:
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    a.focus_cost, a.focus_norm = c, 1.0
    b.focus_cost, b.focus_norm = c, 1 - c * fbar
    return {"frozen": m, "scale only": a, "compensated": b}


def regret_rows(states, models, rng) -> pd.DataFrame:
    L = next(iter(models.values())).L
    rows = []
    with np.errstate(invalid="ignore", divide="ignore"):
        B = L.batch(states, horizon=64, every=10**9, history=0, min_future=1)
        Ys = {n: L.team_view(m.rollout(B)) for n, m in models.items()}
    TY = B["team"]["Y"]
    g = states.groupby(level=0, sort=False)
    seed = g["seed"].first().loc[B["ep"]].to_numpy()
    branch = g["branch"].first().loc[B["ep"]].to_numpy()

    def wide(v):
        return (
            pd.DataFrame({"s": seed, "b": branch, "v": v})
            .pivot(index="s", columns="b", values="v")
            .to_numpy()
        )

    for h in (1.0, 2.0, 4.0, 8.0):
        T = wide(margin_change(TY, TY, h))
        for n, Y in Ys.items():
            M = wide(margin_change(Y, TY, h))
            table, info = rank_table(T, M, rng)
            rows.append(
                {
                    "h": h,
                    "model": n,
                    "regret": table.loc["model", "regret"],
                    "95%": table.loc["model", "regret 95%"],
                    "always adv/nearest": table.loc["always advance/nearest", "regret"],
                    "model − always": info["model − always advance/nearest"],
                    "rank corr": table.loc["model", "rank correlation"],
                    "picks focus": info["model's picks"]["advance_focus"]
                    + info["model's picks"]["hold_focus"],
                    "picks withdraw": info["model's picks"]["withdraw_nearest"],
                }
            )
    return pd.DataFrame(rows).set_index(["h", "model"])


def main() -> None:
    frozen = pickle.load(open(MODELS, "rb"))
    rng = np.random.default_rng(0)
    lines = [
        "# Exploratory: the frozen model plus a focus cost on Blue's firing",
        "",
        "Script: `experiments/planning/focus_variant.py` (set-up in its docstring). Not the frozen "
        "pipeline: an",
        "exploratory variant, evaluated on the counterfactual sets only (all held-out sets are spent).",
        "",
    ]
    for scen, (folder, name) in SETS.items():
        m = frozen[name]
        c, fbar = focus_cost(TRAIN[scen], m.L.types)
        models = variants(m, c, fbar)
        states, _ = from_branch(load(folder), m.L)
        lines += [f"## {scen} ({name} model): c = {c:.4f}, f̄ = {fbar:.3f} (train)", ""]
        lines += ["### Regret on the margin change at h (as 02_model_regret.md)", ""]
        lines += ["```", regret_rows(states, models, rng).round(4).to_string(), "```", ""]
        lines += [
            "### Picks at h judged by the eventual outcome (as 03_focus_checks.md check 2, health margin)",
            "",
        ]
        parts = []
        for n, mm in models.items():
            t = check2(states, mm, rng)
            t = t[[i[0].startswith("health margin: model") for i in t.index]]
            t.index = pd.MultiIndex.from_tuples([(n, i[1]) for i in t.index], names=["model", "h"])
            parts.append(t)
        base = check2(states, m, rng)
        base = base[
            [not i[0].startswith(("alive", "health +", "square", "health margin: model")) for i in base.index]
        ]
        lines += ["```", base.round(4).to_string(), "", pd.concat(parts).round(4).to_string(), "```", ""]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
