"""Where do the frozen model's errors line up with the intents? (train sets only, S1 marines and S2 mixed)

From states every 0.5 s of each train battle where the intent stays the same for the next 1 s, the frozen
model is rolled out 1 s from the measured state (reach starts at the measured value, as in every rollout).
Residual = measured change − predicted change over that 1 s, for team health and alive per side and the
distance. Two parts the rollout does not predict are checked the same way: the reach closure (measured units
in reach − the closure's value, at the start state) and the formation shape (depth and width per side: the
change over 1 s, predicted 0).

Each residual is compared across the ten cells movement × target (withdraw: one cell) × phase (far / contact,
split at the centroid distance 7.6, the model's FAR_GATE). "Intent R²" = share of the residual's variance
explained by the cell means; "max |cell| / sd" = the largest cell mean over the residual's sd. Cell means
are in the quantity's units (health: ten units of the first type; alive and reach: units / 10; distance,
shape: map units).

    uv run --extra explore experiments/planning/intent_residuals.py  -> docs/planning/05_intent_residuals.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_regret import MODELS  # noqa: E402

from sindy_wm.models.intent_model import DT, FAR_GATE, SHAPE, add_controls  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "05_intent_residuals.md"
SETS = {"S1": ("exp06_intents_typed", "marines"), "S2": ("exp08_mixed_typed", "mixed")}
H = round(1.0 / DT)  # 1 s
EVERY = 4  # 0.5 s


def cell(U: np.ndarray, d: np.ndarray) -> np.ndarray:
    move = np.where(U[:, 2] > 0, "withdraw", np.where(U[:, 1] > 0, "hold", "advance"))
    tgt = np.where(U[:, 2] > 0, "", np.where(U[:, 3] > 0, "/focus", "/nearest"))
    phase = np.where(d < FAR_GATE[0], "contact", "far")
    return np.char.add(np.char.add(move.astype(str), tgt.astype(str)), np.char.add(" ", phase.astype(str)))


def summarise(res: dict, cells: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, means = [], {}
    for name, r in res.items():
        ok = np.isfinite(r)
        s = pd.Series(r[ok]).groupby(cells[ok])
        mu, n = s.mean(), s.size()
        total = r[ok].var()
        between = float((n * (mu - r[ok].mean()) ** 2).sum() / ok.sum())
        rows.append(
            {
                "quantity": name,
                "n": int(ok.sum()),
                "residual sd": r[ok].std(),
                "intent R²": between / total if total > 0 else np.nan,
                "max |cell| / sd": float(mu.abs().max() / r[ok].std()),
            }
        )
        means[name] = mu
    return pd.DataFrame(rows).set_index("quantity"), pd.DataFrame(means).T


def run(folder: str, m) -> tuple[pd.DataFrame, pd.DataFrame]:
    L = m.L
    states = add_controls(pd.read_parquet(DATA_DIR / "processed" / folder / "train.parquet"))
    with np.errstate(invalid="ignore", divide="ignore"):
        B = L.batch(states, horizon=H, every=EVERY, history=0, min_future=H + 1)
        steady = (B["U"] == B["U"][:, :1]).all((1, 2))
        B = {
            k: (v[steady] if isinstance(v, np.ndarray) and len(v) == len(steady) else v) for k, v in B.items()
        }
        Y = m.rollout(B)
    T, P = L.team_view(B["Y"]), L.team_view(Y)
    names = ["health blue", "health red", "alive blue", "alive red", "distance"]
    res = {n: (T[:, H, j] - T[:, 0, j]) - (P[:, H, j] - P[:, 0, j]) for j, n in enumerate(names)}
    # reach closure at the start state
    sh = m._shares(B["X0"].astype(float), B["shape0"], B["front0"])
    for s in ("blue", "red"):
        closure = sum(B["X0"][:, L.idx("alive", s, a)] * sh[(s, a)] for a in L.types)
        measured = sum(B["reach0"][:, L.pos(s, a)] for a in L.types)
        res[f"reach {s} (closure)"] = measured - closure
    # formation shape: measured change over 1 s (the rollout holds it constant)
    shape = {ep: g[SHAPE].to_numpy(float) for ep, g in states.groupby(level=0, sort=False)}
    S1 = np.array([shape[e][k + H] for e, k in zip(B["ep"], B["k0"], strict=True)])
    for j, c in enumerate(SHAPE):
        res[f"{c.replace('_', ' ')} (held)"] = S1[:, j] - B["shape0"][:, j]
    return summarise(res, cell(B["U"][:, 0], B["X0"][:, -1]))


def main() -> None:
    models = pickle.load(open(MODELS, "rb"))
    lines = [
        "# Residuals of the frozen model against the intents (train sets)",
        "",
        "Script: `experiments/planning/intent_residuals.py` (set-up in its docstring).",
        "",
    ]
    for scen, (folder, name) in SETS.items():
        summary, means = run(folder, models[name])
        lines += [
            f"## {scen} ({name} model, {folder}/train)",
            "",
            "```",
            summary.round(4).to_string(),
            "```",
            "",
        ]
        lines += ["Cell means:", "", "```", means.round(4).to_string(), "```", ""]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
