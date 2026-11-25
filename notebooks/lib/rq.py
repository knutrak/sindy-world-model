"""Shared helpers of the research-question notebooks (notebooks/06_research_questions)."""

import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import DT, add_controls
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

for p in ("scripts", "experiments/mixed", "experiments/generalisation", "experiments/triple"):
    if str(PROJECT_ROOT / p) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / p))

from sindy_wm.evaluation import harness as H  # noqa: E402

PROC = DATA_DIR / "processed"
DOCS = PROJECT_ROOT / "docs"
# fixed order of hues; grey and black for baselines
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK = "#8a8984", "#0b0b0b"
EXP = {
    "marines": "exp06_intents",
    "marauders": "exp09_marauders",
    "stalkers": "exp07_stalkers",
    "mixed": "exp08_mixed",
    "triple": "exp12_triple",
}

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e5e0",
        "axes.axisbelow": True,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
    }
)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 90)


def held_out() -> dict:
    """The one held-out run (docs/generalisation/04_test_results.json). Read, never recomputed."""
    return json.loads((DOCS / "generalisation" / "04_test_results.json").read_text())


def models() -> dict:
    """The fitted pipeline models: three frozen ones, stalkers and triple (25 Nov)."""
    m = dict(pickle.load(open(PROC / "final_exploration" / "generalisation_frozen.pkl", "rb")))
    m["stalkers"] = pickle.load(open(PROC / "exp07_stalkers_typed" / "stalker_model.pkl", "rb"))
    m["triple"] = pickle.load(open(PROC / "exp12_triple_typed" / "triple_model.pkl", "rb"))
    return {k: m[k] for k in EXP}


def states(name: str, split: str = "val") -> pd.DataFrame:
    """Per-type states with controls. Train and val only."""
    if split not in ("train", "val"):
        raise ValueError("held-out sets are not read in notebooks")
    return add_controls(pd.read_parquet(PROC / f"{EXP[name]}_typed" / f"{split}.parquet"))


def rates(name: str, split: str = "train") -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{EXP[name]}_typed" / f"rates_{split}_w9.parquet")


def rollout(m, B) -> np.ndarray:
    with np.errstate(all="ignore"):
        return m.L.team_view(m.rollout(B))


def paired_ci(diff_by_unit: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of paired differences with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    x = np.asarray(diff_by_unit, float)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def horizon_table(rolls: dict, B: dict, scale: float = 1.0) -> pd.DataFrame:
    """Harness metrics per model; health errors multiplied by `scale`."""
    t = pd.DataFrame({k: H.horizon_metrics(Y, B) for k, Y in rolls.items()}).T
    cols = [c for c in t if c.startswith("health_")]
    t[cols] *= scale
    return t


def last_slope(states_: pd.DataFrame, B: dict, L, history: int = 8) -> np.ndarray:
    """Linear extrapolation: the team state's rate over the last `history` steps continues."""
    by = {ep: L.team_view(g[L.x_cols].to_numpy(float)) for ep, g in states_.groupby(level=0, sort=False)}
    X0 = np.stack([by[e][k] for e, k in zip(B["ep"], B["k0"], strict=True)])
    Xp = np.stack([by[e][k - history] for e, k in zip(B["ep"], B["k0"], strict=True)])
    slope = (X0 - Xp) / (history * DT)
    t = np.arange(B["U"].shape[1] + 1) * DT
    Y = X0[:, None] + slope[:, None] * t[None, :, None]
    return np.maximum(Y, 0)


def used_closure_numbers(m, n: int = 200) -> dict:
    """Numbers fitted outside SINDy that change a rollout when perturbed (random states, all intents)."""
    import copy

    from sindy_wm.world_model import INTENTS, plan_controls

    rng, k = np.random.default_rng(0), 2 * m.L.n
    h = rng.uniform(0.1, 1, (n, k)) / m.L.n
    a = np.minimum(h * rng.uniform(1, 1.5, h.shape), 1.2)
    B = {
        "X0": np.column_stack([h, a, rng.uniform(3, 14, n)]),
        "shape0": rng.uniform(0.5, 2.5, (n, 4)),
        "front0": rng.uniform(-0.5, 0.5, (n, k)),
        "reach0": a * rng.uniform(0, 1, a.shape),
        "U": np.stack([plan_controls(INTENTS[i % 5], 4.0) for i in range(n)]),
    }
    base = m.rollout(B)
    groups = {
        "reach closure": lambda x: x.share,
        "movement gates": lambda x: x.P_team["gates"],
        "withdraw closure": lambda x: {s: x.P_team[s] for s in ("share_blue", "share_red")},
        "target choice": lambda x: x.alloc_b,
    }
    out = {}
    for lab, get in groups.items():
        used = 0
        for key, arr in get(m).items():
            for i in range(len(arr)):
                c = copy.deepcopy(m)
                v = np.array(get(c)[key], float)
                v[i] = v[i] * 1.01 + 0.01
                get(c)[key][...] = v
                used += not np.array_equal(c.rollout(B), base)
        out[lab] = used
    return out
