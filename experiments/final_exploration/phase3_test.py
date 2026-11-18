"""Phase 3: score the chosen candidates ONCE on the fresh test2 sets (see 02_phase3_plan.md).

Every model is fitted on all of exp06 `train` (as in the `val` split) and scored on:
    test2           exp06 test2, 1000 battles (seeds 9,000,000+)
    test2_cf        exp06 test2_cf_<intent>, 6 x 600 (9,500,000+); bootstrap groups = seeds
    test2_stalkers  exp07 test2, 300 stalker battles (7,900,000+), scaled models only

Outputs: rows in runs/final_exploration/results.csv (split = test2*), every number in
docs/final_exploration/03_test_results.json, and tables in docs/final_exploration/03_test_results.md.
Paired 95% bootstrap intervals (1000 resamples by battle / seed) for each candidate minus ref/A.

    FINAL_EXPLORATION_PHASE3=1 uv run --extra explore experiments/final_exploration/phase3_test.py
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import DT  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOCS = PROJECT_ROOT / "docs" / "final_exploration"
TEST_SPLITS = ("test2", "test2_cf", "test2_stalkers")
N_BOOT = 1000

# fixed in 02_phase3_plan.md before running: (module, experiment name)
CANDIDATES = [
    ("explore_deep", "deep/A*"),  # interpretable, one-step fits only (10 terms)
    ("ref_baselines", "ref/B"),  # A's support, trajectory-refined (= deep/B*, identical coefficients)
    ("ref_baselines", "ref/S"),  # B + share in reach as a state (= deep/S*)
]
REFERENCES = [
    ("ref_baselines", "ref/A"),
    ("ref_baselines", "ref/persistence"),
    ("ref_baselines", "ref/lanchester-square"),
    ("ref_baselines", "ref/no-control"),
    ("explore_DIJ", "J/mlp-wide"),
]


def per_start(Y: np.ndarray, B: dict, h: int = 4) -> dict:
    """Per start point: health error and the real / predicted margin change at h seconds."""
    n = round(h / DT)
    Yt = B["Y"]
    ok = ~np.isnan(Yt[:, n, 0])
    return {
        "ok": ok,
        "health": np.nan_to_num(np.abs(Y[:, n, :2] - Yt[:, n, :2]).mean(1)),
        "real": np.nan_to_num(Yt[:, n, 0] - Yt[:, n, 1] - (Yt[:, 0, 0] - Yt[:, 0, 1])),
        "pred": Y[:, n, 0] - Y[:, n, 1] - (Y[:, 0, 0] - Y[:, 0, 1]),
    }


def per_battle_nrmse(Y: np.ndarray, B: dict) -> np.ndarray:
    Yt, end = B["Y"], B["end"]
    start = np.maximum(Yt[:, 0, :2], 1e-6)
    mask = np.arange(Yt.shape[1])[None, :] <= end[:, None]
    err = np.where(mask[:, :, None], (Y[:, :, :2] - Yt[:, :, :2]) / start[:, None, :], 0.0)
    return np.sqrt((err**2).sum((1, 2)) / (2 * (end + 1)))


def paired_ci(a: dict, b: dict, Bh: dict, nr_a, nr_b, Bb: dict, seed: int = 0) -> dict:
    """95% intervals of b − a: health 4 s, margin R² 4 s (start points), battle nRMSE (battles)."""
    rng = np.random.default_rng(seed)
    gh, ih = np.unique(Bh["group"], return_inverse=True)
    gb, ib = np.unique(Bb["group"], return_inverse=True)
    ok = a["ok"]
    out = {"health_4s": [], "margin_r2_4s": [], "battle_health_nrmse": []}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(gh), len(gh)), minlength=len(gh))[ih] * ok
        mu = (w * a["real"]).sum() / w.sum()
        den = (w * (a["real"] - mu) ** 2).sum()
        ma = 1 - (w * (a["pred"] - a["real"]) ** 2).sum() / den
        mb = 1 - (w * (b["pred"] - b["real"]) ** 2).sum() / den
        out["health_4s"].append(((w * b["health"]).sum() - (w * a["health"]).sum()) / w.sum())
        out["margin_r2_4s"].append(mb - ma)
        v = np.bincount(rng.integers(0, len(gb), len(gb)), minlength=len(gb))[ib]
        out["battle_health_nrmse"].append(((v * nr_b).sum() - (v * nr_a).sum()) / v.sum())
    return {k: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] for k, x in out.items()}


def main() -> None:
    assert os.environ.get(H.PHASE3_FLAG) == "1", "Phase 3 only: set FINAL_EXPLORATION_PHASE3=1"
    exps = []
    for module, name in REFERENCES + CANDIDATES:
        exps.append({e.name: e for e in importlib.import_module(module).EXPERIMENTS}[name])
    cache: dict = {}
    commit, dirty = H._git()
    report = {"commit": commit, "dirty": dirty, "splits": {}}
    for split in TEST_SPLITS:
        data, Bh, Bb, stalkers = H._split_data(split, 0, cache)
        preds = {}
        rows = []
        for exp in exps:
            model = exp.make()
            if stalkers and not getattr(model, "can_scale", False):
                continue
            model.fit(data)
            Yh, Yb = model.rollout(Bh, stalkers=stalkers), model.rollout(Bb, stalkers=stalkers)
            pw = model.win_probability(Bb, stalkers=stalkers) if hasattr(model, "win_probability") else None
            m = {**H.horizon_metrics(Yh, Bh), **H.battle_metrics(Yb, Bb, pw)}
            row = {
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "commit": commit,
                "dirty": dirty,
                "axis": exp.axis,
                "experiment": exp.name,
                "config_hash": exp.hash,
                "split": split,
                "fold": 0,
                **{f"n_{k}": v for k, v in model.complexity().items()},
                **m,
                "config": json.dumps(exp.config, sort_keys=True, default=str),
                "note": "phase 3 (test2), run once",
            }
            rows.append(row)
            preds[exp.name] = (per_start(Yh, Bh), per_battle_nrmse(Yb, Bb))
            print(
                f"{split:15s} {exp.name:24s} health 4 s {m['health_4s']:.4f}  "
                f"mR² 4 s {m['margin_r2_4s']:+.3f}  "
                f"nRMSE {m['battle_health_nrmse']:.3f}",
                flush=True,
            )
        H._append(H.RESULTS, pd.DataFrame(rows))
        cis = {}
        if "ref/A" in preds:
            a, nra = preds["ref/A"]
            for name, (b, nrb) in preds.items():
                if name != "ref/A":
                    cis[name] = paired_ci(a, b, Bh, nra, nrb, Bb)
        report["splits"][split] = {"metrics": {r["experiment"]: r for r in rows}, "minus_A_95": cis}
    (DOCS / "03_test_results.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [
        "# Phase 3: test2 results (run once)\n",
        f"Commit `{commit}`, dirty = {dirty}. Plan: `02_phase3_plan.md`.\n",
    ]
    cols = [
        "health_4s",
        "margin_r2_2s",
        "margin_r2_4s",
        "margin_r2_8s",
        "alive_4s",
        "battle_health_nrmse",
        "winner_acc",
        "brier",
        "duration_err_s",
        "blowup_rate",
        "n_terms",
    ]
    for split, rep in report["splits"].items():
        lines.append(f"## {split}\n")
        lines.append(
            "| model | "
            + " | ".join(cols)
            + " | Δ health 4 s vs A [95%] | Δ mR² 4 s vs A [95%] | Δ nRMSE vs A [95%] |"
        )
        lines.append("|" + "---|" * (len(cols) + 4))
        for name, r in rep["metrics"].items():
            ci = rep["minus_A_95"].get(name)
            cells = [f"{r[c]:.3f}" if c != "n_terms" else f"{r[c]:.0f}" for c in cols]
            extra = (
                [
                    f"[{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]"
                    for k in ("health_4s", "margin_r2_4s", "battle_health_nrmse")
                ]
                if ci
                else ["", "", ""]
            )
            lines.append(f"| {name} | " + " | ".join(cells + extra) + " |")
        lines.append("")
    (DOCS / "03_test_results.md").write_text("\n".join(lines))
    print("wrote", DOCS / "03_test_results.md")


if __name__ == "__main__":
    main()
