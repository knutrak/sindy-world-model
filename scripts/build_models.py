"""Fit the exp06 world models and build the tables the notebooks read, from the state tables.

Until now these files were made by hand inside notebooks or outside them; this script makes them
reproducible after a re-collection. Outputs (data/processed/exp06_intents/ by default):

    closures.pkl          reach curve (normal CDF) and movement switches, `fit_closures`     (sindyc_02)
    model_rich.pkl        all candidates, health threshold 0.05, closure reach               (sindyc_02)
    model_sparse.pkl      no alive x control terms, threshold 0.2, measured reach            (sindyc_02)
                          (both without the later hold-split candidates, as originally fitted)
    model_split.pkl       as sparse, hold's movement split by contact (no `close*hold`)      (sindyc_05)
    model_derived.pkl     as split, the derived reach curve (`fit_derived_share`),
                          threshold 0.15: the current core model                             (sindyc_05)
    formation_train2000.parquet, formation_val.parquet
                          `formation_numbers` per step, for train shards 0-1 and val         (sindyc_05)

Not built here (no recipe in the repo; negative results, read by no notebook): model_healthfit,
model_rules, model_rules_fixed.

Order of the whole exp06 pipeline:

    uv run scripts/collect_intents.py ...         raw battles (data/raw/exp06_intents)
    uv run scripts/build_intent_states.py         team-level state tables (train/val/cf/...)
    uv run scripts/build_models.py                this script
    uv run scripts/build_firing_counts.py         units firing (sindyc_06)
    uv run scripts/play_hold_advance_pairs.py     hold vs advance pairs (sindyc_05)
    uv run scripts/eval_planners.py ...           planner results (sindyc_03, sindyc_05)

Usage:

    uv run scripts/build_models.py                  build what is missing
    uv run scripts/build_models.py --force          rebuild everything
    uv run scripts/build_models.py --out DIR        write somewhere else (e.g. to compare with the old files)
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from sindy_wm.io import storage
from sindy_wm.models.intent_model import (
    ALIVE_CONTROL_TERMS,
    IntentModel,
    add_controls,
    add_rates,
    fit_closures,
    fit_derived_share,
)
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.geometry import formation_numbers

PROCESSED = DATA_DIR / "processed" / "exp06_intents"
RAW = RAW_DIR / "exp06_intents"
SPLIT_EXCLUDE = ALIVE_CONTROL_TERMS + ("close*hold",)
# rich and sparse predate the hold split: keep them as fitted then, without its two candidates
NO_SPLIT = ("close*hold*far", "close*hold*contact")


def training_rows() -> pd.DataFrame:
    """Training steps with both sides alive, with controls and smoothed rates (window 9)."""
    train = add_rates(add_controls(pd.read_parquet(PROCESSED / "train.parquet")))
    return train[(train["alive_blue"] > 0) & (train["alive_red"] > 0)]


def derived_closures(rows: pd.DataFrame, closures: dict) -> dict:
    """The closures with the reach curve derived from the geometry (lam, beta, g fitted; rho the game's)."""
    return {
        "gates": closures["gates"],
        "share_blue": fit_derived_share(rows, "blue")[0],
        "share_red": fit_derived_share(rows, "red")[0],
        "form": "derived",
    }


def fit_models(rows: pd.DataFrame, closures: dict) -> dict:
    derived = derived_closures(rows, closures)
    return {
        "rich": IntentModel(closures, exclude=NO_SPLIT).fit(rows, thr_health=0.05),
        "sparse": IntentModel(closures, exclude=ALIVE_CONTROL_TERMS + NO_SPLIT).fit(
            rows, thr_health=0.2, reach_true=True
        ),
        "split": IntentModel(closures, exclude=SPLIT_EXCLUDE).fit(rows, thr_health=0.2, reach_true=True),
        "derived": IntentModel(derived, exclude=SPLIT_EXCLUDE).fit(rows, thr_health=0.15, reach_true=True),
    }


def formation_table(shards: list[str], offsets: list[int]) -> pd.DataFrame:
    """`formation_numbers` for whole raw shards, with episode ids as in the state tables."""
    parts = []
    for name, offset in zip(shards, offsets, strict=True):
        f = formation_numbers(storage.load_snapshots(RAW / name)).reset_index()
        f["episode_idx"] += offset
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROCESSED)
    parser.add_argument("--force", action="store_true", help="rebuild files that already exist")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def todo(name: str) -> bool:
        return args.force or not (out / name).exists()

    model_files = [f"model_{n}.pkl" for n in ("rich", "sparse", "split", "derived")]
    if todo("closures.pkl") or any(todo(f) for f in model_files):
        rows = training_rows()
        if todo("closures.pkl"):
            closures = fit_closures(rows)
            pickle.dump(closures, open(out / "closures.pkl", "wb"))
            print("built closures.pkl", flush=True)
        else:
            closures = pickle.load(open(out / "closures.pkl", "rb"))
        for name, model in fit_models(rows, closures).items():
            if todo(f"model_{name}.pkl"):
                pickle.dump(model, open(out / f"model_{name}.pkl", "wb"))
                print(f"built model_{name}.pkl", flush=True)

    for name, shards, offsets in (
        ("formation_train2000.parquet", ["train_0", "train_1"], [0, 1000]),
        ("formation_val.parquet", ["val"], [0]),
    ):
        if todo(name):
            formation_table(shards, offsets).to_parquet(out / name)
            print(f"built {name}", flush=True)


if __name__ == "__main__":
    main()
