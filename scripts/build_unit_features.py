"""Build the per-unit feature tables (`states.unit_features`) for the final exploration.

Writes data/processed/final_exploration/features_<experiment>_<set>.parquet, indexed like the
state tables (exp06 train episode ids = shard * 1000 + episode_idx).

    uv run scripts/build_unit_features.py                 train, val, exp07 train
    uv run scripts/build_unit_features.py --sets val
    FINAL_EXPLORATION_PHASE3=1 uv run scripts/build_unit_features.py --sets test2 test2_cf stalkers_test2
"""

import argparse
import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

from sindy_wm.evaluation.harness import CACHE, PHASE3_FLAG  # noqa: E402
from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import RAW_DIR  # noqa: E402
from sindy_wm.states.unit_features import features_for_dataset  # noqa: E402

STALKER = {"reach": 6 + 2 * 0.625, "unit_health": 160.0, "damage": 18.0}
CF = ["advance_focus", "advance_nearest", "hold_focus", "hold_nearest", "withdraw_focus", "withdraw_nearest"]


def jobs_for(name: str) -> list[tuple]:
    """(raw dir, episode ids, id offset, kwargs, output name) chunks of one set."""
    if name == "train":
        sources = [(RAW_DIR / "exp06_intents" / f"train_{i}", i * 1000, {}) for i in range(6)]
    elif name in ("val", "test2"):
        sources = [(RAW_DIR / "exp06_intents" / name, 0, {})]
    elif name == "test2_cf":  # ids as harness.load_states: 10_000 * (k + 1) + episode
        sources = [
            (RAW_DIR / "exp06_intents" / f"test2_cf_{c}", 10_000 * (k + 1), {}) for k, c in enumerate(CF)
        ]
    elif name in ("stalkers_train", "stalkers_test2"):
        sources = [(RAW_DIR / "exp07_stalkers" / name.split("_")[1], 0, STALKER)]
    else:
        raise ValueError(name)
    if name.startswith("test2") or name.endswith("test2"):
        assert os.environ.get(PHASE3_FLAG) == "1", f"{name} is Phase 3 only"
    out = []
    for raw, offset, kw in sources:
        ids = storage.load_episode_index(raw)["episode_idx"].tolist()
        for k in range(0, len(ids), 100):
            out.append((raw, ids[k : k + 100], offset, kw))
    return out


def _run(job):
    raw, ids, offset, kw = job
    return features_for_dataset(raw, ids, offset=offset, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="*", default=["train", "val", "stalkers_train"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for name in args.sets:
            path = CACHE / f"features_{name}.parquet"
            if path.exists():
                print("exists", path.name, flush=True)
                continue
            table = pd.concat(pool.map(_run, jobs_for(name))).sort_index()
            table.to_parquet(path)
            print("built", path.name, table.shape, flush=True)


if __name__ == "__main__":
    main()
