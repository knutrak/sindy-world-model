"""Run final-exploration experiments through the harness, in parallel (one process per split / fold).

Each experiment module in this folder defines `EXPERIMENTS: list[harness.Experiment]`.

    uv run --extra explore experiments/final_exploration/run.py ref_baselines
    uv run --extra explore experiments/final_exploration/run.py ref_baselines --only ref/A --splits cv val
    uv run --extra explore experiments/final_exploration/run.py ref_baselines --workers 8

Rows already in runs/final_exploration/results.csv (same config hash, split and fold) are skipped.
"""

import argparse
import importlib
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sindy_wm.evaluation import harness as H  # noqa: E402

_CACHE: dict = {}


def _job(args):
    module, name, split, fold = args
    exp = {e.name: e for e in importlib.import_module(module).EXPERIMENTS}[name]
    try:
        H.run(exp, splits=[split], cache=_CACHE, cv_folds=[fold] if split == "cv" else None)
        return name, split, fold, "ok"
    except Exception as e:  # report and carry on with the other jobs
        import traceback

        traceback.print_exc()
        return name, split, fold, f"FAILED: {e!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("module")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=list(H.SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    exps = importlib.import_module(args.module).EXPERIMENTS
    if args.only:
        exps = [e for e in exps if e.name in args.only]
    done = H.done_keys()
    jobs = [
        (args.module, e.name, s, f)
        for e in exps
        for s in args.splits
        for f in (range(H.N_FOLDS) if s == "cv" else [0])
        if (e.hash, s, f) not in done
    ]
    # slow experiments first, so the pool finishes evenly
    jobs.sort(key=lambda j: -next(e for e in exps if e.name == j[1]).config.get("cost", 1))
    print(f"{len(jobs)} jobs", flush=True)
    # "spawn": Accelerate deadlocks in forked children (see scripts/evaluate_final.py)
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for name, split, fold, status in pool.imap_unordered(_job, jobs):
            print(f"[{status}] {name} {split} {fold}", flush=True)


if __name__ == "__main__":
    main()
