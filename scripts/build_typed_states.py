"""Per-type state tables for any scenario, with the reach capped at 6 (docs/generalisation/00_plan.md, WP3).

One builder for every scenario of the generalisation test: reads data/raw/<experiment>/<set>, computes
`states.typed.load_typed_states(..., max_reach=6)`, joins the intents and the episode index, and writes
data/processed/<experiment>_typed/<set>.parquet. Train shards (train_0, train_1, ...) are merged with
episode ids shard * 1000 + episode. Held-out sets (names starting with "test") need --held-out.

    uv run scripts/build_typed_states.py exp06_intents MARINE train val
    uv run scripts/build_typed_states.py exp09_marauders MARAUDER train val
    uv run scripts/build_typed_states.py exp08_mixed MARINE MARAUDER train val
"""

import argparse
import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.typed import load_typed_states  # noqa: E402

MAX_REACH = 6.0  # SMAClite's AGENT_TARGET_RANGE
INTENT_COLS = ("segment", "movement", "target", "branched", "reverted", "branch_at")


def build_one(job: tuple[str, str, tuple[str, ...]]) -> pd.DataFrame:
    experiment, name, types = job
    d = RAW_DIR / experiment / name
    states = load_typed_states(d, types, max_reach=MAX_REACH)
    controls = storage.load_controls(d).set_index(["episode_idx", "step"])
    cols = [c for c in INTENT_COLS if c in controls]
    states = states.join(controls[cols])
    states[cols] = states.groupby(level="episode_idx")[cols].ffill()
    index = storage.load_episode_index(d).set_index("episode_idx")
    keep = ["seed", "battle_won", "n_blue", "n_red"] + [
        c for c in index if c.startswith(("n_marine", "n_marauder"))
    ]
    return states.join(index[keep], on="episode_idx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("items", nargs="+", help="unit types (upper case), then set names")
    parser.add_argument("--held-out", action="store_true", help="allow sets whose name starts with 'test'")
    args = parser.parse_args()
    types = tuple(x for x in args.items if x.isupper())
    sets = [x for x in args.items if not x.isupper()]
    if any(s.startswith("test") for s in sets) and not args.held_out:
        raise SystemExit("held-out set requested: pass --held-out (only in the pre-registered test)")
    raw, out = RAW_DIR / args.experiment, DATA_DIR / "processed" / f"{args.experiment}_typed"
    out.mkdir(parents=True, exist_ok=True)
    for name in sets:
        if (out / f"{name}.parquet").exists():
            print("exists", name)
            continue
        shards = sorted(p.name for p in raw.glob(f"{name}_*") if p.name.split("_")[-1].isdigit())
        if (raw / name).exists():
            table = build_one((args.experiment, name, types))
        elif shards:
            with multiprocessing.get_context("spawn").Pool(len(shards)) as pool:
                parts = pool.map(build_one, [(args.experiment, s, types) for s in shards])
            for s, t in zip(shards, parts, strict=True):
                t.index = t.index.set_levels(t.index.levels[0] + int(s.split("_")[-1]) * 1000, level=0)
            table = pd.concat(parts)
        else:
            raise SystemExit(f"no raw data for {args.experiment}/{name}")
        table.to_parquet(out / f"{name}.parquet")
        print(
            f"built {args.experiment}_typed/{name}: {table.index.get_level_values(0).nunique()} battles",
            flush=True,
        )


if __name__ == "__main__":
    main()
