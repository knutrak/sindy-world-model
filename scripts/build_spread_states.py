"""Health-spread states per side and unit type (docs/planning/09_spread_state_plan.md).

Over the alive units of a side and type, with h_i = (hp + shield) / full:
    even_<side>_<type>      evenness c = (Σ h_i)² / (N Σ h_i²); 1 with one unit or none
    fragile_<side>_<type>   share of alive units with h_i < 0.5; 0 with none alive
Indexed (episode_idx, step) like the per-type tables (train shards: episode ids shard * 1000 + episode).
Writes data/processed/<experiment>_spread/<set>.parquet. Held-out sets (names starting with "test" or
"final") are refused.

    uv run scripts/build_spread_states.py exp06_intents MARINE train val cf_advance_nearest ...
"""

import argparse
import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.spread import spread  # noqa: E402


def build_one(job: tuple) -> pd.DataFrame:
    experiment, name, types = job
    d = RAW_DIR / experiment / name
    ids = storage.load_episode_index(d)["episode_idx"].tolist()
    parts = [
        spread(storage.load_snapshots(d, c.tolist()), types) for c in np.array_split(np.asarray(ids), 10)
    ]
    return pd.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("items", nargs="+", help="unit types (upper case), then set names")
    args = parser.parse_args()
    types = tuple(x for x in args.items if x.isupper())
    sets = [x for x in args.items if not x.isupper()]
    if any(s.startswith(("test", "final")) for s in sets):
        raise SystemExit("held-out set requested: not allowed (all spent)")
    raw, out = RAW_DIR / args.experiment, DATA_DIR / "processed" / f"{args.experiment}_spread"
    out.mkdir(parents=True, exist_ok=True)
    for name in sets:
        if (out / f"{name}.parquet").exists():
            print("exists", name)
            continue
        shards = sorted(p.name for p in raw.glob(f"{name}_*") if p.name.split("_")[-1].isdigit())
        if (raw / name).exists():
            table = build_one((args.experiment, name, types))
        else:
            with multiprocessing.get_context("spawn").Pool(len(shards)) as pool:
                parts = pool.map(build_one, [(args.experiment, s, types) for s in shards])
            for s, t in zip(shards, parts, strict=True):
                t.index = t.index.set_levels(t.index.levels[0] + int(s.split("_")[-1]) * 1000, level=0)
            table = pd.concat(parts)
        table.to_parquet(out / f"{name}.parquet")
        print(
            f"built {args.experiment}_spread/{name}: {table.index.get_level_values(0).nunique()} battles",
            flush=True,
        )


if __name__ == "__main__":
    main()
