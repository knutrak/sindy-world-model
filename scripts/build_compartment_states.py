"""Health-compartment states per side and unit type (docs/planning/13_compartment_plan.md): W, H_W and the
mean
start health h0, from the raw snapshots (`states.compartments`). Indexed (episode_idx, step) like the per-type
tables (train shards: episode ids shard * 1000 + episode). Writes
data/processed/<experiment>_compart/<set>.parquet.
Held-out sets (names starting with "test" or "final") are refused.

    uv run scripts/build_compartment_states.py exp06_intents MARINE train val cf_advance_nearest ...
"""

import argparse
import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.compartments import compartment_events, compartments  # noqa: E402


def build_one(job: tuple) -> pd.DataFrame:
    experiment, name, types, events = job
    d = RAW_DIR / experiment / name
    ids = storage.load_episode_index(d)["episode_idx"].tolist()
    f = compartment_events if events else compartments
    return pd.concat(
        f(storage.load_snapshots(d, c.tolist()), types) for c in np.array_split(np.asarray(ids), 10)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment")
    parser.add_argument("items", nargs="+", help="unit types (upper case), then set names")
    parser.add_argument("--events", action="store_true", help="per-step events (docs/planning/15) instead")
    args = parser.parse_args()
    types = tuple(x for x in args.items if x.isupper())
    sets = [x for x in args.items if not x.isupper()]
    if any(s.startswith(("test", "final")) for s in sets):
        raise SystemExit("held-out set requested: not allowed (all spent)")
    raw, out = RAW_DIR / args.experiment, DATA_DIR / "processed" / f"{args.experiment}_compart"
    out.mkdir(parents=True, exist_ok=True)
    for name in sets:
        fname = f"{name}_events.parquet" if args.events else f"{name}.parquet"
        if (out / fname).exists():
            print("exists", name)
            continue
        shards = sorted(p.name for p in raw.glob(f"{name}_*") if p.name.split("_")[-1].isdigit())
        if (raw / name).exists():
            table = build_one((args.experiment, name, types, args.events))
        else:
            with multiprocessing.get_context("spawn").Pool(len(shards)) as pool:
                parts = pool.map(build_one, [(args.experiment, s, types, args.events) for s in shards])
            for s, t in zip(shards, parts, strict=True):
                t.index = t.index.set_levels(t.index.levels[0] + int(s.split("_")[-1]) * 1000, level=0)
            table = pd.concat(parts)
        table.to_parquet(out / fname)
        n = table.index.get_level_values(0).nunique()
        print(f"built {args.experiment}_compart/{name}: {n} battles", flush=True)


if __name__ == "__main__":
    main()
