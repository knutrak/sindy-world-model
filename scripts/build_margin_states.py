"""Reach-margin state tables: mean and spread of each unit's margin per side and type (generalisation/10).

Reads data/raw/<experiment>/<set>, writes data/processed/<experiment>_margins/<set>.parquet indexed by
(episode_idx, step) as the typed tables: train shards (train_0, train_1, ...) are merged with episode ids
shard * 1000 + episode. Held-out sets (names starting with "test" or "final") are refused.

    uv run scripts/build_margin_states.py exp08_mixed MARINE MARAUDER train val cf_advance_nearest
"""

import argparse
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.io import storage  # noqa: E402
from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.margins import margins  # noqa: E402


def one(job) -> pd.DataFrame:
    d, ids, types, offset = job
    snaps = storage.load_snapshots(d, ids)
    parts = []
    for ep, g in snaps.groupby("episode_idx"):
        t = margins(g, types)
        t["episode_idx"] = ep + offset
        parts.append(t.reset_index().set_index(["episode_idx", "step"]))
    return pd.concat(parts)


def jobs_for(d, types, offset):
    ids = storage.load_episode_index(d)["episode_idx"].to_numpy()
    return [(d, c.tolist(), types, offset) for c in np.array_split(ids, max(1, len(ids) // 60)) if len(c)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("experiment")
    p.add_argument("items", nargs="+", help="unit types (upper case), then set names")
    a = p.parse_args()
    types = tuple(x for x in a.items if x.isupper())
    sets = [x for x in a.items if not x.isupper()]
    if any(s.startswith(("test", "final")) for s in sets):
        raise SystemExit("held-out set requested: refused")
    raw, out = RAW_DIR / a.experiment, DATA_DIR / "processed" / f"{a.experiment}_margins"
    out.mkdir(parents=True, exist_ok=True)
    for name in sets:
        if (out / f"{name}.parquet").exists():
            print("exists", name)
            continue
        shards = sorted(q.name for q in raw.glob(f"{name}_*") if q.name.split("_")[-1].isdigit())
        jobs = []
        if shards:
            for s in shards:
                jobs += jobs_for(raw / s, types, int(s.split("_")[-1]) * 1000)
        else:
            jobs = jobs_for(raw / name, types, 0)
        with Pool(8) as pool:
            t = pd.concat(pool.map(one, jobs)).sort_index()
        t.to_parquet(out / f"{name}.parquet")
        print(f"built {out.name}/{name}: {t.index.get_level_values(0).nunique()} battles", flush=True)


if __name__ == "__main__":
    main()
