"""Copies of the processed state tables with Red's reach measured at its *acquisition* radius.

SMAClite's Red AI (attack-move) only picks up a new target within `minimum_scan_range` + the
target's radius (5 + 0.375 = 5.375 for marines, centre to centre), while Blue can shoot out to
attack range + both radii (5.75). Red keeps a target it already has out to 5.75 and always fights
back against a unit that is shooting it, so 5.375 is a lower bound on Red's effective reach, not
an exact count of Red units that can fire.

For each set, only `reach_red` changes; every other column is the original table's. Written to
data/processed/<experiment>_redscan/<name>.parquet (the originals are untouched). Pickles that
`scripts/evaluate_final.py` loads from its processed folder are symlinked alongside.

    uv run scripts/build_red_scan_states.py            # exp06 train / val / cf_*, exp07 train / val
"""

import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.paths import DATA_DIR, RAW_DIR  # noqa: E402
from sindy_wm.states.geometry import MARINE_HP, load_team_states  # noqa: E402

PROCESSED = DATA_DIR / "processed"
SCAN = {"exp06_intents": 5 + 0.375, "exp07_stalkers": 5 + 0.625}  # minimum_scan_range + target radius
HEALTH = {"exp06_intents": MARINE_HP, "exp07_stalkers": 80.0 + 80.0}
PICKLES = ("model_derived.pkl", "closures.pkl", "sindyc_07_models.pkl")


def scan_reach(experiment: str, raw_name: str) -> pd.Series:
    states = load_team_states(
        RAW_DIR / experiment / raw_name, reach=SCAN[experiment], unit_health=HEALTH[experiment]
    )
    return states["reach_red"]


def build(job: tuple[str, str]) -> str:
    experiment, name = job
    out = PROCESSED / f"{experiment}_redscan" / f"{name}.parquet"
    if out.exists():
        return f"skip {experiment}/{name}"
    table = pd.read_parquet(PROCESSED / experiment / f"{name}.parquet")
    if experiment == "exp06_intents" and name == "train":
        parts = []
        for shard in range(6):
            r = scan_reach(experiment, f"train_{shard}").reset_index()
            r["episode_idx"] += shard * 1000
            parts.append(r.set_index(["episode_idx", "step"])["reach_red"])
        reach = pd.concat(parts)
    else:
        reach = scan_reach(experiment, name)
    reach = reach.reindex(table.index)
    assert reach.notna().all(), f"{experiment}/{name}: steps missing in the recomputed reach"
    assert (reach.to_numpy() <= table["reach_red"].to_numpy() + 1e-9).all(), "scan reach above full reach"
    table["reach_red"] = reach.to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    full = pd.read_parquet(PROCESSED / experiment / f"{name}.parquet")["reach_red"]
    return f"built {experiment}/{name}: mean reach_red {full.mean():.4f} -> {np.mean(reach):.4f}"


def main() -> None:
    cf = sorted(p.stem for p in (PROCESSED / "exp06_intents").glob("cf_*.parquet"))
    jobs = [("exp06_intents", n) for n in ("train", "val", *cf)] + [
        ("exp07_stalkers", n) for n in ("train", "val")
    ]
    with multiprocessing.get_context("spawn").Pool(9) as pool:
        for msg in pool.imap_unordered(build, jobs):
            print(msg, flush=True)
    folder = PROCESSED / "exp06_intents_redscan"
    for p in PICKLES:
        link = folder / p
        if not link.exists():
            link.symlink_to(PROCESSED / "exp06_intents" / p)


if __name__ == "__main__":
    main()
