"""Count units firing per side and step for exp06 (train shards and val), on the common scale.

Firing = alive and `target_in_range` (own target within attack range). Written to
data/processed/exp06_intents/firing_{train,val}.parquet, indexed like train/val.parquet
(train episode ids are shard * 1000 + episode_idx). Used by notebooks/sindyc_06.

    uv run scripts/build_firing_counts.py
"""

import pandas as pd

from sindy_wm.io import storage
from sindy_wm.paths import DATA_DIR, RAW_DIR

OUT = DATA_DIR / "processed" / "exp06_intents"


def firing(name: str, offset: int = 0) -> pd.DataFrame:
    snap = storage.load_snapshots(RAW_DIR / "exp06_intents" / name)
    snap = snap.loc[snap["alive"] & snap["target_in_range"].astype(bool), ["episode_idx", "step", "team"]]
    F = snap.groupby(["episode_idx", "step", "team"]).size().unstack("team")
    F = F.reindex(columns=["blue", "red"]).fillna(0) / 10
    F.columns = ["firing_blue", "firing_red"]
    F = F.reset_index()
    F["episode_idx"] += offset
    return F.set_index(["episode_idx", "step"])


def main() -> None:
    pd.concat([firing(f"train_{i}", i * 1000) for i in range(6)]).to_parquet(OUT / "firing_train.parquet")
    firing("val").to_parquet(OUT / "firing_val.parquet")


if __name__ == "__main__":
    main()
