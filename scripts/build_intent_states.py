"""Build team-level state tables for the exp06 intent datasets (one parquet per set).

Each table has one row per (set, episode_idx, step): the team state on the common
scale (`states.geometry.load_team_states`) joined with the intent in force at that
step (movement, target, and the branch flags). Train shards are merged into one
table with unique episode ids (shard * 1000 + episode_idx).

    uv run scripts/build_intent_states.py
"""

import pandas as pd

from sindy_wm.io import storage
from sindy_wm.paths import DATA_DIR, RAW_DIR
from sindy_wm.states.geometry import load_team_states

RAW = RAW_DIR / "exp06_intents"
OUT = DATA_DIR / "processed" / "exp06_intents"


def build(name: str) -> pd.DataFrame:
    dataset_dir = RAW / name
    states = load_team_states(dataset_dir)
    controls = storage.load_controls(dataset_dir).set_index(["episode_idx", "step"])
    # the action at step t leads to step t + 1; the last snapshot has no action: carry the last intent forward
    states = states.join(controls[["segment", "movement", "target", "branched", "reverted", "branch_at"]])
    cols = ["segment", "movement", "target", "branched", "reverted", "branch_at"]
    states[cols] = states.groupby(level="episode_idx")[cols].ffill()
    index = storage.load_episode_index(dataset_dir).set_index("episode_idx")
    return states.join(index[["seed", "battle_won", "n_blue", "n_red"]], on="episode_idx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(
        p.name
        for p in RAW.iterdir()
        if (p / "manifest.json").exists() and storage.load_manifest(p)["status"] == "complete"
    )
    if any(n.startswith("train_") for n in names) and len([n for n in names if n.startswith("train_")]) < 6:
        names = [n for n in names if not n.startswith("train_")]  # wait until every shard is done
    train = []
    for name in names:
        out = OUT / f"{name}.parquet"
        if name.startswith("train_"):
            if (OUT / "train.parquet").exists():
                continue
            table = build(name).reset_index()
            table["episode_idx"] += int(name.split("_")[1]) * 1000
            train.append(table.set_index(["episode_idx", "step"]))
        elif not out.exists():
            build(name).to_parquet(out)
            print("built", name, flush=True)
    if train:
        pd.concat(train).to_parquet(OUT / "train.parquet")
        print("built train", flush=True)


if __name__ == "__main__":
    main()
