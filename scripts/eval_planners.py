"""Play close battles with different Blue policies and record who wins (exp06, exploratory).

Each seed is one starting condition (`CloseBattleSampler`); every policy plays the same seeds.
Results go to data/processed/exp06_intents/planner_<tag>.csv: one row per (seed, policy).

    uv run scripts/eval_planners.py --policies fixed model --n 400
"""

import argparse
import pickle
from multiprocessing import Pool

import pandas as pd

from sindy_wm.envs.episode import _set_start_health
from sindy_wm.envs.episode_sampling import CloseBattleSampler
from sindy_wm.envs.intents import IntentPolicy, IntentSchedule
from sindy_wm.envs.planner import ModelPlannerPolicy, OraclePlannerPolicy
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.paths import DATA_DIR
from sindy_wm.states.geometry import MARINE_HP, REF_UNITS

OUT = DATA_DIR / "processed" / "exp06_intents"
SCENARIO, STEP_MUL, MAX_STEPS, FIRST_SEED = "10m_vs_10m", 2, 480, 6_000_000
SAMPLER = CloseBattleSampler()
_MODELS = {}


def model(name: str):
    if name not in _MODELS:
        with open(OUT / f"model_{name}.pkl", "rb") as f:
            _MODELS[name] = pickle.load(f)
    return _MODELS[name]


def make_policy(name: str):
    if name.startswith("fixed:"):
        return IntentPolicy(*name.split(":")[1].split("/"))
    if name == "random switching":
        return IntentSchedule()
    if name.startswith("oracle"):  # oracle[:commit:horizon]
        parts = name.split(":")
        return (
            OraclePlannerPolicy(commit_seconds=float(parts[1]), horizon_seconds=float(parts[2]))
            if len(parts) > 1
            else OraclePlannerPolicy()
        )
    if name.startswith("model:"):  # model:<model name>:<shape>:<commit>:<horizon>[:reach][:margin=x]
        _, mname, shape, commit, horizon, *rest = name.split(":")
        margin = next((float(r.split("=")[1]) for r in rest if r.startswith("margin=")), 0.0)
        return ModelPlannerPolicy(
            model(mname),
            commit_seconds=float(commit),
            horizon_seconds=float(horizon),
            shape=shape,
            use_reach="noreach" not in rest,
            switch_margin=margin,
        )
    raise ValueError(name)


def play(args):
    seed, name = args
    bounds = walkable_bounds(load_map_info(SCENARIO))
    s = SAMPLER.sample(seed, bounds)
    env = make_env(SCENARIO, step_mul=STEP_MUL, spawn=s.spawn, unit_counts=s.unit_counts)
    env.reset(seed=seed)
    _set_start_health(env, *s.start_health)
    policy = make_policy(name)
    policy.reset(seed)
    won, steps, chosen = False, 0, {}
    while steps < MAX_STEPS:
        steps += 1
        actions = policy.act(env)
        key = "/".join(str(policy.setting().get(k)) for k in ("movement", "target"))
        chosen[key] = chosen.get(key, 0) + 1
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            won = bool(info.get("battle_won", False))
            break
    game = env.unwrapped
    scale = REF_UNITS * MARINE_HP
    blue, red = (
        sum(u.hp for u in game.agents.values()) / scale,
        sum(u.hp for u in game.enemies.values()) / scale,
    )
    return {
        "seed": seed,
        "policy": name,
        "won": won,
        "margin": blue - red,
        "steps": steps,
        "strength": s.unit_counts[0] * s.start_health[0] / (s.unit_counts[1] * s.start_health[1]),
        "spawn_distance": s.record()["spawn_distance"],
        **{f"share_{k}": v / steps for k, v in chosen.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(FIRST_SEED + i, p) for p in args.policies for i in range(args.n)]
    with Pool(args.workers) as pool:
        rows = pool.map(play, jobs, chunksize=4)
    out = OUT / f"planner_{args.tag}.csv"
    table = pd.DataFrame(rows)
    if out.exists():
        table = pd.concat([pd.read_csv(out), table]).drop_duplicates(["seed", "policy"], keep="last")
    table.to_csv(out, index=False)
    print(table.groupby("policy")[["won", "margin"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
