"""Phase A of the headroom test (docs/planning/00_headroom_plan.md): how much can choosing intents gain?

Plays the same seeds in each scenario under every fixed intent, random switching and oracle planners
(re-plan every 1 s, commit 2 s, horizon 8 s, with copies of the real game; exact or perturbed), and records
the outcome for four objectives (win, health margin, Blue units preserved, exchange ratio). No world model.
Rows go to data/processed/planning/headroom_<scenario>.csv (one per seed and policy; reruns skip done rows).

    uv run experiments/planning/headroom.py calibrate            S3: marines per marauder for close battles
    uv run experiments/planning/headroom.py time --n 20          seconds per oracle battle
    uv run experiments/planning/headroom.py run S2 --n 400 --policies fixed oracle
"""

import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.episode_sampling import (  # noqa: E402
    CloseBattleSampler,
    MixedBattleSampler,
    RangeAsymmetrySampler,
    UnevenMixSampler,
)
from sindy_wm.envs.intents import INTENTS, IntentPolicy, IntentSchedule  # noqa: E402
from sindy_wm.envs.planner import OraclePlannerPolicy  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "processed" / "planning"
STEP_MUL = 2
MARINES_PER_MARAUDER = 1.8  # S3, calibrated 21 Nov: 1.75 -> Blue wins 57%, 1.85 -> 42% (200 battles)
SCENARIOS = {
    # name: (map, sampler, withdraw radius, max seconds, first seed)
    "S1": ("10m_vs_10m", CloseBattleSampler(), 7.0, 60, 13_000_000),
    "S2": ("mixed_m_mrd", MixedBattleSampler(), 8.0, 90, 13_100_000),
    "S3": (
        "mixed_m_mrd",
        RangeAsymmetrySampler(marines_per_marauder=MARINES_PER_MARAUDER),
        8.0,
        90,
        13_200_000,
    ),
    "S4": ("mixed_m_mrd", MixedBattleSampler(strength_ratio=(0.6, 0.8)), 8.0, 90, 13_300_000),
    "S5": ("mixed_m_mrd", UnevenMixSampler(), 8.0, 90, 13_400_000),
}
PERTURB = (0.2, 0.05)
POLICY_SETS = {
    "fixed": [f"fixed:{m}/{t}" for m, t in INTENTS] + ["random"],
    "oracle": ["oracle", "oracle-perturbed"],
    "survival": ["oracle-survival"],
    "reseed": ["oracle-perturbed-reseed"],  # check: copies with their own unit-order shuffle stream
}


def make_policy(name: str, wd: float):
    if name.startswith("fixed:"):
        return IntentPolicy(*name.split(":")[1].split("/"), withdraw_radius=wd)
    if name == "random":
        return IntentSchedule(withdraw_radius=wd)
    kw = {"commit_seconds": 2.0, "horizon_seconds": 8.0, "withdraw_radius": wd}
    if name == "oracle":
        return OraclePlannerPolicy(**kw)
    if name == "oracle-perturbed":
        return OraclePlannerPolicy(perturb=PERTURB, **kw)
    if name == "oracle-perturbed-reseed":
        return OraclePlannerPolicy(perturb=PERTURB, reseed=True, **kw)
    if name == "oracle-survival":  # perturbed, optimising Blue units preserved (S4's natural objective)
        return OraclePlannerPolicy(perturb=PERTURB, objective="survival", **kw)
    raise ValueError(name)


def play(job: tuple) -> dict:
    scen, seed, name, scenarios = job
    mapname, sampler, wd, max_s, _ = scenarios[scen]
    s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        mapname,
        step_mul=STEP_MUL,
        spawn=s.spawn,
        unit_counts=None if types else s.unit_counts,
        unit_types=types,
    )
    env.reset(seed=seed)
    _set_start_health(env, *s.start_health)
    game = env.unwrapped
    hp0 = {
        "blue": sum(u.max_hp + u.max_shield for u in game.agents.values()),
        "red": sum(u.max_hp + u.max_shield for u in game.enemies.values()),
    }
    n0 = {"blue": len(game.agents), "red": len(game.enemies)}
    policy = make_policy(name, wd)
    policy.reset(seed)
    t0, steps, won, chosen = time.time(), 0, False, {}
    while steps < max_s * 16 // STEP_MUL:
        steps += 1
        actions = policy.act(env)
        st = policy.setting() if hasattr(policy, "setting") else {}
        key = f"{st.get('movement')}/{st.get('target')}"
        chosen[key] = chosen.get(key, 0) + 1
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            won = bool(info.get("battle_won", False))
            break
    blue_hp = sum(u.hp + u.shield for u in game.agents.values())
    red_hp = sum(u.hp + u.shield for u in game.enemies.values())
    lost_b, lost_r = n0["blue"] - len(game.agents), n0["red"] - len(game.enemies)
    return {
        "scenario": scen,
        "seed": seed,
        "policy": name,
        "won": won,
        "margin": blue_hp / hp0["blue"] - red_hp / hp0["red"],  # health left, as a share of full health
        "blue_preserved": len(game.agents) / n0["blue"],
        "exchange": (lost_r + 0.5) / (lost_b + 0.5),
        "n_blue": n0["blue"],
        "n_red": n0["red"],
        "seconds": steps * STEP_MUL / 16,
        "wall_s": time.time() - t0,
        **{f"share_{k}": v / steps for k, v in chosen.items()},
    }


def run(scen: str, policies: list[str], n: int, workers: int, tag: str = "") -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"headroom_{scen}{tag}.csv"
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["seed"], old["policy"], strict=True))
    first = SCENARIOS[scen][4]
    jobs = [(scen, first + i, p, SCENARIOS) for p in policies for i in range(n) if (first + i, p) not in done]
    with Pool(workers) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    table = pd.DataFrame(rows)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table])
    table.to_csv(path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["calibrate", "time", "run"])
    parser.add_argument("scenario", nargs="?")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--policies", nargs="+", default=["fixed"])
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--tag", default="", help="suffix for the output file, e.g. _reseed")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0], help="calibrate")
    args = parser.parse_args()
    if args.what == "calibrate":
        rows = []
        for r in args.ratios:
            sc = dict(SCENARIOS)
            sc["S3"] = (
                *SCENARIOS["S3"][:1],
                RangeAsymmetrySampler(marines_per_marauder=r),
                *SCENARIOS["S3"][2:],
            )
            jobs = [("S3", 19_000_000 + i, "fixed:advance/nearest", sc) for i in range(args.n)]
            with Pool(args.workers) as pool:
                res = pool.map(play, jobs)
            rows.append(
                {"marines per marauder": r, "Blue wins (advance/nearest)": pd.DataFrame(res)["won"].mean()}
            )
        print(pd.DataFrame(rows).to_string(index=False))
    elif args.what == "time":
        jobs = [
            ("S2", 19_500_000 + i, p, SCENARIOS)
            for i in range(args.n)
            for p in ("oracle", "fixed:advance/nearest")
        ]
        with Pool(args.workers) as pool:
            res = pd.DataFrame(pool.map(play, jobs))
        print(res.groupby("policy")[["wall_s", "seconds"]].describe().round(2).to_string())
    else:
        policies = [p for key in args.policies for p in POLICY_SETS.get(key, [key])]
        t = run(args.scenario, policies, args.n, args.workers, args.tag)
        print(
            t.groupby("policy")[["won", "margin", "blue_preserved", "exchange"]].mean().round(3).to_string()
        )


if __name__ == "__main__":
    main()
