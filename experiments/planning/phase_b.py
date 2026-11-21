"""Phase B (docs/planning/07_phase_b_plan.md): the frozen pipeline model as a planner, on new seeds.

Policies per scenario: the model planner (primary: horizon 3 s; secondary: 2 s and 4 s; exploratory: the
focus-cost variant at 3 s), all re-planning every 1 s with a 2 s commit; the six fixed intents and random
switching; the exact oracle (Phase A's: commit 2 s, horizon 8 s) on the first 200 seeds. Rows go to
data/processed/planning/phase_b_<scenario>.csv (one per seed and policy; reruns skip done rows).

    uv run experiments/planning/phase_b.py smoke --n 20                  every policy, seeds 19 700 000 +
    uv run experiments/planning/phase_b.py run S1 --n 400 --policies model fixed
    uv run experiments/planning/phase_b.py run S1 --n 200 --policies oracle
"""

import argparse
import copy
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.planner import CANDIDATES, PipelinePlannerPolicy  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402

FIRST_SEED = {"S1": 14_000_000, "S2": 14_100_000, "S3": 14_200_000, "S4": 14_300_000, "S5": 14_400_000}
SMOKE_SEED = 19_700_000
MODEL_OF = {"S1": "marines", "S2": "mixed", "S3": "mixed", "S4": "mixed", "S5": "mixed"}
MODELS = A.DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
COMPART = A.OUT / "compart_models.pkl"  # docs/planning/13, fitted on train
SPREAD = A.OUT / "spread_models.pkl"  # docs/planning/09, fitted on train
VARIANTS = A.OUT / "intent_variants.pkl"  # focus cost c, f̄ per scenario (S1 marines, S2 mixed), from train
POLICY_SETS = {
    "model": ["model:h3", "model:h2", "model:h4", "model-focuscost:h3"],
    "fixed": A.POLICY_SETS["fixed"],
    "oracle": ["oracle"],
    "spread": ["model:h3", "spread:h3", "spread-focuscost:h3"],  # docs/planning/09, H6
    "compart": ["model:h3", "compart:h3", "compart-focuscost:h3"],  # docs/planning/13, H5
}
_cache = {}


def planner(name: str, scen: str, wd: float) -> PipelinePlannerPolicy:
    if "models" not in _cache:
        _cache["models"] = pickle.load(open(MODELS, "rb"))
        _cache["variants"] = pickle.load(open(VARIANTS, "rb"))
    kind, h = name.split(":")
    m = _cache["models"][MODEL_OF[scen]]
    key = "S1" if MODEL_OF[scen] == "marines" else "S2"  # the fitted variants are per model
    if kind.startswith("spread"):  # docs/planning/09_spread_state_plan.md
        if "spread" not in _cache:
            _cache["spread"] = pickle.load(open(SPREAD, "rb"))
        m = copy.deepcopy(m)
        m.spread_model = _cache["spread"][key]
    if kind.startswith("compart"):  # docs/planning/13_compartment_plan.md
        if "compart" not in _cache:
            _cache["compart"] = pickle.load(open(COMPART, "rb"))
        m = copy.deepcopy(m)
        m.compart_model = {"laws": _cache["compart"][key]["compartments"], "use_hw": True}
    if kind.endswith("focuscost"):
        f = _cache["variants"][key]
        m = copy.deepcopy(m)
        m.focus_cost, m.focus_norm = f["c"], 1 - f["c"] * f["fbar"]
    if kind == "wfire":  # docs/planning/25: Blue keeps this share of its fire while withdrawing (train)
        m = copy.deepcopy(m)
        m.withdraw_fire = {"marines": 0.093, "mixed": 0.127}[MODEL_OF[scen]]
    if kind == "terminal":  # docs/planning/25: score by the projected final margin
        from sindy_wm.envs.planner import team_firepower

        p = PipelinePlannerPolicy(m, commit_seconds=2.0, horizon_seconds=float(h[1:]), withdraw_radius=wd)
        p.terminal = team_firepower(m)
        return p
    if kind == "gated":  # docs/planning/22: default = Phase A's best fixed intent, tau per model
        import json

        from analyse_phase_b import BEST_FIXED

        from sindy_wm.envs.planner import GatedPlannerPolicy

        tau = json.load(open(A.OUT / "gated_tau.json"))[MODEL_OF[scen]]["tau"]
        default = tuple(BEST_FIXED[scen].removeprefix("fixed:").split("/"))
        return GatedPlannerPolicy(
            m, default=default, tau=tau, commit_seconds=2.0, horizon_seconds=float(h[1:]), withdraw_radius=wd
        )
    return PipelinePlannerPolicy(m, commit_seconds=2.0, horizon_seconds=float(h[1:]), withdraw_radius=wd)


def play(job: tuple) -> dict:
    scen, seed, name = job
    mapname, sampler, wd, max_s, _ = A.SCENARIOS[scen]
    s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        mapname,
        step_mul=A.STEP_MUL,
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
    is_planner = name.startswith(("model", "spread", "compart", "gated", "wfire", "terminal"))
    policy = planner(name, scen, wd) if is_planner else A.make_policy(name, wd)
    policy.reset(seed)
    t0, steps, won, chosen = time.time(), 0, False, {}
    while steps < max_s * 16 // A.STEP_MUL:
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
    row = {
        "scenario": scen,
        "seed": seed,
        "policy": name,
        "won": won,
        "margin": blue_hp / hp0["blue"] - red_hp / hp0["red"],
        "blue_preserved": len(game.agents) / n0["blue"],
        "exchange": (lost_r + 0.5) / (lost_b + 0.5),
        "n_blue": n0["blue"],
        "n_red": n0["red"],
        "start_health_blue": float(s.start_health[0]),
        "start_health_red": float(s.start_health[1]),
        "seconds": steps * A.STEP_MUL / 16,
        "wall_s": time.time() - t0,
        **{f"share_{k}": v / steps for k, v in chosen.items()},
    }
    if hasattr(policy, "deviations") and policy.log:
        row["deviate_share"] = policy.deviations / len(policy.log)
    if is_planner and policy.log:
        S = np.array(policy.log)
        picks = S.argmax(1)
        top2 = np.sort(S, 1)[:, -2:]
        row |= {
            "replans": len(S),
            "switches": int((np.diff(picks) != 0).sum()),
            "gap_best_second": float((top2[:, 1] - top2[:, 0]).mean()),
            **{f"pick_{'/'.join(c)}": float((picks == j).mean()) for j, c in enumerate(CANDIDATES)},
        }
    return row


def run(
    scen: str,
    policies: list[str],
    n: int,
    workers: int,
    first: int | None = None,
    tag: str = "",
    prefix: str = "phase_b_",
) -> None:
    A.OUT.mkdir(parents=True, exist_ok=True)
    path = A.OUT / f"{prefix}{scen}{tag}.csv"
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["seed"], old["policy"], strict=True))
    first = FIRST_SEED[scen] if first is None else first
    jobs = [(scen, first + i, p) for p in policies for i in range(n) if (first + i, p) not in done]
    if not jobs:
        return
    with Pool(workers) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    table = pd.DataFrame(rows)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table])
    table.to_csv(path, index=False)
    print(f"{scen}: {len(rows)} battles -> {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["smoke", "run"])
    parser.add_argument("scenario", nargs="?")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--policies", nargs="+", default=["model", "fixed"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    if args.what == "smoke":
        for scen in FIRST_SEED:
            run(scen, POLICY_SETS["model"] + POLICY_SETS["fixed"], args.n, args.workers, SMOKE_SEED, "_smoke")
            run(scen, ["oracle"], 3, args.workers, SMOKE_SEED, "_smoke")
        return
    run(args.scenario, [p for s in args.policies for p in POLICY_SETS[s]], args.n, args.workers)


if __name__ == "__main__":
    main()
