"""Why does the model value an early withdrawal in S3 but not in S1? (exploratory, docs/planning/24)

A small counterfactual set: per seed, random intents up to an early branch moment (0.5-4 s), then
advance/focus,
hold/focus or withdraw for 2 s, then advance/nearest. At the branch and 3 s later the per-type state is
recorded.
The frozen model is rolled out from the branch state under each branch's controls; predicted and true
changes are
compared, split into Blue's and Red's health loss.

    uv run --extra explore experiments/planning/s3_mechanism.py run       (a few minutes)
    uv run --extra explore experiments/planning/s3_mechanism.py analyse
"""

import argparse
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.intents import IntentSchedule  # noqa: E402
from sindy_wm.envs.planner import typed_batch_from_row  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger  # noqa: E402
from sindy_wm.models.intent_model import DT, control_vector  # noqa: E402
from sindy_wm.states.typed import typed_states  # noqa: E402

OUT = A.OUT / "s3_mechanism.parquet"
SEEDS = {"S3": 19_800_500, "S1": 19_800_000}
N = 150
BR = (("advance", "focus"), ("hold", "focus"), ("withdraw", "nearest"))
MODEL_OF = {"S1": "marines", "S3": "mixed"}
TYPES = {"S1": ("MARINE",), "S3": ("MARINE", "MARAUDER")}


def play(job) -> dict | None:
    scen, seed, (mv, tg) = job
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
    log = SnapshotLogger(env)
    log.start_episode()
    pol = IntentSchedule(
        branch_intent=(mv, tg), branch_seconds=(0.5, 4.0), branch_hold_seconds=2.0, withdraw_radius=wd
    )
    pol.reset(seed)
    k = round(pol._branch_at / env.seconds_per_step)
    n3 = round(3.0 / env.seconds_per_step)
    rows, step = {}, 0
    while step <= k + n3:
        if step in (k, k + n3):
            if not (env.unwrapped.agents and env.unwrapped.enemies):
                return None
            rows[step] = typed_states(log.snapshot(step=step), TYPES[scen], max_reach=6.0).iloc[0].to_dict()
        _, _, term, trunc, _ = env.step(pol.act(env))
        step += 1
        if (term or trunc) and step <= k + n3:
            return None
    return {"scenario": scen, "seed": seed, "branch": f"{mv}/{tg}", "at": rows[k], "after": rows[k + n3]}


def run() -> None:
    jobs = [(scen, first + i, b) for scen, first in SEEDS.items() for i in range(N) for b in BR]
    with Pool(9) as pool:
        res = [r for r in pool.map(play, jobs, chunksize=3) if r is not None]
    pd.to_pickle(res, OUT)
    print(len(res), "branch states")


def analyse(withdraw_fire: dict | None = None) -> list[str]:
    """withdraw_fire: {model name: f} for the wfire variant (docs/planning/25); None = frozen."""
    models = pickle.load(
        open(A.DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl", "rb")
    )
    res = pd.read_pickle(OUT)
    out = []
    for scen in SEEDS:
        m = models[MODEL_OF[scen]]
        if withdraw_fire:
            m.withdraw_fire = withdraw_fire[MODEL_OF[scen]]
        L = m.L
        by = {}
        for r in res:
            if r["scenario"] == scen:
                by.setdefault(r["seed"], {})[r["branch"]] = r
        seeds = [s for s, v in by.items() if len(v) == len(BR)]
        recs = []
        for sd in seeds:
            for b in by[sd].values():
                mv, tg = b["branch"].split("/")
                steps, commit = round(3.0 / DT), round(2.0 / DT)
                U = np.repeat(control_vector("advance", "nearest")[None], steps, 0)
                U[:commit] = control_vector(mv, tg)
                row = pd.Series(b["at"])
                B = typed_batch_from_row(row, L, U[None])
                with np.errstate(invalid="ignore", divide="ignore"):
                    Y = L.team_view(m.rollout(B))[0]
                X0 = L.team_view(np.array([[row[c] for c in L.x_cols]], float))[0]
                X1 = L.team_view(np.array([[pd.Series(b["after"])[c] for c in L.x_cols]], float))[0]
                recs.append(
                    {
                        "seed": sd,
                        "branch": b["branch"],
                        "blue loss, true": X0[0] - X1[0],
                        "red loss, true": X0[1] - X1[1],
                        "blue loss, model": Y[0, 0] - Y[-1, 0],
                        "red loss, model": Y[0, 1] - Y[-1, 1],
                        "distance": row["distance"],
                    }
                )
        f = pd.DataFrame(recs)
        f["margin change, true"] = f["red loss, true"] - f["blue loss, true"]
        f["margin change, model"] = f["red loss, model"] - f["blue loss, model"]
        piv = f.pivot_table(
            index="seed", columns="branch", values=[c for c in f if c.startswith(("blue", "red", "margin"))]
        )
        rows = []
        for ref in ("advance/focus", "hold/focus"):
            d = {
                c: (piv[(c, "withdraw/nearest")] - piv[(c, ref)]).mean()
                for c in (
                    "margin change, true",
                    "margin change, model",
                    "blue loss, true",
                    "blue loss, model",
                    "red loss, true",
                    "red loss, model",
                )
            }
            agree = np.sign(
                piv[("margin change, model", "withdraw/nearest")] - piv[("margin change, model", ref)]
            )
            truth = np.sign(
                piv[("margin change, true", "withdraw/nearest")] - piv[("margin change, true", ref)]
            )
            rows.append(
                {
                    "withdraw minus": ref,
                    **d,
                    "sign agreement": float((agree == truth).mean()),
                    "model says withdraw better": float((agree > 0).mean()),
                    "truly better": float((truth > 0).mean()),
                    "states": len(piv),
                }
            )
        out += [
            f"## {scen}: withdraw for 2 s at an early branch (0.5-4 s), effect over 3 s",
            "(differences of means over states; losses in ten units of the first type)",
            "",
            "```",
            pd.DataFrame(rows).set_index("withdraw minus").T.round(4).to_string(),
            "```",
            "",
        ]
    print("\n".join(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["run", "analyse"])
    {"run": run, "analyse": analyse}[parser.parse_args().what]()


if __name__ == "__main__":
    main()
