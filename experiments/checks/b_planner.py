# ruff: noqa: B023, E402, E501, UP031
"""A, B and S as planners in S1 (seeds 21 000 000 +, already used in docs/planning/35 and 37).

uv run --extra explore python experiments/checks/b_planner.py 400"""

import os
import pickle
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
for p in ("experiments/planning", "notebooks/lib"):
    sys.path.insert(0, f"{ROOT}/{p}")
from multiprocessing import Pool

import numpy as np
import pandas as pd
import phase_b as PB
import sindyc_07_lab as lab

from sindy_wm.envs.planner import PipelinePlannerPolicy
from sindy_wm.models.intent_model import SHAPE, STATE
from sindy_wm.states.geometry import state_from_game

P = pickle.load(open(f"{ROOT}/data/processed/exp06_intents/sindyc_07_models.pkl", "rb"))
MODELS = {"A": P["A model_derived"], "B": P["B refit, same terms"], "S": P["S share as a state"]}
OUT = f"{ROOT}/data/processed/planning/b_planner_S1.csv"
FROZEN = pickle.load(open(PB.MODELS, "rb"))["marines"]


class Lab(PipelinePlannerPolicy):
    def __init__(s, p, h):
        super().__init__(FROZEN, commit_seconds=2.0, horizon_seconds=h, withdraw_radius=7.0)
        s.p = p

    def predict(s, env):
        st = state_from_game(env.unwrapped)
        n = len(s._U)

        def rep(cols):
            return np.nan_to_num(np.repeat(np.array([[float(st[c]) for c in cols]]), n, 0))

        B = {"X0": rep(STATE), "shape0": rep(SHAPE), "reach0": rep(["reach_blue", "reach_red"]), "U": s._U}
        Y = lab.simulate(s.p, B)
        return Y[:, -1, 0] - Y[:, -1, 1]


orig = PB.planner


def planner(name, scen, wd):
    k, h = name.split(":")
    if k.startswith("modellab"):
        return Lab(MODELS[k[len("modellab") :]], float(h[1:]))
    return orig(name, scen, wd)


PB.planner = planner


def job(j):
    return PB.play(j)


if __name__ == "__main__":
    t = time.time()
    seeds = range(21_000_000, 21_000_000 + int(sys.argv[1]))
    names = [f"modellab{m}:h3" for m in "ABS"] + ["model:h3"]
    with Pool(8) as pool:
        rows = pool.map(job, [("S1", s, n) for s in seeds for n in names], chunksize=4)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("seconds", round(time.time() - t))
    d = pd.DataFrame(rows)
    print(d.groupby("policy")[["won", "margin", "wall_s"]].mean().round(3))
