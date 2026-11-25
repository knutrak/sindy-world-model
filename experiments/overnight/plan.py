# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 4: planning (docs/overnight/00_plan.md). Seeds 50 x00 000 +, five candidates, horizon 3 s, as planning/43.

plan.py time | run S1 ... | analyse
"""

import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
for p in ("experiments/overnight", "experiments/planning"):
    sys.path.insert(0, str(ROOT / p))
import closed34 as C34
import numpy as np
import pandas as pd
import phase_b as PB

from sindy_wm.envs.planner import CANDIDATES, PipelinePlannerPolicy, typed_batch_from_row
from sindy_wm.models.param import Param
from sindy_wm.paths import DATA_DIR

OUT, DOC = DATA_DIR / "processed" / "overnight", ROOT / "docs" / "overnight" / "03_planning.md"
TYPES = {"S1": ("MARINE",)} | {f"S{i}": ("MARINE", "MARAUDER") for i in (2, 3, 4, 5)}
FIRST = {f"S{i + 1}": 50_000_000 + 100_000 * i for i in range(5)}
NAMES = ("B1", "H5", "I7")
_fin = {}
_old = PB.planner


class ParamPlanner(PipelinePlannerPolicy):
    """Adds the live closing speed (from the snapshot's unit velocities) that `vd` models need."""

    def predict(self, env):
        row = self.live_state(env)
        B = typed_batch_from_row(row, self.model.L, self._U)
        s = self._snap[self._snap["alive"]]
        c = s.groupby("team")[["x", "y", "vx", "vy"]].mean()
        ax = c.loc["red", ["x", "y"]].to_numpy(float) - c.loc["blue", ["x", "y"]].to_numpy(float)
        dv = c.loc["red", ["vx", "vy"]].to_numpy(float) - c.loc["blue", ["vx", "vy"]].to_numpy(float)
        B["vd0"] = np.full(len(self._U), float(ax @ dv / max(np.linalg.norm(ax), 1e-6)))
        with np.errstate(all="ignore"):
            Y = self.model.L.team_view(self.model.rollout(B))
        self._Y = Y
        return Y[:, -1, 0] - Y[:, -1, 1]


def planner(name, scen, wd):
    key = name.removeprefix("model").split(":")[0]
    if key not in NAMES:
        return _old(name, scen, wd)
    if not _fin:
        _fin.update(pickle.load(open(OUT / "finalists.pkl", "rb")))
    m = Param(TYPES[scen], wd, _fin[(key, "all")])
    p = ParamPlanner(m, commit_seconds=2.0, horizon_seconds=3.0, withdraw_radius=wd, candidates=CANDIDATES)
    p.trace = True
    return p


PB.planner = planner


def play(job):
    return PB.play(job)


def policies(scen):
    return [C34.BEST_FIXED[scen], "model:h3", *[f"model{n}:h3" for n in NAMES], "oracle5-h3"]


def run(scen, names, n, first, tag=""):
    path = OUT / f"plan_{scen}{tag}.csv"
    done = set()
    if path.exists():
        o = pd.read_csv(path)
        done = set(zip(o["seed"], o["policy"], strict=True))
    jobs = [(scen, first + i, p) for p in names for i in range(n) if (first + i, p) not in done]
    rows = []
    with Pool(8) as pool:
        for k, r in enumerate(pool.imap_unordered(play, jobs, chunksize=2)):
            rows.append(r)
            if len(rows) >= 200 or k == len(jobs) - 1:
                t = pd.DataFrame(rows)
                if path.exists():
                    t = pd.concat([pd.read_csv(path), t])
                t.to_csv(path, index=False)
                rows = []
    print(scen, "done", len(jobs), flush=True)


def analyse():
    rng = np.random.default_rng(0)
    wins, rows, ver = [], [], {}
    for scen in FIRST:
        path = OUT / f"plan_{scen}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        best = C34.BEST_FIXED[scen]
        wins.append(d.groupby("policy")["won"].mean().rename(scen))
        for a in ("modelB1:h3", "modelH5:h3", "modelI7:h3", "oracle5-h3"):
            for lab, b in (("L1", "model:h3"), ("L2", best)):
                mean, lo, hi, read, n = C34.paired(d, a, b, rng)
                rows.append(
                    {
                        "scenario": scen,
                        "rule": lab,
                        "policy": a,
                        "against": b,
                        "difference": mean,
                        "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                        "read-out": read,
                        "n": n,
                    }
                )
                ver[(scen, a, lab)] = read
        w = d.groupby("policy")["won"].mean()
        rows[-1]["share of perfect model's lead"] = ", ".join(
            f"{k}: {(w[f'model{k}:h3'] - w[best]) / (w['oracle5-h3'] - w[best]):+.2f}" for k in NAMES
        )
    L = [
        "# Planning with the finalists (seeds 50 x00 000 +, 400 battles)",
        "",
        "Plan: `00_plan.md`. Script: `experiments/overnight/plan.py`.",
        "",
        "## Rules",
        "",
    ]
    for a in ("modelB1:h3", "modelH5:h3", "modelI7:h3"):
        for lab, b in (("L1", "the frozen planner"), ("L2", "the best fixed intent")):
            r = [ver[(s, a, lab)] for s in FIRST if (s, a, lab) in ver]
            L.append(
                f"- {a} {lab} against {b}: beats {r.count('beats')}, ties {r.count('ties')}, loses {r.count('loses')} -> {'met' if r.count('beats') >= 2 and not r.count('loses') else 'not met'}"
            )
    L += [
        "",
        "## Wins",
        "",
        "```",
        pd.concat(wins, axis=1).round(3).to_string(),
        "```",
        "",
        "## Paired comparisons",
        "",
        "```",
        pd.DataFrame(rows).round(3).to_string(index=False),
        "```",
        "",
    ]
    DOC.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "time":
        for sc in ("S1", "S2"):
            run(sc, policies(sc), 8, 59_000_000, "_time")
            print(
                pd.read_csv(OUT / f"plan_{sc}_time.csv")
                .groupby("policy")["wall_s"]
                .mean()
                .round(2)
                .to_string()
            )
    elif sys.argv[1] == "run":
        for sc in sys.argv[2:]:
            run(sc, policies(sc), 400, FIRST[sc])
    else:
        analyse()
