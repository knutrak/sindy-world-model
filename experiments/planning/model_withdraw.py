"""Does the model planner withdraw in the oracle's states? (exploratory, after docs/planning/19)

Replays the Phase B primary planner (frozen model, horizon 3 s, commit 2 s; deterministic, so the battles
are Phase
B's) on Phase B's first 200 seeds of S3 and S1, logging the state, the five predicted scores and the choice
at every
re-plan. Compares its withdraw states with the oracle's (`oracle_decisions.parquet`) and asks whether its
S3 gain
over hold/focus comes with early withdrawals.

    uv run experiments/planning/model_withdraw.py run       (about 1-2 minutes)
    uv run --extra explore experiments/planning/model_withdraw.py analyse   ->
    docs/planning/20_model_withdraw.md
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402
import phase_b as PB  # noqa: E402
from oracle_withdraw import TYPES, features, md  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.intents import IntentPolicy  # noqa: E402
from sindy_wm.envs.planner import PipelinePlannerPolicy  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402
from sindy_wm.states.compartments import compartments, unit_health  # noqa: E402
from sindy_wm.states.typed import typed_states  # noqa: E402

OUT = A.OUT / "model_decisions.parquet"
DOC = PROJECT_ROOT / "docs" / "planning" / "20_model_withdraw.md"
SCENS, N = ("S3", "S1"), 200


class LoggingPlanner(PipelinePlannerPolicy):
    def reset(self, seed=None):
        super().reset(seed)
        self.rows = []

    def act(self, env):
        game = env.unwrapped
        if self._logger is None:
            self.live_state(env)
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            scores = self.predict(env)  # sets self._snap
            snap = self._snap.assign(episode_idx=0)
            s0 = self._snap0.assign(episode_idx=0)
            start = s0.assign(h=unit_health(s0)).set_index(["episode_idx", "team", "unit_id"])["h"]
            row = typed_states(snap.drop(columns="episode_idx"), TYPES, max_reach=6.0).iloc[0].to_dict()
            row |= compartments(snap, TYPES, start=start).iloc[0].to_dict()
            best = int(np.argmax(scores))
            row |= {
                "t": self._step * env.seconds_per_step,
                "choice": "/".join(self.candidates[best]),
                "gap": float(np.sort(scores)[-1] - np.sort(scores)[-2]),
            }
            self.rows.append(row)
            self.log.append(scores)
            self._current = IntentPolicy(*self.candidates[best], withdraw_radius=self.withdraw_radius)
        self._step += 1
        return self._current.act(env)


def play(job) -> list[dict]:
    scen, seed = job
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
    base = PB.planner("model:h3", scen, wd)
    pol = LoggingPlanner(base.model, commit_seconds=2.0, horizon_seconds=3.0, withdraw_radius=wd)
    pol.reset(seed)
    steps, won = 0, False
    while steps < max_s * 16 // A.STEP_MUL:
        steps += 1
        _, _, terminated, truncated, info = env.step(pol.act(env))
        if terminated or truncated:
            won = bool(info.get("battle_won", False))
            break
    return [r | {"scenario": scen, "seed": seed, "won": won} for r in pol.rows]


def run() -> None:
    jobs = [(scen, PB.FIRST_SEED[scen] + i) for scen in SCENS for i in range(N)]
    with Pool(9) as pool:
        rows = sum(pool.map(play, jobs, chunksize=2), [])
    pd.DataFrame(rows).to_parquet(OUT)
    print(len(rows), "decisions")


def analyse() -> None:
    from analyse_phase_b import BEST_FIXED
    from sklearn.tree import DecisionTreeClassifier

    d = pd.read_parquet(OUT)
    o = pd.read_parquet(A.OUT / "oracle_decisions.parquet")
    cols = [
        "t",
        "distance",
        "in_reach_share_blue",
        "in_reach_share_red",
        "depth_red",
        "hp_balance",
        "unit_balance",
        "wounded_share_blue",
        "wounded_share_red",
    ]
    lines = [
        "# Does the model planner withdraw where the oracle does?",
        "",
        "Script: `experiments/planning/model_withdraw.py`. Phase B's primary planner replayed "
        "(deterministic) on",
        f"Phase B's first {N} seeds of {' and '.join(SCENS)}; the oracle's decisions from "
        "`19_oracle_withdraw.md`.",
        "Exploratory.",
        "",
    ]
    for scen in SCENS:
        dm, om = d[d.scenario == scen], o[o.scenario == scen]
        fd, fo = features(dm), features(om)
        wd_m, wd_o = dm.choice.str.startswith("withdraw"), om.choice.str.startswith("withdraw")
        comp = pd.DataFrame(
            {
                "planner: withdraw": fd[wd_m].median(),
                "planner: other": fd[~wd_m].median(),
                "oracle: withdraw": fo[wd_o].median(),
                "oracle: other": fo[~wd_o].median(),
            }
        ).loc[cols]
        # the oracle's rule applied to the planner's states
        tree = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=40, class_weight="balanced", random_state=0
        )
        tree.fit(fo[cols].to_numpy(), wd_o)
        rule = tree.predict(fd[cols].to_numpy()).astype(bool)
        agree = pd.crosstab(
            pd.Series(rule, name="oracle's rule says withdraw"),
            pd.Series(wd_m.to_numpy(), name="planner withdraws"),
            normalize="index",
        )
        early = dm[dm.t <= 4].groupby("seed")["choice"].apply(lambda c: c.str.startswith("withdraw").any())
        pb = pd.read_csv(A.OUT / f"phase_b_{scen}.csv")
        W = pb.pivot(index="seed", columns="policy", values="won").astype(float)
        bf = BEST_FIXED[scen]
        gain = (W["model:h3"] - W[bf]).reindex(early.index)
        g = pd.DataFrame(
            {
                "battles": early.groupby(early).size(),
                "planner won": W["model:h3"].reindex(early.index).groupby(early).mean(),
                f"{bf.removeprefix('fixed:')} won": W[bf].reindex(early.index).groupby(early).mean(),
                "planner − fixed": gain.groupby(early).mean(),
            }
        )
        g.index = g.index.map({True: "withdrew in the first 4 s", False: "did not"})
        lines += [
            f"## {scen}",
            "",
            f"Withdraw share: planner {wd_m.mean():.3f} of re-plans, oracle {wd_o.mean():.3f}. "
            "Predicted gap best − second when the planner withdraws: median "
            f"{dm.loc[wd_m, 'gap'].median():.4f} "
            f"(all re-plans {dm['gap'].median():.4f}).",
            "",
            "Median state at withdraw and other decisions:",
            "",
            md(comp.round(3)),
            "",
            "The oracle's depth-3 rule (fitted on the oracle's decisions) applied to the planner's states:",
            "",
            md(agree.round(3)),
            "",
            "Battles by whether the planner withdrew in the first 4 s (wins in Phase B):",
            "",
            md(g.round(3)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["run", "analyse"])
    {"run": run, "analyse": analyse}[parser.parse_args().what]()


if __name__ == "__main__":
    main()
