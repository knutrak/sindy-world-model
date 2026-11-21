"""When does the exact oracle withdraw? (exploratory, after docs/planning/18; simulator only)

Replays Phase A's exact oracle (re-plan 1 s, commit 2 s, horizon 8 s, five candidates; deterministic, so the
battles are Phase A's) on Phase A's first 100 seeds of S1, S2, S3, S5, logging at every re-plan the
per-type state
(`states.typed`, reach capped at 6), the wounded counts (`states.compartments`), the five candidate scores
and the
choice. Then compares the states where it withdraws with those where it does not.

    uv run experiments/planning/oracle_withdraw.py run       ->
    data/processed/planning/oracle_decisions.parquet
    uv run --extra explore experiments/planning/oracle_withdraw.py analyse  ->
    docs/planning/19_oracle_withdraw.md
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

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.planner import OraclePlannerPolicy  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402
from sindy_wm.states.compartments import compartments, unit_health  # noqa: E402
from sindy_wm.states.typed import typed_states  # noqa: E402

OUT = A.OUT / "oracle_decisions.parquet"
DOC = PROJECT_ROOT / "docs" / "planning" / "19_oracle_withdraw.md"
SCENS = ("S1", "S2", "S3", "S5")
N = 100
TYPES = ("MARINE", "MARAUDER")


class LoggingOracle(OraclePlannerPolicy):
    """The exact oracle; at each re-plan it also records the state, the scores and the choice (read-only)."""

    def reset(self, seed=None):
        super().reset(seed)
        self.log, self._logger, self._start = [], None, None

    def act(self, env):
        game = env.unwrapped
        if self._logger is None:
            self._logger = SnapshotLogger(env)
            s0 = self._logger.start_episode().assign(episode_idx=0)
            self._start = s0.assign(h=unit_health(s0)).set_index(["episode_idx", "team", "unit_id"])["h"]
        replan = (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        )
        if replan:
            snap = self._logger.snapshot(step=self._step).assign(episode_idx=0)
            row = typed_states(snap.drop(columns="episode_idx"), TYPES, max_reach=6.0).iloc[0].to_dict()
            row |= compartments(snap, TYPES, start=self._start).iloc[0].to_dict()
            scores = [self._lookahead(env, intent) for intent in self.candidates]
            best = int(np.argmax(scores))
            row |= {f"score_{m}/{t}": v for (m, t), v in zip(self.candidates, scores, strict=True)}
            row |= {"t": self._step * env.seconds_per_step, "choice": "/".join(self.candidates[best])}
            self.log.append(row)
            from sindy_wm.envs.intents import IntentPolicy

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
    pol = LoggingOracle(commit_seconds=2.0, horizon_seconds=8.0, withdraw_radius=wd)
    pol.reset(seed)
    steps, won = 0, False
    while steps < max_s * 16 // A.STEP_MUL:
        steps += 1
        _, _, terminated, truncated, info = env.step(pol.act(env))
        if terminated or truncated:
            won = bool(info.get("battle_won", False))
            break
    return [
        r
        | {
            "scenario": scen,
            "seed": seed,
            "won": won,
            "battle_s": steps * A.STEP_MUL / 16,
            "start_health_blue": s.start_health[0],
            "start_health_red": s.start_health[1],
        }
        for r in pol.log
    ]


def run() -> None:
    jobs = [(scen, A.SCENARIOS[scen][4] + i) for scen in SCENS for i in range(N)]
    with Pool(9) as pool:
        rows = sum(pool.map(play, jobs, chunksize=1), [])
    pd.DataFrame(rows).to_parquet(OUT)
    print(len(rows), "decisions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["run", "analyse", "test"])
    a = parser.parse_args()
    if a.what == "test":
        rows = play(("S3", A.SCENARIOS["S3"][4]))
        base = pd.read_csv(A.OUT / "headroom_S3.csv")
        ref = base[(base.seed == A.SCENARIOS["S3"][4]) & (base.policy == "oracle")].iloc[0]
        d = pd.DataFrame(rows)
        print(d[["t", "choice", "distance", "reach_blue_marauder", "reach_red_marine"]].to_string())
        print(
            "won",
            rows[0]["won"],
            "Phase A won",
            ref["won"],
            "| withdraw share",
            (d.choice.str.startswith("withdraw")).mean(),
            "Phase A share_withdraw/nearest",
            ref["share_withdraw/nearest"],
        )
        return
    {"run": run, "analyse": analyse}[a.what]()


HP = {"marine": 45.0, "marauder": 125.0}


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Team-level features of each decision state (hp-weighted health, units, reach shares, geometry)."""
    f = pd.DataFrame(index=d.index)
    for side in ("blue", "red"):
        hp = sum(d.get(f"health_{side}_{t}", 0).fillna(0) * 10 * HP[t] for t in HP)
        full = sum(d.get(f"alive_{side}_{t}", 0).fillna(0) * 10 * HP[t] for t in HP)
        A = sum(d.get(f"alive_{side}_{t}", 0).fillna(0) * 10 for t in HP)
        R = sum(d.get(f"reach_{side}_{t}", 0).fillna(0) * 10 for t in HP)
        W = sum(d.get(f"W_{side}_{t}", 0).fillna(0) * 10 for t in HP)
        f[f"hp_{side}"], f[f"units_{side}"] = hp, A
        f[f"in_reach_share_{side}"] = (R / A.replace(0, np.nan)).fillna(0)
        f[f"wounded_share_{side}"] = (W / A.replace(0, np.nan)).fillna(0)
        f[f"mean_health_{side}"] = (hp / full.replace(0, np.nan)).fillna(0)
    f["hp_balance"] = f["hp_blue"] / (f["hp_blue"] + f["hp_red"])
    f["unit_balance"] = f["units_blue"] / (f["units_blue"] + f["units_red"])
    f["reach_asymmetry"] = f["in_reach_share_blue"] - f["in_reach_share_red"]
    for c in ("distance", "front_distance", "depth_blue", "depth_red", "width_blue", "width_red", "t"):
        f[c] = d[c]
    return f


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def analyse() -> None:
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    d = pd.read_parquet(OUT)
    f = features(d)
    wd = d["choice"].str.startswith("withdraw")
    cols = [
        "hp_balance",
        "unit_balance",
        "reach_asymmetry",
        "in_reach_share_blue",
        "in_reach_share_red",
        "mean_health_blue",
        "mean_health_red",
        "wounded_share_blue",
        "wounded_share_red",
        "distance",
        "front_distance",
        "depth_red",
        "width_red",
        "t",
    ]
    lines = [
        "# When does the exact oracle withdraw?",
        "",
        "Script: `experiments/planning/oracle_withdraw.py`. Phase A's exact oracle replayed "
        "(deterministic) on",
        f"Phase A's first {N} seeds of {', '.join(SCENS)}, with the state at each re-plan (1 s). "
        "Exploratory.",
        "Reach shares: share of a side's alive units with an enemy in reach (capped at 6); reach "
        "asymmetry = Blue's",
        "share − Red's. Health balance = Blue hp / (Blue + Red hp). Wounded: below half of own start health.",
        "",
    ]
    summary = []
    for scen in SCENS:
        m = d["scenario"] == scen
        g, y = f[m], wd[m]
        summary.append(
            {
                "scenario": scen,
                "decisions": int(m.sum()),
                "withdraw share": y.mean(),
                "battles with any withdraw": d[m]
                .groupby("seed")["choice"]
                .apply(lambda c: c.str.startswith("withdraw").any())
                .mean(),
                "won": d[m].groupby("seed")["won"].first().mean(),
            }
        )
        # compare states
        comp = pd.DataFrame(
            {
                "withdraw": g[y].median(),
                "other": g[~y].median(),
                "std. difference": (g[y].mean() - g[~y].mean()) / g.std().replace(0, np.nan),
            }
        )
        lines += [
            f"## {scen}",
            "",
            "Median state when the oracle withdraws vs when it does not:",
            "",
            md(comp.loc[cols].round(3)),
            "",
        ]
        # what kind of withdrawal: run lengths, scores
        runs = []
        for _, b in d[m].sort_values(["seed", "t"]).groupby("seed"):
            w = b["choice"].str.startswith("withdraw").to_numpy()
            k = 0
            for v in list(w) + [False]:
                if v:
                    k += 1
                elif k:
                    runs.append(k)
                    k = 0
        sc = [c for c in d if c.startswith("score_")]
        S = d.loc[m & wd, sc].to_numpy()
        wcol = sc.index("score_withdraw/nearest")
        others = np.delete(S, wcol, axis=1).max(1)
        gap = S[:, wcol] - others
        hpscale = (f.loc[m & wd, "hp_blue"] + f.loc[m & wd, "hp_red"]).to_numpy()
        lines += [
            f"Withdraw spells: {len(runs)}; length in re-plans (1 s): median {np.median(runs):.0f}, "
            f"share of length 1: {np.mean(np.array(runs) == 1):.0%}, longest {max(runs)}.",
            f"Score gap when it withdraws (withdraw − best other, hp): median {np.median(gap):.1f}; "
            f"as a share of both sides' hp: median {np.median(gap / hpscale):.3f}.",
            "",
        ]
        # a shallow tree, cross-validated by battle
        X = g[cols].to_numpy()
        tree = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=40, class_weight="balanced", random_state=0
        )
        acc = cross_val_score(
            tree, X, y, groups=d.loc[m, "seed"], cv=GroupKFold(5), scoring="balanced_accuracy"
        )
        tree.fit(X, y)
        lines += [
            f"Depth-3 tree, withdraw vs not: balanced accuracy {acc.mean():.3f} (5-fold by battle; 0.5 "
            "= chance).",
            "",
            "```",
            export_text(tree, feature_names=cols, decimals=2),
            "```",
            "",
        ]
    lines[7:7] = ["## Overview", "", md(pd.DataFrame(summary).set_index("scenario").round(3)), ""]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
