"""Where do the model planners disagree with a perfect model? (docs/planning/44_disagreement_plan.md)

uv run experiments/planning/disagree44.py run        oracle5-h3 replays, 100 seeds per scenario, models scored
uv run experiments/planning/disagree44.py analyse    -> docs/planning/45_disagreement.md
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
import closed34 as C34  # noqa: E402
import headroom as A  # noqa: E402
import phase_b as PB  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.planner import CANDIDATES  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

OUT = A.OUT / "disagree44.csv"
DOC = PROJECT_ROOT / "docs" / "planning" / "45_disagreement.md"
N = 100
NAMES = ["/".join(c) for c in CANDIDATES]
TWIN = np.array(
    [0, 1, 0, 1, 2]
)  # advance/nearest, advance/focus, hold/nearest, hold/focus, withdraw -> 3 choices


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
    oracle = A.make_policy("oracle5-h3", wd)
    oracle.reset(seed)
    models = {"musigma": PB.planner("musigma5:h3", scen, wd), "frozen": PB.planner("model:h3", scen, wd)}
    for m in models.values():
        m.reset(seed)
        m.live_state(env)
    rows, won, steps, first_contact = [], False, 0, None
    while steps < max_s * 16 // A.STEP_MUL:
        n_before = len(oracle.score_log)
        actions = oracle.act(env)
        if len(oracle.score_log) > n_before:
            row = models["musigma"].live_state(env)
            reach_b = float(sum(v for k, v in row.items() if str(k).startswith("reach_blue_")))
            if first_contact is None and reach_b > 0:
                first_contact = steps
            red_sd = models["musigma"].live_margins()["sd0"]
            n = len(models["musigma"].model.L.types)
            r = {
                "scenario": scen,
                "seed": seed,
                "t": steps * A.STEP_MUL / 16,
                "contact": reach_b > 0,
                "since_contact": None if first_contact is None else (steps - first_contact) * A.STEP_MUL / 16,
                "red_sd": float(np.nanmean(red_sd[n:])),
                "oracle": list(oracle.score_log[-1]),
            }
            for name, m in models.items():
                r[name] = list(m.predict(env))
            rows.append(r)
        steps += 1
        _, _, term, trunc, info = env.step(actions)
        if term or trunc:
            won = bool(info.get("battle_won", False))
            break
    for r in rows:
        r["won"] = won
    return rows or [{"scenario": scen, "seed": seed, "won": won}]


def run():
    jobs = [(scen, C34.FIRST[scen] + i) for scen in C34.FIRST for i in range(N)]
    with Pool(8) as pool:
        out = pool.map(play, jobs, chunksize=2)
    t = pd.DataFrame([r for rows in out for r in rows])
    t.to_pickle(OUT.with_suffix(".pkl"))
    mism = 0
    for scen in C34.FIRST:
        d = pd.read_csv(A.OUT / f"closed34_{scen}.csv")
        d = d[d.policy == "oracle5-h3"].set_index("seed")["won"]
        w = t[t.scenario == scen].groupby("seed")["won"].first()
        mism += int((w != d.loc[w.index]).sum())
    print(f"{len(t)} re-plans; replay mismatches against 43 (should be 0): {mism}")


def analyse():
    t = pd.read_pickle(OUT.with_suffix(".pkl")).dropna(subset=["oracle"])
    S_or = np.array(t.oracle.tolist())
    best = S_or.argmax(1)
    top2 = np.sort(S_or, 1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) > 20
    contact = t.contact.to_numpy(bool)
    sc = t.since_contact.to_numpy(float)
    phase = np.where(np.isnan(sc), "before contact", np.where(sc <= 3, "first 3 s of contact", "later"))
    t = t.assign(phase=phase, decisive=decisive)
    t["red_sd_tercile"] = pd.qcut(t.red_sd.rank(method="first"), 3, labels=["compact", "middle", "spread"])
    rows, conf = [], {}
    for name in ("musigma", "frozen"):
        P = np.array(t[name].tolist()).argmax(1)
        same = np.where(contact, TWIN[P] == TWIN[best], P == best)
        regret = S_or[np.arange(len(S_or)), best] - S_or[np.arange(len(S_or)), P]
        t[f"agree_{name}"], t[f"regret_{name}"] = same, regret
        conf[name] = pd.crosstab(
            pd.Series([NAMES[i] for i in best], name="oracle"), pd.Series([NAMES[i] for i in P], name=name)
        )
    for by in (None, "scenario", "phase", "red_sd_tercile", "decisive"):
        g = [("all", t)] if by is None else list(t.groupby(by, observed=True))
        for key, s in g:
            rows.append(
                {
                    "by": by or "all",
                    "group": key,
                    "re-plans": len(s),
                    **{f"agree {n}": s[f"agree_{n}"].mean() for n in ("musigma", "frozen")},
                    **{f"regret hp {n}": s[f"regret_{n}"].mean() for n in ("musigma", "frozen")},
                }
            )
    oracle_use = pd.Series([NAMES[i] for i in best]).value_counts(normalize=True)
    pd.set_option("display.width", 250)
    lines = [
        "# Where the model planners disagree with a perfect model: results",
        "",
        "Plan: `44_disagreement_plan.md` (exploratory, read-outs fixed before the run).",
        "Script: `experiments/planning/disagree44.py`. Oracle `oracle5-h3` replays, 100 seeds per scenario;",
        "agreement merges hold/advance twins in contact; regret in hp under the oracle's true 3 s scores.",
        "",
        "```",
        pd.DataFrame(rows).round(3).to_string(index=False),
        "```",
        "",
        "## The oracle's picks",
        "",
        "```",
        oracle_use.round(3).to_string(),
        "```",
        "",
    ]
    for name, c in conf.items():
        lines += [
            f"## Confusion: oracle pick (rows) vs {name} pick (columns)",
            "",
            "```",
            c.to_string(),
            "```",
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "analyse"])
    {"run": run, "analyse": analyse}[p.parse_args().what]()


if __name__ == "__main__":
    main()
