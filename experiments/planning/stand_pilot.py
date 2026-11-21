"""Pilot for the hold-position intent `stand` (docs/planning/17_stand_pilot_plan.md). Simulator only.

uv run experiments/planning/stand_pilot.py p1        twin check (counterfactual branches, S1 and S2)
uv run experiments/planning/stand_pilot.py p2        fixed stand intents on Phase A's seeds
uv run experiments/planning/stand_pilot.py check     five-candidate oracle on 10 seeds: reproduces Phase A?
uv run experiments/planning/stand_pilot.py p3        seven-candidate exact oracle, 100 Phase A seeds
uv run experiments/planning/stand_pilot.py report    -> docs/planning/18_stand_pilot.md
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
from sindy_wm.envs.intents import IntentSchedule  # noqa: E402
from sindy_wm.envs.policies import FIRST_TARGET  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "planning" / "18_stand_pilot.md"
P1_SEEDS = {"S1": 19_600_000, "S2": 19_610_000}
BRANCHES = (("advance", "nearest"), ("advance", "focus"), ("stand", "nearest"), ("stand", "focus"))
MAIN = ("S1", "S2", "S5")
N_BOOT = 4000


def _state(game, hp0) -> tuple:
    bh = sum(u.hp + u.shield for u in game.agents.values()) / hp0["blue"]
    rh = sum(u.hp + u.shield for u in game.enemies.values()) / hp0["red"]
    return round(bh, 9), round(rh, 9), len(game.agents), len(game.enemies)


def play_branch(job) -> dict:
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
    game = env.unwrapped
    hp0 = {
        "blue": sum(u.max_hp + u.max_shield for u in game.agents.values()),
        "red": sum(u.max_hp + u.max_shield for u in game.enemies.values()),
    }
    pol = IntentSchedule(
        branch_intent=(mv, tg), branch_seconds=(0.5, 8.0), branch_hold_seconds=4.0, withdraw_radius=wd
    )
    pol.reset(seed)
    k_branch = round(pol._branch_at / env.seconds_per_step)
    out = {
        "scenario": scen,
        "seed": seed,
        "branch": f"{mv}/{tg}",
        "contact": None,
        "at_branch": None,
        "at_4s": None,
    }
    won, step = False, 0
    while step < max_s * 16 // A.STEP_MUL:
        if step == k_branch:
            masks = game.get_avail_actions()
            out["contact"] = any(m[FIRST_TARGET:].any() for i, m in enumerate(masks) if i in game.agents)
            out["at_branch"] = _state(game, hp0)
        if step == k_branch + round(4.0 / env.seconds_per_step):
            out["at_4s"] = _state(game, hp0)
        actions = pol.act(env)
        _, _, terminated, truncated, info = env.step(actions)
        step += 1
        if terminated or truncated:
            won = bool(info.get("battle_won", False))
            break
    end = _state(game, hp0)
    out["at_4s"] = out["at_4s"] or end
    out |= {"won": won, "end": end, "ended_before_branch": out["at_branch"] is None}
    return out


def p1() -> None:
    jobs = [(scen, first + i, b) for scen, first in P1_SEEDS.items() for i in range(200) for b in BRANCHES]
    with Pool(9) as pool:
        rows = pool.map(play_branch, jobs, chunksize=4)
    pd.DataFrame(rows).to_csv(A.OUT / "stand_p1.csv", index=False)
    print(len(rows), "branch battles")


def p2() -> None:
    for scen in A.SCENARIOS:
        A.run(scen, ["fixed:stand/nearest", "fixed:stand/focus"], 400, 9, tag="_stand")


def check() -> None:
    for scen in ("S1", "S3"):
        A.run(scen, ["oracle"], 10, 9, tag="_standcheck")


def p3() -> None:
    for scen in ("S1", "S2", "S3", "S5"):
        A.run(scen, ["oracle-stand"], 100, 9, tag="_stand")


def paired(a, b, rng):
    x = a - b
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def md(t) -> str:
    return "```\n" + t.to_string() + "\n```"


def report() -> None:
    rng = np.random.default_rng(0)
    from analyse_phase_b import BEST_FIXED

    lines = [
        "# Hold-position pilot: results",
        "",
        "Plan and decision rule: `17_stand_pilot_plan.md`. Script: "
        "`experiments/planning/stand_pilot.py`. Simulator only.",
        "",
    ]
    # P1
    d = pd.read_csv(A.OUT / "stand_p1.csv")
    d = d[~d["ended_before_branch"]]
    rows = []
    for scen, g in d.groupby("scenario"):
        w = {b: g[g["branch"] == b].set_index("seed") for b in g["branch"].unique()}
        seeds = sorted(set.intersection(*[set(v.index) for v in w.values()]))
        contact = w["advance/nearest"].loc[seeds, "contact"].astype(bool).to_numpy()
        for tg in ("nearest", "focus"):
            a, s = w[f"advance/{tg}"].loc[seeds], w[f"stand/{tg}"].loc[seeds]
            diff4 = (a["at_4s"] != s["at_4s"]).to_numpy()
            win = (a["won"] == s["won"]).to_numpy()
            for lab, m in (("in contact at branch", contact), ("not in contact", ~contact)):
                rows.append(
                    {
                        "scenario": scen,
                        "pair": f"stand/{tg} vs advance/{tg}",
                        "at branch": lab,
                        "states": int(m.sum()),
                        "different 4 s battle": diff4[m].mean(),
                        "same winner": win[m].mean(),
                        "stand won": s["won"].to_numpy()[m].mean(),
                        "advance won": a["won"].to_numpy()[m].mean(),
                    }
                )
    p1t = pd.DataFrame(rows).set_index(["scenario", "pair", "at branch"])
    lines += ["## P1: is stand distinct? (counterfactual branches)", "", md(p1t.round(3)), ""]
    # P2
    rows = []
    for scen in A.SCENARIOS:
        base = pd.read_csv(A.OUT / f"headroom_{scen}.csv")
        new = pd.read_csv(A.OUT / f"headroom_{scen}_stand.csv")
        W = (
            pd.concat([base, new])
            .pivot_table(index="seed", columns="policy", values="won", aggfunc="first")
            .astype(float)
        )
        bf = BEST_FIXED[scen]
        for p in ("fixed:stand/nearest", "fixed:stand/focus"):
            s = W[[p, bf]].dropna()
            mean, lo, hi = paired(s[p].to_numpy(), s[bf].to_numpy(), rng)
            rows.append(
                {
                    "scenario": scen,
                    "intent": p.removeprefix("fixed:"),
                    "won": s[p].mean(),
                    "best fixed (Phase A)": bf.removeprefix("fixed:"),
                    "its won": s[bf].mean(),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "n": len(s),
                }
            )
    lines += [
        "## P2: fixed stand intents against Phase A's best fixed intent (won, 400 paired seeds)",
        "",
        md(pd.DataFrame(rows).set_index(["scenario", "intent"]).round(3)),
        "",
    ]
    # determinism check
    chk = []
    for scen in ("S1", "S3"):
        base = pd.read_csv(A.OUT / f"headroom_{scen}.csv")
        new = pd.read_csv(A.OUT / f"headroom_{scen}_standcheck.csv")
        m = new.merge(base[base.policy == "oracle"], on=["seed", "policy"], suffixes=("", "_a"))
        chk.append(
            {
                "scenario": scen,
                "seeds": len(m),
                "same won": (m["won"] == m["won_a"]).mean(),
                "max |margin diff|": float((m["margin"] - m["margin_a"]).abs().max()),
            }
        )
    lines += [
        "## Determinism check (five-candidate oracle replayed on 10 seeds)",
        "",
        md(pd.DataFrame(chk)),
        "",
    ]
    # P3
    rows = []
    for scen in ("S1", "S2", "S3", "S5"):
        base = pd.read_csv(A.OUT / f"headroom_{scen}.csv")
        new = pd.read_csv(A.OUT / f"headroom_{scen}_stand.csv")
        if "oracle-stand" not in set(new["policy"]):
            continue
        W = (
            pd.concat([base, new])
            .pivot_table(index="seed", columns="policy", values="won", aggfunc="first")
            .astype(float)
        )
        s = W[["oracle-stand", "oracle"]].dropna()
        mean, lo, hi = paired(s["oracle-stand"].to_numpy(), s["oracle"].to_numpy(), rng)
        o7 = new[new.policy == "oracle-stand"]
        share = {c.removeprefix("share_"): o7[c].fillna(0).mean() for c in o7 if c.startswith("share_stand")}
        rows.append(
            {
                "scenario": scen,
                "oracle, 7 candidates": s["oracle-stand"].mean(),
                "oracle, 5": s["oracle"].mean(),
                "difference": mean,
                "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                "n": len(s),
                **share,
            }
        )
    if rows:
        lines += [
            "## P3: exact oracle with seven candidates against five (won, paired)",
            "",
            md(pd.DataFrame(rows).set_index("scenario").round(3)),
            "",
        ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["p1", "p2", "check", "p3", "report"])
    {"p1": p1, "p2": p2, "check": check, "p3": p3, "report": report}[parser.parse_args().what]()


if __name__ == "__main__":
    main()
