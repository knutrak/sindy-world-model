"""The oracle with the planners' settings, and an exploitation check (approved plan:
docs/planning/38_oracle_setup_plan.md).

    uv run experiments/planning/oracle38.py time               5 battles per oracle (S4, smoke seeds), timed
    uv run experiments/planning/oracle38.py exploit            part 2: planners replayed with a trace
    uv run experiments/planning/oracle38.py run S4 --n 400     part 1: oracle7-h3 and oracle7-h8 on 34's seeds
    uv run experiments/planning/oracle38.py analyse            -> docs/planning/39_oracle_setup.md
"""

import argparse
import os
import sys
import time
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
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

ORACLES = ["oracle7-h3", "oracle7-h8"]
MAIN = ("S2", "S4", "S5")
PLANNERS = ["model:h3", "musigma:h3", "mlp:h3"]
TRACE = A.OUT / "trace38.csv"
DOC = PROJECT_ROOT / "docs" / "planning" / "39_oracle_setup.md"
N_BOOT = 4000


def time_it():
    for name in ORACLES:
        t0 = time.time()
        PB.run("S4", [name], 5, 5, C34.SMOKE, "_smoke38", prefix="closed34_")
        print(f"{name}: 5 battles in parallel, {time.time() - t0:.0f} s", flush=True)
    d = pd.read_csv(A.OUT / "closed34_S4_smoke38.csv")
    print(d.groupby("policy")["wall_s"].describe().round(1).to_string())


def play_trace(job) -> list[dict]:
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
    policy = PB.planner(name, scen, wd)
    policy.trace = True
    policy.reset(seed)
    won, steps = False, 0
    while steps < max_s * 16 // A.STEP_MUL:
        steps += 1
        _, _, term, trunc, info = env.step(policy.act(env))
        if term or trunc:
            won = bool(info.get("battle_won", False))
            break
    log = getattr(policy, "trace_log", [])
    rows = []
    for a, b in zip(log, log[1:], strict=False):
        if b["step"] - a["step"] != 8:
            continue
        p = np.array(a["pred1"])
        rows.append(
            {
                "scenario": scen,
                "seed": seed,
                "policy": name,
                "won": won,
                "step": a["step"],
                "pred_chosen": p[a["chosen"]],
                "pred_mean": p.mean(),
                "realised": b["now"] - a["now"],
            }
        )
    if not rows:
        rows.append({"scenario": scen, "seed": seed, "policy": name, "won": won})
    return rows


def exploit():
    jobs = [(scen, C34.FIRST[scen] + i, p) for scen in MAIN for p in PLANNERS for i in range(100)]
    with Pool(8) as pool:
        out = pool.map(play_trace, jobs, chunksize=1)
    t = pd.DataFrame([r for rows in out for r in rows])
    t.to_csv(TRACE, index=False)
    # deterministic replay: wins equal the closed-loop runs
    mism = 0
    for scen in MAIN:
        d = pd.read_csv(A.OUT / f"closed34_{scen}.csv").set_index(["seed", "policy"])["won"]
        w = t[t.scenario == scen].groupby(["seed", "policy"])["won"].first()
        mism += int((w != d.loc[w.index]).sum())
    print("replay mismatches (should be 0):", mism)


def paired(d, a, b, rng):
    return C34.paired(d, a, b, rng)


def boot(x, grp, rng):
    u, inv = np.unique(grp, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    m = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        m.append((w * sums).sum() / (w * cnt).sum())
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def analyse():
    rng = np.random.default_rng(0)
    rows, wins = [], []
    for scen in MAIN:
        d = pd.read_csv(A.OUT / f"closed34_{scen}.csv")
        wins.append(d.groupby("policy")["won"].mean().rename(scen))
        for label, a, b in (
            ("O1", "oracle7-h3", C34.BEST_KITE[scen]),
            ("O2", "oracle7-h8", C34.BEST_KITE[scen]),
            ("O3", "oracle7-h3", "musigma:h3"),
        ):
            if a not in set(d.policy):
                continue
            mean, lo, hi, read, n = paired(d, a, b, rng)
            rows.append(
                {
                    "scenario": scen,
                    "rule": label,
                    "policy": a,
                    "against": b.removeprefix("fixed:"),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": read,
                    "n": n,
                }
            )
    t = pd.DataFrame(rows)
    o1 = t[t.rule == "O1"]["read-out"]
    if len(o1) == 3 and not (o1 == "beats").any():
        verdict = (
            "the setup claim holds: even a perfect model with the planners' settings does not beat fixed kite"
        )
    elif (o1 == "beats").any() and not (o1 == "loses").any():
        verdict = (
            "the setup claim is wrong for these settings: a perfect model would beat fixed kite, so model "
            "accuracy matters; 37's wording and the supervisor doc are to be corrected"
        )
    else:
        verdict = "mixed: reported per scenario, wording to be narrowed"
    ex = []
    if TRACE.exists():
        tr = pd.read_csv(TRACE).dropna(subset=["realised"])
        for (scen, pol), g in tr.groupby(["scenario", "policy"]):
            m, lo, hi = boot((g.pred_chosen - g.realised).to_numpy(), g.seed.to_numpy(), rng)
            u, ulo, uhi = boot((g.pred_chosen - g.pred_mean).to_numpy(), g.seed.to_numpy(), rng)
            ex.append(
                {
                    "scenario": scen,
                    "planner": pol,
                    "re-plans": len(g),
                    "optimism (pred − real)": m,
                    "95%": f"[{lo:+.4f}, {hi:+.4f}]",
                    "predicted uplift (chosen − mean)": u,
                    "uplift 95%": f"[{ulo:+.4f}, {uhi:+.4f}]",
                }
            )
    pd.set_option("display.width", 250)
    lines = [
        "# The oracle with the planners' settings, and exploitation: results",
        "",
        "Plan: `38_oracle_setup_plan.md` (approved 22 Nov). Script: `experiments/planning/oracle38.py`.",
        "Seeds: `34`'s (21 x00 000 +), paired with every policy played there.",
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
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**Interpretation fixed in the plan: {verdict}.**",
        "",
        "## Exploitation check (100 seeds per scenario; margin gain over the next 1 s, team health units)",
        "",
        "```",
        pd.DataFrame(ex).round(4).to_string(index=False) if ex else "(not run)",
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["time", "exploit", "run", "analyse"])
    p.add_argument("scenario", nargs="?")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.what == "run":
        PB.run(a.scenario, ORACLES, a.n, a.workers, C34.FIRST[a.scenario], prefix="closed34_")
        return
    {"time": time_it, "exploit": exploit, "analyse": analyse}[a.what]()


if __name__ == "__main__":
    main()
