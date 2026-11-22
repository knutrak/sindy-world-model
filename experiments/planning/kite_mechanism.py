"""Why does kite win? A mechanism check (exploratory, docs/planning/31_kite_mechanism.md).

Plays fixed Blue intents against Red's scripted attack-move and against a Red that kites with the
same rule as Blue's kite (shoot when ready, step away from the nearest enemy while reloading;
attack-move otherwise). Per side it counts shots (a unit's weapon cooldown going up), the firing duty
cycle (shots x cooldown / unit-seconds alive), and after first contact how often a unit's weapon is
ready with no enemy inside its range (lost fire). Pilot seeds 19 7x0 000.

    uv run experiments/planning/kite_mechanism.py run --n 100
    uv run experiments/planning/kite_mechanism.py report
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from smaclite.env.units.unit_command import AttackMoveCommand, MoveCommand  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom as A  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.intents import KITE_READY, IntentPolicy  # noqa: E402
from sindy_wm.envs.policies import FIRST_TARGET  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

OUT = DATA_DIR / "processed" / "planning" / "kite_mechanism.csv"
DOC = PROJECT_ROOT / "docs" / "planning" / "31_kite_mechanism.md"
SEEDS = {"S1": 19_700_000, "S2": 19_710_000, "S3": 19_720_000}
CONFIGS = (
    ("advance/focus", "a-move"),
    ("advance/nearest", "a-move"),
    ("reach/focus", "a-move"),  # control: kite's firing rule without the step back
    ("kite/focus", "a-move"),
    ("kite/nearest", "a-move"),
    ("advance/focus", "kite"),
    ("kite/focus", "kite"),
)
STEP = 2.0  # smaclite MOVE_AMOUNT


class ReachPolicy(IntentPolicy):
    """Kite's firing rule without the step back: shoot an enemy inside own range (by the target rule), else
    advance."""

    def _kite_action(self, game, unit, mask, focus):
        offered = np.flatnonzero(mask[FIRST_TARGET:])
        if any(unit.has_within_attack_range(game.enemies[int(k)]) for k in offered):
            return self._stand_action(game, unit, mask, focus)
        return self._unit_action(game, unit, mask, focus)


def _in_range(unit, foes) -> bool:
    return any(unit.has_within_attack_range(f) for f in foes)


def red_kite(game, amove, bounds) -> None:
    """Red's kite: step away from the nearest Blue unit while reloading if one is in range, else
    attack-move."""
    xmin, xmax, ymin, ymax = bounds
    blues = list(game.agents.values())
    for e in game.enemies.values():
        if not blues or e.cooldown <= KITE_READY or not _in_range(e, blues):
            e.command = amove
            continue
        near = min(blues, key=lambda b: np.linalg.norm(b.pos - e.pos))
        d = e.pos - near.pos
        n = np.linalg.norm(d)
        goal = e.pos + (d / n if n > 0 else np.array([1.0, 0.0])) * STEP
        goal = np.clip(goal, [xmin + 0.5, ymin + 0.5], [xmax - 0.5, ymax - 0.5])
        e.command = MoveCommand(goal) if np.linalg.norm(goal - e.pos) > 0.5 else amove


def play(job) -> dict:
    scen, seed, blue, red = job
    mapname, sampler, wd, max_s, _ = A.SCENARIOS[scen]
    bounds = walkable_bounds(load_map_info(mapname))
    s = sampler.sample(seed, bounds)
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
    amove = next((e.command for e in game.enemies.values() if isinstance(e.command, AttackMoveCommand)), None)
    hp = lambda us: sum(u.hp + u.shield for u in us)  # noqa: E731
    hp0 = {"blue": hp(game.agents.values()), "red": hp(game.enemies.values())}
    mv, tg = blue.split("/")
    pol = (
        ReachPolicy("kite", tg, withdraw_radius=wd)
        if mv == "reach"
        else IntentPolicy(mv, tg, withdraw_radius=wd)
    )
    pol.reset(seed)
    dt = env.seconds_per_step
    acc = {
        f"{k}_{s}": 0.0
        for k in ("shots", "duty_num", "alive_s", "ready_idle", "post", "inrange")
        for s in "br"
    }
    contact, won, steps = False, False, 0
    while steps < max_s * 16 // A.STEP_MUL:
        steps += 1
        if red == "kite" and amove is not None:
            red_kite(game, amove, bounds)
        sides = {"b": (game.agents, game.enemies), "r": (game.enemies, game.agents)}
        before = {s: {i: u.cooldown for i, u in own.items()} for s, (own, _) in sides.items()}
        if not contact:
            contact = any(_in_range(u, game.enemies.values()) for u in game.agents.values())
        if contact:
            for s, (own, foe) in sides.items():
                for u in own.values():
                    r = _in_range(u, foe.values())
                    acc[f"post_{s}"] += 1
                    acc[f"inrange_{s}"] += r
                    acc[f"ready_idle_{s}"] += (u.cooldown == 0) and not r
        _, _, term, trunc, info = env.step(pol.act(env))
        for s, (own, _) in sides.items():
            for i, u in own.items():
                acc[f"alive_s_{s}"] += dt
                if u.cooldown > before[s].get(i, np.inf):
                    acc[f"shots_{s}"] += 1
                    acc[f"duty_num_{s}"] += u.max_cooldown
        if term or trunc:
            won = bool(info.get("battle_won", False))
            break
    out = {"scenario": scen, "seed": seed, "blue": blue, "red": red, "won": won, "seconds": steps * dt}
    for s, name in (("b", "blue"), ("r", "red")):
        out[f"{name} duty"] = acc[f"duty_num_{s}"] / max(acc[f"alive_s_{s}"], 1e-9)
        out[f"{name} ready, none in range"] = acc[f"ready_idle_{s}"] / max(acc[f"post_{s}"], 1)
        out[f"{name} in range"] = acc[f"inrange_{s}"] / max(acc[f"post_{s}"], 1)
        out[f"{name} shots"] = acc[f"shots_{s}"]
    out["blue hp lost"] = 1 - hp(game.agents.values()) / hp0["blue"]
    out["red hp lost"] = 1 - hp(game.enemies.values()) / hp0["red"]
    return out


def report() -> None:
    d = pd.read_csv(OUT)
    cols = [
        "won",
        "blue duty",
        "red duty",
        "blue ready, none in range",
        "red ready, none in range",
        "blue in range",
        "red in range",
        "red hp lost",
        "blue hp lost",
        "seconds",
    ]
    t = d.groupby(["scenario", "red", "blue"], sort=False)[cols].mean().round(3)
    lines = [
        "# Why kite wins: a mechanism check",
        "",
        "Script: `experiments/planning/kite_mechanism.py` (see its docstring). Red is SMAClite's scripted",
        "attack-move or the same kite rule; 'reach' = kite's firing rule (shoot only inside own range)",
        "without the step back. Means over battles.",
        "Duty = shots x weapon cooldown / unit-seconds alive (1 = fires at every chance);",
        "'ready, none in range' and 'in range' are shares of unit-steps after first contact.",
        "",
        "```",
        t.to_string(),
        "```",
        "",
        f"Battles per row: {d.groupby(['scenario', 'red', 'blue']).size().min()}.",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "report"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.what == "report":
        return report()
    jobs = [(sc, first + i, b, r) for sc, first in SEEDS.items() for b, r in CONFIGS for i in range(a.n)]
    with Pool(a.workers) as pool:
        rows = pool.map(play, jobs, chunksize=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    report()


if __name__ == "__main__":
    main()
