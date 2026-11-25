# ruff: noqa: B023, E402, E501, UP031
"""Log of one SMAClite MMM battle: what the medivacs do."""

import warnings

warnings.filterwarnings("ignore")
import gymnasium as gym
import numpy as np
import smaclite  # noqa: F401  (registers the environments)

env = gym.make("smaclite/MMM-v0")
env.reset(seed=1)
g = env.unwrapped


def med(d):
    return [u for u in d.values() if u.type.stats.name == "medivac"][0]


m, b = med(g.enemies), med(g.agents)


def heal(u):
    return u.target is not None and u.target.type.stats.name


def run():
    rows = []
    for t in range(600):
        acts = []
        for i, mask in enumerate(g.get_avail_actions()):
            av = np.flatnonzero(mask)
            u = g.agents.get(i)
            if u is None:
                acts.append(0)
                continue
            tg = av[av >= 6]
            if u is b:  # heal the most hurt ally in range, else follow
                hurt = [k for k in tg if g.agents[k - 6].hp < g.agents[k - 6].max_hp]
                acts.append(int(min(hurt, key=lambda k: g.agents[k - 6].hp)) if hurt else 1)
                continue
            acts.append(
                int(min(tg, key=lambda k: np.linalg.norm(g.enemies[k - 6].pos - u.pos)))
                if len(tg)
                else (4 if mask[4] else 1)
            )
        tgt = m.target.type.stats.name if m.target is not None else None
        hits = sum(1 for u in g.agents.values() if u.target is m)
        sum(u.hp for u in g.enemies.values())
        sum(u.hp for u in g.agents.values())
        _, _, term, tr, _ = env.step(acts)
        rows.append(
            (
                t,
                m.hp,
                m.energy,
                tgt,
                b.hp,
                b.energy,
                len(g.enemies),
                len(g.agents),
                hits,
                min((np.linalg.norm(m.pos - u.pos) for u in g.agents.values()), default=0),
                b.target.type.stats.name if b.target is not None else None,
                sum(1 for u in g.enemies.values() if u.target is b),
            )
        )
        if term or tr:
            break
    return rows


R = run()
print("step(0.5s?)", "seconds_per_step", getattr(g, "seconds_per_step", None))
for r in R[::6]:
    print(
        "t%3d red med hp %3.0f en %4.1f tgt %-9s | blue med hp %3.0f en %4.1f tgt %-9s | red %2d blue %2d | aimed at red med %d, at blue med %d | red med to nearest blue %.1f"
        % (r[0], r[1], r[2], r[3], r[4], r[5], r[10], r[6], r[7], r[8], r[11], r[9])
    )
print(R[-1])
