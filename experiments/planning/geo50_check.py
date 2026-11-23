"""Sanity check for docs/planning/51: geo5 scores against the exact oracle's at the same states."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/planning")
import headroom as A  # noqa: E402
import phase_b as PB  # noqa: E402

from sindy_wm.envs.episode import _set_start_health  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402

agree, rows = [], []
for scen in ("S1", "S2"):
    for seed in range(
        21_000_000 + (0 if scen == "S1" else 100_000), 21_000_000 + (0 if scen == "S1" else 100_000) + 8
    ):
        mapname, sampler, wd, max_s, _ = A.SCENARIOS[scen]
        s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
        types = getattr(s, "unit_types", None)
        env = make_env(
            mapname, step_mul=2, spawn=s.spawn, unit_counts=None if types else s.unit_counts, unit_types=types
        )
        env.reset(seed=seed)
        _set_start_health(env, *s.start_health)
        geo = PB.planner("geo5:h3", scen, wd)
        geo.reset(seed)
        orc = A.make_policy("oracle5-h3", wd)
        orc.reset(seed)
        for step in range(200):
            g = env.unwrapped
            if not g.agents or not g.enemies:
                break
            a = orc.act(env)
            geo.act(env)  # same state; oracle plays
            if len(orc.score_log) and len(geo.score_log) and step % 8 == 0:
                o, m = np.array(orc.score_log[-1]), np.array(geo.score_log[-1])
                agree.append(o.argmax() == m.argmax())
                rows.append(
                    (
                        scen,
                        np.corrcoef(o, m)[0, 1] if o.std() > 0 and m.std() > 0 else np.nan,
                        o.argmax(),
                        m.argmax(),
                    )
                )
            _, _, t, tr, _ = env.step(a)
            if t or tr:
                break

d = pd.DataFrame(rows, columns=["scen", "corr", "oracle_pick", "geo_pick"])
print("agreement", np.mean(agree), "n", len(agree))
print(d.groupby("scen")["corr"].median())
print(pd.crosstab(d.oracle_pick, d.geo_pick))
