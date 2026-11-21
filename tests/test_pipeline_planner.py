"""The Phase B planner (docs/planning/07_phase_b_plan.md): its live per-type state equals the table built from
the battle's full snapshot record, and its predictions equal `Pipeline.rollout` on that table's row."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode_sampling import CloseBattleSampler, MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.planner import PipelinePlannerPolicy, typed_batch_from_row
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger
from sindy_wm.models.intent_model import SHAPE
from sindy_wm.paths import DATA_DIR
from sindy_wm.states.typed import typed_states

MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
CASES = [
    ("10m_vs_10m", CloseBattleSampler(), "marines", 7.0),
    ("mixed_m_mrd", MixedBattleSampler(), "mixed", 8.0),
]


def _battle(mapname, sampler, seed, steps=60):
    s = sampler.sample(seed, walkable_bounds(load_map_info(mapname)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        mapname, step_mul=2, spawn=s.spawn, unit_counts=None if types else s.unit_counts, unit_types=types
    )
    env.reset(seed=seed)
    return env


@pytest.mark.skipif(not MODELS.exists(), reason="frozen models not built")
@pytest.mark.parametrize("mapname,sampler,name,wd", CASES)
def test_live_state_and_predictions_match_the_recorded_table(mapname, sampler, name, wd):
    model = pickle.load(open(MODELS, "rb"))[name]
    env = _battle(mapname, sampler, seed=3)
    planner = PipelinePlannerPolicy(model, withdraw_radius=wd)
    planner.reset(3)
    record = SnapshotLogger(env)
    snaps = [record.start_episode()]
    driver = IntentSchedule(withdraw_radius=wd)
    driver.reset(3)
    live, preds = {}, {}
    for k in range(1, 60):
        _, _, term, trunc, _ = env.step(driver.act(env))
        snaps.append(record.snapshot(step=k))
        if term or trunc:
            break
        if k % 8 == 0 and env.unwrapped.agents and env.unwrapped.enemies:
            live[k] = planner.live_state(env)
            preds[k] = planner.predict(env)
    assert live, "no step with both sides alive"
    table = typed_states(pd.concat(snaps), planner.types, max_reach=model.max_reach)
    L = model.L
    cols = [*L.x_cols, *L.r_cols, *L.front_cols, *SHAPE]
    for k, row in live.items():
        np.testing.assert_allclose(row[cols].to_numpy(float), table.loc[k, cols].to_numpy(float), atol=1e-9)
        Y = L.team_view(model.rollout(typed_batch_from_row(table.loc[k], L, planner._U)))
        np.testing.assert_allclose(preds[k], Y[:, -1, 0] - Y[:, -1, 1], atol=1e-9)
