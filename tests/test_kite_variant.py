"""The kite variant (docs/planning/32_kite_refit_plan.md): with its kite terms switched off it rolls out
exactly as the frozen pipeline, and a four-weight schedule plays kite."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from sindy_wm.envs.episode_sampling import MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.models.intent_model import add_controls
from sindy_wm.models.kite import CTRL_K, KitePipeline, add_kite, control_vector_k
from sindy_wm.paths import DATA_DIR

MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
VAL = DATA_DIR / "processed" / "exp08_mixed_typed" / "val.parquet"


@pytest.mark.skipif(not (MODELS.exists() and VAL.exists()), reason="frozen models or exp08 val not built")
def test_kite_variant_without_kite_terms_equals_the_frozen_model():
    m = pickle.load(open(MODELS, "rb"))["mixed"]
    states = pd.read_parquet(VAL)
    states = add_kite(add_controls(states[states.index.get_level_values(0) < 5]))
    with np.errstate(invalid="ignore", divide="ignore"):
        B4 = m.L.batch(states, horizon=24, every=20)
        B5 = m.L.batch(states, horizon=24, every=20, ctrl=CTRL_K)
    k = copy.deepcopy(m)
    k.__class__ = KitePipeline
    k.P_team["gates"]["kite"] = k.P_team["gates"]["advance"]
    k.dist_names = [*m.dist_names, "close*kite"]
    k.dist_coef = np.append(m.dist_coef, 0.0)
    assert B5["U"].shape[-1] == 5
    np.testing.assert_allclose(k.rollout(B5), m.rollout(B4), rtol=0, atol=1e-12)


def test_control_vector_k():
    assert control_vector_k("kite", "focus").tolist() == [0, 0, 0, 1, 1]
    assert control_vector_k("advance", "nearest").tolist() == [1, 0, 0, 0, 0]


def test_four_weight_schedule_plays_kite_and_three_weights_never_do():
    s = MixedBattleSampler().sample(5, walkable_bounds(load_map_info("mixed_m_mrd")))
    seen = {}
    for w in ((0.5, 0.3, 0.2), (0.0, 0.0, 0.0, 1.0)):
        env = make_env("mixed_m_mrd", step_mul=2, spawn=s.spawn, unit_types=s.unit_types)
        env.reset(seed=5)
        pol = IntentSchedule(movement_weights=w, withdraw_radius=8.0)
        pol.reset(5)
        moves = set()
        for _ in range(40):
            _, _, term, trunc, _ = env.step(pol.act(env))
            moves.add(pol.setting()["movement"])
            if term or trunc:
                break
        seen[len(w)] = moves
    assert "kite" not in seen[3]
    assert seen[4] == {"kite"}
