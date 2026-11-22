"""Reach as a state (docs/generalisation/07_share_weak_plan.md): with no share equations the variant rolls out
exactly as the frozen pipeline; the weak-form Gram fit recovers a known sparse equation."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sindy_wm.models.intent_model import add_controls
from sindy_wm.models.reach_state import ReachStatePipeline, gram, group_index, two_level_gram
from sindy_wm.paths import DATA_DIR

MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
VAL = DATA_DIR / "processed" / "exp08_mixed_typed" / "val.parquet"


@pytest.mark.skipif(not (MODELS.exists() and VAL.exists()), reason="frozen models or exp08 val not built")
def test_reach_state_without_equations_equals_the_frozen_model():
    m = pickle.load(open(MODELS, "rb"))["mixed"]
    states = pd.read_parquet(VAL)
    states = add_controls(states[states.index.get_level_values(0) < 5])
    with np.errstate(invalid="ignore", divide="ignore"):
        B = m.L.batch(states, horizon=24, every=20)
        r = ReachStatePipeline.from_frozen(m, None)
        np.testing.assert_allclose(r.rollout(B), m.rollout(B), rtol=0, atol=1e-12)
    assert r.R_log.shape[:2] == (len(B["X0"]), 24)


def test_two_level_gram_recovers_a_sparse_equation():
    rng = np.random.default_rng(0)
    n, names, controls = 4000, ["1", "x", "y", "x*y"], ("1", "u")
    groups, base = group_index(names, controls)
    x, y, u = rng.normal(size=n), rng.normal(size=n), (rng.random(n) < 0.3).astype(float)
    G = np.column_stack([np.ones(n), x, y, x * y])
    Theta = (G[:, :, None] * np.column_stack([np.ones(n), u])[:, None, :]).reshape(n, -1)
    b = 2.0 * x - 1.5 * x * u + 0.01 * rng.normal(size=n)
    c = two_level_gram(*gram(Theta, b), groups, base, 0.05)
    want = np.zeros(8)
    want[2], want[3] = 2.0, -1.5  # x*1, x*u
    np.testing.assert_allclose(c, want, atol=0.01)
