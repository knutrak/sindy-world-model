"""μ, σ reach states (docs/generalisation/10_mu_sigma_plan.md): with zero coefficients μ and σ stay at their
start
values, and the reach the rollout uses is A·Φ((c − μ)/(kσ)) with the model's own alive counts."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sindy_wm.models.intent_model import add_controls
from sindy_wm.models.kite import CTRL_K, add_kite
from sindy_wm.models.mu_sigma import MuSigmaPipeline, attach_margins, group_names, share
from sindy_wm.paths import DATA_DIR

MODELS = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
VAL = DATA_DIR / "processed" / "exp08_mixed_typed" / "val.parquet"
MARG = DATA_DIR / "processed" / "exp08_mixed_margins" / "val.parquet"


@pytest.mark.skipif(not (MODELS.exists() and VAL.exists() and MARG.exists()), reason="data not built")
def test_zero_equations_keep_mu_sigma_and_reach_follows_the_mapping():
    m = pickle.load(open(MODELS, "rb"))["mixed"]
    st = pd.read_parquet(VAL).join(pd.read_parquet(MARG), how="inner")
    st = add_kite(add_controls(st[st.index.get_level_values(0) < 5]))
    with np.errstate(invalid="ignore", divide="ignore"):
        B = attach_margins(m.L.batch(st, horizon=16, every=20, ctrl=CTRL_K), st, m.L)
    names, ctrls = group_names(True), ("1", "adv", "hold", "wd", "focus", "kite")
    zero = np.zeros(len(names) * len(ctrls))
    eqs = {(s, a): {"mu": zero, "sd": zero} for s in ("blue", "red") for a in m.L.types}
    ms = MuSigmaPipeline.build(m, eqs, names, ctrls, True, 0.07, 1.2)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = ms.rollout(B)
    for kk in eqs:
        np.testing.assert_allclose(ms._mu[kk], np.nan_to_num(B["mu0"][:, m.L.pos(*kk)]))
    blue = sum(
        X[:, 0, m.L.idx("alive", "blue", a)]
        * share(
            np.nan_to_num(B["mu0"][:, m.L.pos("blue", a)]),
            np.nan_to_num(B["sd0"][:, m.L.pos("blue", a)]),
            0.07,
            1.2,
        )
        for a in m.L.types
    )
    np.testing.assert_allclose(ms.R_log[:, 0, 0], blue)
    assert not ms.blown.any()
