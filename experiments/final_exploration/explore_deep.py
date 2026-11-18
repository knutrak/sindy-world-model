"""Depth stage: combine what held up in screening, and test what the closure needs.

deep/A*              A + one firepower coefficient for both sides + alive law on the reach
                     regressor (one-step fits only)
deep/B*              A*'s support, coefficients refitted on 1-8 s rollout error (trajectory refinement)
deep/S*              B* plus the share in reach as a state (S's terms, refitted on rollouts)
deep/share-sindy     S's share equation fitted by one-step STLSQ (thr 0.01, sign rule) on top of
                     A with the reach alive law: does S's gain need rollout fitting?
"""

from sindy_wm.evaluation.explore_models import CombinedA, ShareSINDy
from sindy_wm.evaluation.harness import Experiment
from sindy_wm.evaluation.harness_models import ModelB, ModelS

EXPERIMENTS = [
    Experiment(
        "deep/A*", CombinedA, {"health": "symmetric k", "alive": "reach regressor", "cost": 1}, axis="deep"
    ),
    Experiment(
        "deep/B*",
        lambda: ModelB(base=CombinedA),
        {"base": "A*", "refit": "rollout 1-8 s", "cost": 10},
        axis="deep",
    ),
    Experiment(
        "deep/S*",
        lambda: ModelS(base=CombinedA),
        {"base": "A*", "share": "as S, rollout", "cost": 20},
        axis="deep",
    ),
    Experiment(
        "deep/share-sindy",
        lambda: ShareSINDy(thr=0.01, signs=True),
        {"thr": 0.01, "signs": True, "alive": "reach", "cost": 2},
        axis="deep",
    ),
]
