"""Axes E and G: how robust are A's terms, which death law, and win probabilities.

E/thr-<t>          A's recipe with health threshold t (threshold sweep, chosen by rollout error)
E/ensemble         E-SINDy: 30 bootstrap refits by battle, keep terms with inclusion >= 0.5,
                   median coefficients; ensemble rollouts give a win probability
E/ensemble-noalivectrl  the same with no intent terms in the alive law
G/alive-noctrl     A with the alive law refitted without intent terms (G3b's moving terms)
G/alive-proportional    alive in proportion to health (p = 1): the naive death rule
G/alive-reach      A's law fitted on the regressor the rollout uses (R_enemy A / H)
G/alive-rollout    A with only p_b, p_r, p_r_focus refitted on 1-8 s rollout error
"""

from sindy_wm.evaluation.explore_models import AliveRolloutA, AliveVariantA, EnsembleA
from sindy_wm.evaluation.harness import Experiment
from sindy_wm.evaluation.harness_models import ModelA

EXPERIMENTS = [
    *[
        Experiment(f"E/thr-{t:g}", lambda t=t: ModelA(thr_health=t), {"thr_health": t, "cost": 1}, axis="E")
        for t in (0.02, 0.05, 0.1, 0.2, 0.3)
    ],
    Experiment("E/ensemble", lambda: EnsembleA(30), {"n_models": 30, "keep": 0.5, "cost": 3}, axis="E"),
    Experiment(
        "E/ensemble-noalivectrl",
        lambda: EnsembleA(30, alive_controls=False),
        {"n_models": 30, "keep": 0.5, "alive_controls": False, "cost": 3},
        axis="E",
    ),
    Experiment(
        "G/alive-noctrl", lambda: AliveVariantA("no-control"), {"law": "no-control", "cost": 1}, axis="G"
    ),
    Experiment(
        "G/alive-proportional", lambda: AliveVariantA("proportional"), {"law": "p = 1", "cost": 1}, axis="G"
    ),
    Experiment(
        "G/alive-reach",
        lambda: AliveVariantA("reach"),
        {"law": "A' = c R_enemy A / H (+ focus for Red), p = c / k", "cost": 1},
        axis="G",
    ),
    Experiment(
        "G/alive-rollout", AliveRolloutA, {"refit": "p_b, p_r, p_r_focus on 1-8 s", "cost": 5}, axis="G"
    ),
]
