"""Axes D/I (one model across battle sizes) and J (black-box reference on the same state).

D/size-scaled          A's library plus every health / distance candidate times battle size z
D/size-scaled-thr0.05  the same at a lower health threshold (do size terms survive at all?)
J/mlp                  MLP world model on state + formation + intents + start share, trained on
                       2 s rollout loss: what the same inputs allow without interpretability
J/mlp-wide             the same with 128 hidden units and 4 s rollouts
"""

from sindy_wm.evaluation.explore_models import MLPWorld, SizeScaledA
from sindy_wm.evaluation.harness import Experiment

EXPERIMENTS = [
    Experiment(
        "D/size-scaled", SizeScaledA, {"thr_health": 0.15, "z": "(n0 - 20) / 10", "cost": 1}, axis="D"
    ),
    Experiment(
        "D/size-scaled-thr0.05",
        lambda: SizeScaledA(thr_health=0.05),
        {"thr_health": 0.05, "z": "(n0 - 20) / 10", "cost": 1},
        axis="D",
    ),
    Experiment(
        "J/mlp", MLPWorld, {"hidden": 64, "steps": 16, "epochs": 30, "n_start": 40_000, "cost": 4}, axis="J"
    ),
    Experiment(
        "J/mlp-wide",
        lambda: MLPWorld(hidden=128, steps=32),
        {"hidden": 128, "steps": 32, "epochs": 30, "n_start": 40_000, "cost": 8},
        axis="J",
    ),
]
