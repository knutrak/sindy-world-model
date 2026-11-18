"""Phase 1 reference rows: the current models and the baselines, refitted per split by the harness.

    ref/persistence        nothing changes
    ref/lanchester-square  H_B' = -a H_R, H_R' = -b H_B (no range, no control)
    ref/lanchester-linear  H_B' = -c H_B H_R, ...
    ref/lanchester-mixed   both
    ref/no-control         physics library without intents
    ref/A                  model_derived's recipe (STLSQ, threshold 0.15)
    ref/B                  A's terms, refitted on rollout error
    ref/S                  B + the share in reach as a state

`cost` only orders the jobs (slow first); it is part of the hashed config, so don't change it.
"""

from sindy_wm.evaluation.harness import Experiment
from sindy_wm.evaluation.harness_models import Lanchester, ModelA, ModelB, ModelS, NoControl, Persistence

EXPERIMENTS = [
    Experiment("ref/persistence", Persistence, {"model": "persistence", "cost": 0}),
    *[
        Experiment(f"ref/lanchester-{law}", lambda law=law: Lanchester(law), {"law": law, "cost": 0})
        for law in ("square", "linear", "mixed")
    ],
    Experiment("ref/no-control", NoControl, {"model": "IntentModel ctrl=False", "cost": 1}),
    Experiment("ref/A", ModelA, {"thr_health": 0.15, "rates": "SG 9", "cost": 1}),
    Experiment("ref/B", ModelB, {"n_start": 20_000, "refit": "lab.refit ALIVE+HEALTH+DIST+TAU", "cost": 10}),
    Experiment("ref/S", ModelS, {"n_start": 20_000, "share terms": "as sindyc_07", "cost": 20}),
]
