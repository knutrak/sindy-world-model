"""Axes C (phases) and E (optimizers, symmetry constraints), all on A's library and closures.

E/sr3, E/sr3-unbiased   SR3 with an L0 penalty matched to STLSQ's threshold 0.15; without and with
                        the least-squares refit on the selected terms (selection vs shrinkage)
E/ssr-<k>, E/frols-<k>  stepwise sparse regression / forward orthogonal regression, L0 penalty k
E/symmetric             one firepower coefficient and one alive exponent for both sides
E/symmetric-health      one firepower coefficient only (alive as A)
C/phase-hybrid          separate fits in approach / engaged / mop-up, switched by a rule on the
                        predicted state (Hybrid SINDy)
"""

from sindy_wm.evaluation.explore_models import OptimizerA, PhaseA, SymmetricA
from sindy_wm.evaluation.harness import Experiment

EXPERIMENTS = [
    Experiment(
        "E/sr3",
        lambda: OptimizerA("sr3", 0.15),
        {"optimizer": "SR3 L0", "lam": "0.15^2/2", "cost": 1},
        axis="E",
    ),
    Experiment(
        "E/sr3-unbiased",
        lambda: OptimizerA("sr3-unbiased", 0.15),
        {"optimizer": "SR3 L0 unbias", "lam": "0.15^2/2", "cost": 1},
        axis="E",
    ),
    *[
        Experiment(
            f"E/{o}-{k:g}",
            lambda o=o, k=k: OptimizerA(o, k),
            {"optimizer": o, "kappa": k, "cost": 1},
            axis="E",
        )
        for o in ("ssr", "frols")
        for k in (0.01, 0.1)
    ],
    Experiment("E/symmetric", SymmetricA, {"shared": "k, k_wd, p", "cost": 1}, axis="E"),
    Experiment(
        "E/symmetric-health", lambda: SymmetricA(alive=False), {"shared": "k, k_wd", "cost": 1}, axis="E"
    ),
    Experiment(
        "C/phase-hybrid", PhaseA, {"phases": "approach/engaged/mop-up", "mop": 0.3, "cost": 1}, axis="C"
    ),
]
