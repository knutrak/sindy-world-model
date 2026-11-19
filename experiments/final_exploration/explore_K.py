"""K: is one firepower coefficient for both sides justified? (19 Nov, after A* was chosen; post-hoc)

Red (SMAClite attack-move AI) acquires new targets only within minimum_scan_range + target radius
(5.375 for marines), Blue shoots out to 5.75. A* fits one k for both sides on reach measured at 5.75.

K/separate-k            = G/alive-reach: A's separate k_B, k_R + A*'s alive law (already in the table;
                          listed here for the summary, not refitted)
K/redscan-separate-k    A's recipe with separate k on the `_redscan` tables (Red's reach at 5.375,
                          Red's closure fitted at 5.375): does the k gap close?
K/redscan-shared-k      A* (one k) on the `_redscan` tables

The `_redscan` experiments read data/processed/<experiment>_redscan (scripts/build_red_scan_states.py)
through `redscan_loaders()`, which swaps the experiment folder in the harness's loaders.

    uv run --extra explore experiments/final_exploration/explore_K.py
"""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import RedScanA  # noqa: E402
from sindy_wm.evaluation.harness import Experiment  # noqa: E402

EXPERIMENTS = [
    Experiment(
        "K/redscan-separate-k",
        functools.partial(RedScanA, shared=False),
        {"health": "separate k", "alive": "reach regressor", "red reach": 5.375, "cost": 1},
        axis="K",
        note="post-hoc (19 Nov): Red reach at its acquisition radius",
    ),
    Experiment(
        "K/redscan-shared-k",
        functools.partial(RedScanA, shared=True),
        {"health": "symmetric k", "alive": "reach regressor", "red reach": 5.375, "cost": 1},
        axis="K",
        note="post-hoc (19 Nov): Red reach at its acquisition radius",
    ),
]


def redscan_loaders() -> None:
    """Point the harness's loaders at the `_redscan` tables (rates are cached under their own name)."""
    load, rates = H.load_states, H.rows_with_rates

    def load_states(name, experiment="exp06_intents"):
        return load(name, experiment + "_redscan")

    def rows_with_rates(name, window=9, experiment="exp06_intents"):
        return rates(name, window, experiment + "_redscan")

    H.load_states, H.rows_with_rates = load_states, rows_with_rates


def main() -> None:
    redscan_loaders()
    cache: dict = {}
    for exp in EXPERIMENTS:
        H.run(exp, splits=("cv", "val", "stalkers"), cache=cache)


if __name__ == "__main__":
    main()
