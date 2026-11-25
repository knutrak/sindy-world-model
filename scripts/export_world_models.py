"""Write the frozen pipeline models to configs/world_models/*.json for `sindy_wm.world_model`.

Reads the fitted models (`experiments/generalisation/frozen_models.py`, train only) and the error numbers
already recorded by the one held-out run (docs/generalisation/04_test_results.json). Nothing is fitted or
evaluated here.

    uv run scripts/export_world_models.py
"""

import json
import pickle

import yaml

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT
from sindy_wm.world_model import MODEL_DIR, WorldModel

FROZEN = DATA_DIR / "processed" / "final_exploration" / "generalisation_frozen.pkl"
RESULTS = PROJECT_ROOT / "docs" / "generalisation" / "04_test_results.json"
CONFIG = PROJECT_ROOT / "configs" / "generalisation_pipeline.yaml"
# G6 on the held-out set (margin R² >= 0.3, lower bound > 0), docs/generalisation/04_test_results.md
RELIABLE_HORIZON_S = {"marines": 2, "marauders": 3, "mixed": 3}
HORIZONS_S = (1, 2, 4)


def main() -> None:
    models = pickle.load(open(FROZEN, "rb"))
    results = json.loads(RESULTS.read_text())
    scenarios = yaml.safe_load(CONFIG.read_text())["scenarios"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        table = results[name]["table"]
        # health error is per side, in ten full units of the first type (evaluation.harness.horizon_metrics)
        hp = 10 * m.L.hp[m.L.types[0]]
        info = {
            "source": "docs/generalisation/06_frozen_equations.md",
            "train": f"{scenarios[name]['experiment']}/train",
            "held_out": f"{scenarios[name]['experiment']}/{scenarios[name]['held_out']}",
            "reliable_horizon_s": RELIABLE_HORIZON_S[name],
            "held_out_error": {
                "horizon_s": list(HORIZONS_S),
                "hp": [table[f"health_{h}s"]["pipeline"] * hp for h in HORIZONS_S],
                "units": [table[f"alive_{h}s"]["pipeline"] for h in HORIZONS_S],
                "distance": [table[f"distance_{h}s"]["pipeline"] for h in HORIZONS_S],
            },
        }
        path = MODEL_DIR / f"{name}.json"
        WorldModel(name, m, info).save(path)
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
