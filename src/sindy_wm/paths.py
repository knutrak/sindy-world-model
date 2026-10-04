"""Standard locations in the project, independent of where code is run from."""

from pathlib import Path

# src/sindy_wm/paths.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SCENARIO_DIR = PROJECT_ROOT / "configs" / "scenarios"
