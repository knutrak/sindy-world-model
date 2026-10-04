"""Creating SMAClite environments from scenario names.

A scenario is either one of SMAClite's built-in maps (e.g. "3s5z") or one of
our own JSON files in configs/scenarios/ (e.g. "5m_vs_5m").
"""

from pathlib import Path

import gymnasium as gym
import smaclite  # noqa: F401  (importing registers the SMAClite environments)
from smaclite.env.maps.map import MapPreset

# src/sindy_wm/envs/scenarios.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = PROJECT_ROOT / "configs" / "scenarios"


def builtin_scenarios() -> list[str]:
    """Names of SMAClite's built-in maps."""
    return sorted(preset.value.name for preset in MapPreset)


def custom_scenarios() -> list[str]:
    """Names of our own scenarios in configs/scenarios/."""
    return sorted(path.stem for path in SCENARIO_DIR.glob("*.json"))


def make_env(scenario: str, use_cpp_rvo2: bool = False, **kwargs) -> gym.Env:
    """Create a SMAClite environment for a scenario.

    Args:
        scenario: Name of a built-in map ("3s5z"), name of a JSON file in
            configs/scenarios/ without the extension ("5m_vs_5m"), or a path
            to any scenario JSON file.
        use_cpp_rvo2: Use SMAClite's faster C++ collision avoidance
            (requires the optional C++ extension to be installed).
        **kwargs: Passed on to gym.make.
    """
    path = Path(scenario)
    if path.suffix == ".json":
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        return gym.make(
            "smaclite/custom-v0", map_file=str(path), use_cpp_rvo2=use_cpp_rvo2, **kwargs
        )

    is_builtin = scenario in builtin_scenarios()
    is_custom = scenario in custom_scenarios()

    if is_builtin and is_custom:
        raise ValueError(
            f"'{scenario}' is both a built-in map and a file in {SCENARIO_DIR}. "
            "Rename the custom file to avoid confusion."
        )
    if is_builtin:
        return gym.make(f"smaclite/{scenario}-v0", use_cpp_rvo2=use_cpp_rvo2, **kwargs)
    if is_custom:
        map_file = SCENARIO_DIR / f"{scenario}.json"
        return gym.make(
            "smaclite/custom-v0", map_file=str(map_file), use_cpp_rvo2=use_cpp_rvo2, **kwargs
        )

    raise ValueError(
        f"Unknown scenario '{scenario}'.\n"
        f"Built-in: {builtin_scenarios()}\n"
        f"Custom: {custom_scenarios()}"
    )