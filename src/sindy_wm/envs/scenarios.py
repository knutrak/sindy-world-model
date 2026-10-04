"""Creating SMAClite environments from scenario names.

A scenario is either one of SMAClite's built-in maps (e.g. "3s5z") or one of
our own JSON files in configs/scenarios/ (e.g. "5m_vs_5m").
"""

from pathlib import Path

import gymnasium as gym
import smaclite  # noqa: F401  (importing registers the SMAClite environments)
import smaclite.env.smaclite as smaclite_module
from smaclite.env.maps.map import MapPreset
from smaclite.env.units.unit import TICKS_PER_SECOND

# src/sindy_wm/envs/scenarios.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = PROJECT_ROOT / "configs" / "scenarios"


def builtin_scenarios() -> list[str]:
    """Names of SMAClite's built-in maps."""
    return sorted(preset.value.name for preset in MapPreset)


def custom_scenarios() -> list[str]:
    """Names of our own scenarios in configs/scenarios/."""
    return sorted(path.stem for path in SCENARIO_DIR.glob("*.json"))


class StepMulWrapper(gym.Wrapper):
    """Gives an environment its own step_mul (game ticks per env.step).

    SMAClite stores step_mul as a global constant, STEP_MUL. This wrapper sets
    it right before every step, so environments with different values can be
    used side by side without affecting each other.
    """

    def __init__(self, env: gym.Env, step_mul: int):
        super().__init__(env)
        if step_mul < 1:
            raise ValueError(f"step_mul must be at least 1, got {step_mul}")
        self.step_mul = step_mul
        self.seconds_per_step = step_mul / TICKS_PER_SECOND

    def step(self, action):
        smaclite_module.STEP_MUL = self.step_mul
        return self.env.step(action)


def make_env(
    scenario: str, step_mul: int = 8, use_cpp_rvo2: bool = False, **kwargs
) -> StepMulWrapper:
    """Create a SMAClite environment for a scenario.

    Args:
        scenario: Name of a built-in map ("3s5z"), name of a JSON file in
            configs/scenarios/ without the extension ("5m_vs_5m"), or a path
            to any scenario JSON file.
        step_mul: Game ticks per env.step. The game runs at 16 ticks per
            second, so the default of 8 gives 0.5 s per step (as in SMAC).
        use_cpp_rvo2: Use SMAClite's faster C++ collision avoidance
            (requires the optional C++ extension to be installed).
        **kwargs: Passed on to gym.make.
    """
    env = _make_smaclite_env(scenario, use_cpp_rvo2=use_cpp_rvo2, **kwargs)
    return StepMulWrapper(env, step_mul)


def _make_smaclite_env(scenario: str, use_cpp_rvo2: bool, **kwargs) -> gym.Env:
    """Create the plain SMAClite environment for a scenario name or path."""
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
