"""Creating SMAClite environments from scenario names.

A scenario is either one of SMAClite's built-in maps (e.g. "3s5z") or one of
our own JSON files in configs/scenarios/ (e.g. "5m_vs_5m").
"""

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import smaclite  # noqa: F401  (importing registers the SMAClite environments)
import smaclite.env.smaclite as smaclite_module
from smaclite.env.maps.map import MapInfo, MapPreset
from smaclite.env.terrain.terrain import TerrainType
from smaclite.env.units.unit import TICKS_PER_SECOND
from smaclite.env.util.faction import Faction

from sindy_wm.envs.spawn_sampling import SpawnSample

# src/sindy_wm/envs/scenarios.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = PROJECT_ROOT / "configs" / "scenarios"


def builtin_scenarios() -> list[str]:
    """Names of SMAClite's built-in maps."""
    return sorted(preset.value.name for preset in MapPreset)


def custom_scenarios() -> list[str]:
    """Names of our own scenarios in configs/scenarios/."""
    return sorted(path.stem for path in SCENARIO_DIR.glob("*.json"))


def seconds_per_step(step_mul: int) -> float:
    """Game seconds per env.step at a given step_mul (16 ticks = 1 s)."""
    return step_mul / TICKS_PER_SECOND


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
        self.seconds_per_step = seconds_per_step(step_mul)

    def step(self, action):
        smaclite_module.STEP_MUL = self.step_mul
        return self.env.step(action)


def make_env(
    scenario: str,
    step_mul: int = 8,
    use_cpp_rvo2: bool = False,
    spawn: SpawnSample | None = None,
    **kwargs,
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
        spawn: Override the scenario's ally/enemy starting positions (and the
            built-in Red AI's attack point), e.g. from `sample_opposing_spawns()`.
            Only scenarios with exactly one ALLY group and one ENEMY group
            support this (true for 5m_vs_5m); ignored for built-in maps.
        **kwargs: Passed on to gym.make.
    """
    if spawn is not None:
        map_info = _with_spawn(load_map_info(scenario), spawn)
        env = _make_quietly(
            lambda: gym.make("smaclite/custom-v0", map_info=map_info, use_cpp_rvo2=use_cpp_rvo2, **kwargs)
        )
    else:
        env = _make_quietly(lambda: _make_smaclite_env(scenario, use_cpp_rvo2=use_cpp_rvo2, **kwargs))
    return StepMulWrapper(env, step_mul)


# SMAClite prints this unconditionally every time an environment is constructed
# (NumpyVelocityUpdater.__init__), with no way to silence it from the outside.
# With randomize_spawn=True, collect() builds one env per episode instead of one
# per dataset, so this would otherwise print thousands of times per run.
_SMACLITE_CONSTRUCTION_NOISE = "Using the numpy RVO2 port"


def _make_quietly(build) -> gym.Env:
    """Call build(), swallowing SMAClite's own construction-time print (see above).

    Anything else printed during construction is not swallowed, so a genuine
    warning from a future SMAClite version would still surface.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = build()
    leftover = "\n".join(
        line for line in buf.getvalue().splitlines() if line.strip() != _SMACLITE_CONSTRUCTION_NOISE
    )
    if leftover:
        print(leftover)
    return env


def load_map_info(scenario: str) -> MapInfo:
    """Load the MapInfo of one of our custom scenario JSON files (not a built-in map)."""
    path = Path(scenario) if scenario.endswith(".json") else SCENARIO_DIR / f"{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    return MapInfo.from_file(str(path))


def walkable_bounds(map_info: MapInfo) -> tuple[float, float, float, float]:
    """Bounding box (xmin, xmax, ymin, ymax) of the map's walkable (NORMAL) terrain.

    Assumes the walkable area is one simply-connected rectangle. True for the
    SIMPLE preset used by 5m_vs_5m (an open band, bordered top and bottom) but
    not guaranteed for maps with terrain features inside that box.
    """
    normal = np.array([[cell == TerrainType.NORMAL for cell in row] for row in map_info.terrain])
    ys, xs = np.nonzero(normal)
    if len(xs) == 0:
        raise ValueError("Map has no walkable (NORMAL) terrain")
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _with_spawn(map_info: MapInfo, spawn: SpawnSample) -> MapInfo:
    """Copy of map_info with its ally/enemy group centers and attack_point overridden."""
    by_faction = {}
    for g in map_info.groups:
        by_faction.setdefault(g.faction, []).append(g)
    if len(by_faction.get(Faction.ALLY, [])) != 1 or len(by_faction.get(Faction.ENEMY, [])) != 1:
        raise NotImplementedError(
            "Randomized spawns need a scenario with exactly one ALLY group and one ENEMY group."
        )
    groups = [
        replace(g, x=spawn.ally_pos[0], y=spawn.ally_pos[1])
        if g.faction == Faction.ALLY
        else replace(g, x=spawn.enemy_pos[0], y=spawn.enemy_pos[1])
        for g in map_info.groups
    ]
    return replace(map_info, groups=groups, attack_point=spawn.attack_point)


def _make_smaclite_env(scenario: str, use_cpp_rvo2: bool, **kwargs) -> gym.Env:
    """Create the plain SMAClite environment for a scenario name or path."""
    path = Path(scenario)
    if path.suffix == ".json":
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        return gym.make("smaclite/custom-v0", map_file=str(path), use_cpp_rvo2=use_cpp_rvo2, **kwargs)

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
        return gym.make("smaclite/custom-v0", map_file=str(map_file), use_cpp_rvo2=use_cpp_rvo2, **kwargs)

    raise ValueError(
        f"Unknown scenario '{scenario}'.\nBuilt-in: {builtin_scenarios()}\nCustom: {custom_scenarios()}"
    )
