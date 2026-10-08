"""Randomizing where the two teams start, for scenarios with one group per side."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SpawnSample:
    """Sampled starting positions for one episode.

    attack_point is where SMAClite's built-in Red AI attack-moves to before it
    sees any Blue unit (SMACliteEnv.__enemy_attack). It must track the sampled
    Blue position, or Red would walk toward the scenario's original, now-stale
    spawn point instead of towards Blue.
    """

    ally_pos: tuple[float, float]
    enemy_pos: tuple[float, float]
    attack_point: tuple[float, float]


def sample_opposing_spawns(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float],
    min_distance: float,
    max_distance: float,
    margin: float = 1.5,
    max_tries: int = 1000,
) -> SpawnSample:
    """Sample two group centers at a random distance and bearing from each other.

    Both sides keep their own cohesive starting formation, as in the base
    scenario; only the distance between the groups and their relative
    direction are randomized. This keeps every episode's approach-then-engage
    structure (see `states.aggregate` and the "trim to engagement"
    preprocessing), instead of mixing units from both teams together.

    Args:
        rng: Random generator for this sample. Use a generator independent of
            the one driving the env and the one driving the policy, so spawn
            sampling doesn't consume (or get consumed by) their draws.
        bounds: (xmin, xmax, ymin, ymax) of the walkable area, e.g. from
            `scenarios.walkable_bounds()`.
        min_distance, max_distance: Range the straight-line distance between
            the two group centers is drawn from, in map units.
        margin: Kept clear of the walkable area's edges, to leave room for a
            group's own footprint around its center (bigger units need more;
            5 marines need about 1.2).
        max_tries: Resample up to this many times if a draw falls outside
            `bounds` after the margin; raises if none succeeds.

    Returns:
        A SpawnSample with both group centers inside `bounds`, shrunk by `margin`.
    """
    xmin, xmax, ymin, ymax = bounds
    xmin, ymin = xmin + margin, ymin + margin
    xmax, ymax = xmax - margin, ymax - margin
    if xmin > xmax or ymin > ymax:
        raise ValueError(f"margin {margin} leaves no room in bounds {bounds}")

    for _ in range(max_tries):
        distance = rng.uniform(min_distance, max_distance)
        angle = rng.uniform(0, 2 * np.pi)
        half = np.array([np.cos(angle), np.sin(angle)]) * distance / 2
        center = rng.uniform([xmin, ymin], [xmax, ymax])
        ally, enemy = center - half, center + half
        if (
            xmin <= ally[0] <= xmax
            and ymin <= ally[1] <= ymax
            and xmin <= enemy[0] <= xmax
            and ymin <= enemy[1] <= ymax
        ):
            return SpawnSample(tuple(ally), tuple(enemy), tuple(ally))

    raise RuntimeError(
        f"Could not place both groups within {bounds} (margin {margin}) after "
        f"{max_tries} tries. Lower max_distance, or widen bounds/shrink margin."
    )
