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
) -> SpawnSample:
    """Sample two group centers at a random distance and bearing from each other.

    Both sides keep their own cohesive starting formation, as in the base
    scenario; only the distance between the groups and their relative
    direction are randomized. This keeps every episode's approach-then-engage
    structure (see `states.aggregate` and the "trim to engagement"
    preprocessing), instead of mixing units from both teams together.

    Both groups are centered on the middle of `bounds` every episode: only
    their distance and bearing vary, never where on the map the fight happens.
    The terrain is open, so only the relative configuration affects the
    dynamics, never absolute position (see the README's SINDy-state
    findings) — fixing the center loses no real variability, and means the
    longest distance that fits a given bearing can be computed directly
    (basic trig) instead of guessing a center and checking it.

    Args:
        rng: Random generator for this sample. Use a generator independent of
            the one driving the env and the one driving the policy, so spawn
            sampling doesn't consume (or get consumed by) their draws.
        bounds: (xmin, xmax, ymin, ymax) of the walkable area, e.g. from
            `scenarios.walkable_bounds()`.
        min_distance, max_distance: Range the straight-line distance between
            the two group centers is drawn from, in map units. At bearings
            close to the short axis of `bounds`, the drawn distance is capped
            at whatever fits; `min_distance` itself must fit at every
            bearing, checked up front (see raises below).
        margin: Kept clear of the walkable area's edges, to leave room for a
            group's own footprint around its center (bigger units need more;
            5 marines need about 1.2).

    Raises:
        ValueError: if `margin` alone leaves no room in `bounds`, or if
            `min_distance` can't fit at every bearing even before any margin
            is spent on it.

    Returns:
        A SpawnSample with both group centers inside `bounds`, shrunk by `margin`.
    """
    xmin, xmax, ymin, ymax = bounds
    half_width = (xmax - xmin) / 2 - margin
    half_height = (ymax - ymin) / 2 - margin
    if half_width <= 0 or half_height <= 0:
        raise ValueError(f"margin {margin} leaves no room in bounds {bounds}")
    if min_distance > 2 * min(half_width, half_height):
        raise ValueError(
            f"min_distance {min_distance} cannot fit in {bounds} (margin {margin}) "
            "at every bearing; lower it, or shrink margin."
        )

    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Longest distance in this direction that still fits inside the margin-shrunk bounds
    fit_x = half_width / abs(direction[0]) if direction[0] else np.inf
    fit_y = half_height / abs(direction[1]) if direction[1] else np.inf
    distance = rng.uniform(min_distance, min(max_distance, 2 * min(fit_x, fit_y)))

    half = direction * distance / 2
    ally, enemy = center - half, center + half
    return SpawnSample(tuple(ally), tuple(enemy), tuple(ally))
