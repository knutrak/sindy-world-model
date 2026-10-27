"""Drawing each episode's own starting conditions: unit counts, starting health and spawn.

Used by `io.collect.collect(..., sampler=EpisodeSampler(...))`: instead of one
fixed setting per dataset, every episode draws its own, so a single dataset
covers a range of force sizes, ratios and starting states. The draws are
recorded in the episode index (`EpisodeSettings.record()`).
"""

from dataclasses import asdict, dataclass

import numpy as np

from sindy_wm.envs.spawn_sampling import SpawnSample, formation_half_width, sample_opposing_spawns


@dataclass(frozen=True)
class EpisodeSettings:
    """One episode's starting conditions, as drawn by `EpisodeSampler.sample`."""

    unit_counts: tuple[int, int]  # (Blue, Red)
    start_health: tuple[float, float]  # (Blue, Red) fraction of full health every unit starts with
    spawn: SpawnSample

    def record(self) -> dict:
        """Flat, JSON-serializable summary for the episode index."""
        ally, enemy = np.array(self.spawn.ally_pos), np.array(self.spawn.enemy_pos)
        dx, dy = enemy - ally
        return {
            "n_blue": self.unit_counts[0],
            "n_red": self.unit_counts[1],
            "start_health_blue": self.start_health[0],
            "start_health_red": self.start_health[1],
            "spawn_distance": float(np.hypot(dx, dy)),
            "spawn_bearing_deg": float(np.degrees(np.arctan2(dy, dx)) % 360),
        }


@dataclass(frozen=True)
class EpisodeSampler:
    """Ranges each episode's starting conditions are drawn from.

    Attributes:
        red_units: (min, max) number of Red units, both included.
        force_ratio: (min, max) of Blue's units divided by Red's. Drawn
            log-uniformly, so Blue being ahead by a factor and Red being ahead
            by the same factor are equally likely (0.75 and 1.33 are mirror
            images); Blue's count is Red's times the ratio, rounded. Ratios far
            from 1 mostly give battles whose winner is clear from the start.
        damaged_start_share: Probability, per side, of starting damaged.
        start_health: (min, max) fraction of full health a damaged side's
            units start with (all units of the side get the same fraction).
            A side that isn't damaged starts at 1.
        spawn_distance: (min, max) distance between the two group centers, in
            map units. Below a marine's attack range (5) battles start in
            contact; the minimum is raised if the two formations would overlap.
        unit_size: Diameter of one unit (0.75 for marines), for the formation size.
    """

    red_units: tuple[int, int] = (6, 14)
    force_ratio: tuple[float, float] = (0.75, 1.33)
    damaged_start_share: float = 0.5
    start_health: tuple[float, float] = (0.4, 1.0)
    spawn_distance: tuple[float, float] = (4.0, 16.0)
    unit_size: float = 0.75

    def config(self) -> dict:
        return asdict(self)

    def sample(self, seed: int, bounds: tuple[float, float, float, float]) -> EpisodeSettings:
        """Draw one episode's settings.

        Uses its own random stream, `default_rng([seed, 2])`, independent of the
        policy's (`default_rng(seed)`) and of `SwitchingPolicy`'s schedule
        (`[seed, 1]`), so the same seed still describes the whole episode.

        Args:
            seed: The episode's seed.
            bounds: Walkable area (xmin, xmax, ymin, ymax), from `scenarios.walkable_bounds`.
        """
        rng = np.random.default_rng([seed, 2])
        n_red = int(rng.integers(self.red_units[0], self.red_units[1] + 1))
        ratio = np.exp(rng.uniform(*np.log(self.force_ratio)))
        n_blue = max(1, round(n_red * ratio))
        damaged = rng.random(2) < self.damaged_start_share
        health = np.where(damaged, rng.uniform(*self.start_health, size=2), 1.0)

        half_blue, half_red = (formation_half_width(n, self.unit_size) for n in (n_blue, n_red))
        margin = max(half_blue, half_red) + 0.3
        min_distance = max(self.spawn_distance[0], half_blue + half_red + 0.5)
        spawn = sample_opposing_spawns(
            rng, bounds, min_distance, max(min_distance, self.spawn_distance[1]), margin
        )
        return EpisodeSettings((n_blue, n_red), (float(health[0]), float(health[1])), spawn)
