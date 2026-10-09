"""Turn SMAClite game states into a per-unit table.

SMAClite removes dead units from env.unwrapped.agents / .enemies, so we
record the full roster at reset and fill in dead units with hp = 0.

Columns per unit and step:
    alive, x, y, hp, shield, cooldown   as before
    vx, vy             velocity in map units per second, as applied in the last
                       game tick (after SMAClite's collision avoidance); 0 for
                       dead units
    target_team        team of the unit this unit is going for ("blue"/"red"),
                       or None. Usually the other team; healers (medivac) target
                       their own team
    target_id          unit_id of that target, or -1 if it has none
    target_in_range    True if the target is within attack range, i.e. the unit
                       is shooting (or about to, once its cooldown is over)
                       rather than still moving towards it

A unit's target is the unit it has been ordered to attack, or picked itself
when attack-moving (Red's built-in AI). Having a target does not mean being
in range: an attack order on a far-away unit sets the target immediately and
makes the unit walk towards it. Use target_in_range for "engaged".
"""

import pandas as pd

NO_TARGET = -1


class SnapshotLogger:
    def __init__(self, env):
        self.env = env
        self.roster = None  # (team, unit_id) -> static info

    def _live_units(self):
        game = self.env.unwrapped
        for team, units in (("blue", game.agents), ("red", game.enemies)):
            for uid, u in units.items():
                yield team, uid, u

    def start_episode(self) -> pd.DataFrame:
        """Call right after env.reset(). Records the roster and returns step 0."""
        self.roster = {
            (team, uid): {
                "unit_type": getattr(u.type, "name", str(u.type)),
                "max_hp": float(u.max_hp),
                "max_shield": float(u.max_shield),
            }
            for team, uid, u in self._live_units()
        }
        # Unit object -> (team, unit_id), to identify targets
        self._key_of = {id(u): (team, uid) for team, uid, u in self._live_units()}
        return self.snapshot(step=0)

    def _target(self, u) -> tuple[str | None, int, bool]:
        """(target_team, target_id, target_in_range) of a live unit."""
        t = u.target
        if t is None or t.hp <= 0 or id(t) not in self._key_of:
            return None, NO_TARGET, False
        team, uid = self._key_of[id(t)]
        return team, int(uid), bool(u.has_within_attack_range(t))

    def snapshot(self, step: int) -> pd.DataFrame:
        live = {(team, uid): u for team, uid, u in self._live_units()}
        rows = []
        for (team, uid), static in self.roster.items():
            u = live.get((team, uid))
            alive = u is not None and u.hp > 0
            target_team, target_id, in_range = self._target(u) if alive else (None, NO_TARGET, False)
            rows.append(
                {
                    "step": step,
                    "team": team,
                    "unit_id": uid,
                    **static,
                    "alive": alive,
                    "x": float(u.pos[0]) if u is not None else float("nan"),
                    "y": float(u.pos[1]) if u is not None else float("nan"),
                    "hp": float(u.hp) if alive else 0.0,
                    "shield": float(u.shield) if alive else 0.0,
                    "cooldown": float(u.cooldown) if alive else float("nan"),
                    "vx": float(u.velocity[0]) if alive else 0.0,
                    "vy": float(u.velocity[1]) if alive else 0.0,
                    "target_team": target_team,
                    "target_id": target_id,
                    "target_in_range": in_range,
                }
            )
        return pd.DataFrame(rows)
