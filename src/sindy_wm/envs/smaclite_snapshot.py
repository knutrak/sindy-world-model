"""Turn SMAClite game states into a per-unit table.

SMAClite removes dead units from env.unwrapped.agents / .enemies, so we
record the full roster at reset and fill in dead units with hp = 0.
"""
import pandas as pd


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
        return self.snapshot(step=0)

    def snapshot(self, step: int) -> pd.DataFrame:
        live = {(team, uid): u for team, uid, u in self._live_units()}
        rows = []
        for (team, uid), static in self.roster.items():
            u = live.get((team, uid))
            alive = u is not None and u.hp > 0
            rows.append({
                "step": step, "team": team, "unit_id": uid, **static,
                "alive": alive,
                "x": float(u.pos[0]) if u is not None else float("nan"),
                "y": float(u.pos[1]) if u is not None else float("nan"),
                "hp": float(u.hp) if alive else 0.0,
                "shield": float(u.shield) if alive else 0.0,
                "cooldown": float(u.cooldown) if alive else float("nan"),
            })
        return pd.DataFrame(rows)
