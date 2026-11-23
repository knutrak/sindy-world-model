"""Per-type roles and target-type priority (docs/planning/48_richer_intents_plan.md).

`RolePolicy(marine, marauder, priority)`: every Blue marine plays `IntentPolicy(marine, "nearest")`, every
marauder `IntentPolicy(marauder, "nearest")` (movements advance / stand / withdraw); then, if `priority` is
"marine" or "marauder", an attack is swapped for the nearest enemy of that type inside the unit's own reach,
if there is one.
Name form for the oracle and the runners: "role:<marine movement>-<marauder movement>:<priority>".
"""

from __future__ import annotations

from itertools import product

import numpy as np

from sindy_wm.envs.intents import IntentPolicy
from sindy_wm.envs.policies import FIRST_TARGET, Policy

MOVES = ("advance", "stand", "withdraw")
PRIORITIES = ("nearest", "marine", "marauder")
ROLE_NAMES = tuple(f"role:{a}-{b}:{p}" for a, b, p in product(MOVES, MOVES, PRIORITIES))


def _type(unit) -> str:
    return getattr(unit.type, "name", str(unit.type)).upper()


class RolePolicy(Policy):
    def __init__(self, marine: str, marauder: str, priority: str = "nearest", withdraw_radius: float = 8.0):
        if marine not in MOVES or marauder not in MOVES or priority not in PRIORITIES:
            raise ValueError((marine, marauder, priority))
        self.marine, self.marauder, self.priority, self.withdraw_radius = (
            marine,
            marauder,
            priority,
            withdraw_radius,
        )
        self.movement, self.target = f"{marine}-{marauder}", priority  # for `setting()` logs
        self._pol = {
            m: IntentPolicy(m, "nearest", withdraw_radius=withdraw_radius) for m in {marine, marauder}
        }

    @classmethod
    def from_name(cls, name: str, withdraw_radius: float = 8.0) -> RolePolicy:
        _, moves, prio = name.split(":")
        a, b = moves.split("-")
        return cls(a, b, prio, withdraw_radius)

    def params(self) -> dict:
        return {
            "marine": self.marine,
            "marauder": self.marauder,
            "priority": self.priority,
            "withdraw_radius": self.withdraw_radius,
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        for p in self._pol.values():
            p.reset(seed)

    def setting(self) -> dict:
        return {"movement": self.movement, "target": self.target}

    def act(self, env) -> list[int]:
        game = env.unwrapped
        acts = {m: p.act(env) for m, p in self._pol.items()}
        masks = game.get_avail_actions()
        out = []
        for i in range(len(masks)):
            unit = game.agents.get(i)
            if unit is None:
                out.append(0)
                continue
            move = self.marine if _type(unit) == "MARINE" else self.marauder
            a = acts[move][i]
            if self.priority != "nearest" and a >= FIRST_TARGET:
                want = self.priority.upper()
                reach = [
                    int(k)
                    for k in np.flatnonzero(masks[i][FIRST_TARGET:])
                    if _type(game.enemies[int(k)]) == want
                    and unit.has_within_attack_range(game.enemies[int(k)])
                ]
                if reach:
                    k = min(reach, key=lambda k: np.linalg.norm(game.enemies[k].pos - unit.pos))
                    a = FIRST_TARGET + k
            out.append(a)
        return out
