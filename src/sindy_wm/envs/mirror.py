"""A Red team that plays Blue's intents (docs/planning/40_mirrored_red_payoff_plan.md).

`MirrorRed(policy)` lets any Blue policy (`IntentPolicy`, `IntentSchedule`, ...) choose Red's actions.
The policy sees a mirrored view of the game: its own units are Red's, its enemies are Blue's, and each Red
unit gets an action mask built by SMAClite's rules for agents (stop always; a move if the walkable-terrain
check allows it; attack any alive Blue unit within AGENT_TARGET_RANGE = 6). The chosen action numbers become
Red commands as SMAClite turns agent actions into commands: stop -> StopCommand, move -> MoveCommand
MOVE_AMOUNT units that way, attack k -> AttackUnitCommand on Blue unit k. Call `apply(env)` before every
`env.step`, so Red decides at the same steps as Blue.
"""

from __future__ import annotations

import numpy as np
from smaclite.env.smaclite import AGENT_TARGET_RANGE, MOVE_AMOUNT
from smaclite.env.units.unit_command import AttackUnitCommand, MoveCommand, StopCommand
from smaclite.env.util import point_inside_circle
from smaclite.env.util.direction import Direction

FIRST_TARGET = 6


class _MirrorGame:
    """The game as Red sees it: `agents` = Red units, `enemies` = Blue units, masks for Red."""

    def __init__(self, game):
        self._game = game

    @property
    def agents(self):
        return self._game.enemies

    @property
    def enemies(self):
        return self._game.agents

    def get_avail_actions(self) -> list[np.ndarray]:
        g = self._game
        n = FIRST_TARGET + g.n_agents
        dead = np.zeros(n)
        dead[0] = 1
        out = []
        for i in range(g.n_enemies):
            unit = g.enemies.get(i)
            if unit is None or unit.hp <= 0:
                out.append(dead)
                continue
            m = np.zeros(n)
            m[1] = 1
            for d in Direction:
                m[2 + d.value] = g._SMACliteEnv__can_move(unit, d)
            for k, target in g.agents.items():
                if target.hp > 0 and point_inside_circle(target.pos, unit.pos, AGENT_TARGET_RANGE):
                    m[FIRST_TARGET + k] = 1
            out.append(m)
        return out


class _MirrorEnv:
    def __init__(self, env):
        self.unwrapped = _MirrorGame(env.unwrapped)
        self.seconds_per_step = env.seconds_per_step


class MirrorRed:
    """Red plays `policy` (a Blue policy) through the mirrored view."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, seed: int | None = None) -> None:
        # Caution: with a random policy (IntentSchedule), the battle's seed would give Red the same schedule
        # as Blue. Pass Red its own stream (e.g. seed + an offset) before collecting data with a switching
        # Red.
        self.policy.reset(seed)

    def apply(self, env) -> None:
        game = env.unwrapped
        actions = self.policy.act(_MirrorEnv(env))
        for i, a in enumerate(actions):
            unit = game.enemies.get(i)
            if unit is None or unit.hp <= 0 or a == 0:
                continue
            if a == 1:
                unit.command = StopCommand()
            elif 2 <= a < FIRST_TARGET:
                unit.command = MoveCommand(unit.pos + Direction(a - 2).dx_dy * MOVE_AMOUNT)
            else:
                unit.command = AttackUnitCommand(game.agents[a - FIRST_TARGET])
