"""A unit-level hybrid world model (docs/generalisation/16_hybrid_model_plan.md).

Each unit has a position, health and an alive flag. One step (DT = 0.125 s):
1. target: Blue by its intent's target rule (nearest: nearest enemy in reach, else nearest; focus: the
   team's weakest enemy any Blue unit can reach, else nearest), Red its nearest enemy;
2. mode: walk or stop, from a switch classifier fitted on recorded units (docs/generalisation/15);
3. walk: speed SPEED; heading towards the target (advance, hold, Red), away from the nearest enemy
   (withdraw), kite: away while reloading, else towards;
4. damage: a unit whose target is within its reach (and is not withdrawing) deals -k * hp_target_type per
   second, k the frozen pipeline's firepower for that attacker and target type;
5. death at 0 health. Not modelled: collision avoidance, Red's target stickiness.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from sindy_wm.models.intent_model import DT
from sindy_wm.states.typed import pair_reach, unit_stats

SPEED = 3.15
GATES = (-1.0, -0.5, 0.0, 0.5, 1.0)
GROUPS = ["1", "m", "m2", *[f"g{c:g}" for c in GATES], "ready", "ready*g0", "danger", "tir"]
CTRL_BLUE = ["1", "adv", "hold", "hold*contact", "wd", "focus", "kite"]
MAX_CD = {"MARINE": 0.61, "MARAUDER": 1.07}
KITE_READY = 0.2  # share of the cooldown left at which a kiting unit turns back to fire


def features(m, ready, danger, tir, contact, movement, target, side: str, ctrls, prev=None) -> np.ndarray:
    """The switch classifier's design (groups x controls), as in `generalisation/unit_geometry`."""
    n = len(m)
    mc = np.clip(m, -3, 10)
    G = {"1": np.ones(n), "m": mc, "m2": mc**2}
    for c in GATES:
        G[f"g{c:g}"] = ndtr((m - c) / 0.3)
    G["ready"] = ready
    G["ready*g0"] = ready * G["g0"]
    G["danger"], G["tir"] = danger, tir
    C = {"1": np.ones(n)}
    if side == "blue":
        C |= {
            "adv": (movement == "advance") * 1.0,
            "hold": (movement == "hold") * 1.0,
            "hold*contact": (movement == "hold") * np.asarray(contact, float),
            "wd": (movement == "withdraw") * 1.0,
            "focus": (target == "focus") * 1.0,
            "kite": (movement == "kite") * 1.0,
        }
    groups = list(GROUPS)
    if prev is not None:  # docs/generalisation/20, fix S: whether the unit walked in the previous step
        G["prev"], G["prev*tir"] = prev * 1.0, prev * tir
        groups += ["prev", "prev*tir"]
    Gm = np.column_stack([G[g] for g in groups])
    Cm = np.column_stack([C[c] for c in ctrls])
    return (Gm[:, :, None] * Cm[:, None, :]).reshape(n, -1)


class SwitchModel:
    """Walk/stop classifier: L1 logistic on standardised features, or a shallow tree."""

    def __init__(self, clf, mu=None, sd=None):
        self.clf, self.mu, self.sd = clf, mu, sd

    def __call__(self, X) -> np.ndarray:
        Z = X if self.mu is None else (X - self.mu) / self.sd
        return self.clf.predict(Z).astype(bool)


def separate(pos: np.ndarray, alive: np.ndarray, rad: np.ndarray, passes: int = 4) -> np.ndarray:
    """Push overlapping alive units apart until no two are closer than the sum of their radii
    (docs/generalisation/18): each pass moves both units of a pair half the overlap apart."""
    pos = pos.copy()
    n = len(pos)
    jitter = np.stack([np.cos(np.arange(n)), np.sin(np.arange(n))], 1) * 1e-3  # for exactly coincident units
    for _ in range(passes):
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        over = (rad[:, None] + rad[None, :]) - d
        ok = alive[:, None] & alive[None, :] & (over > 0) & ~np.eye(n, dtype=bool)
        if not ok.any():
            break
        unit = np.where(d[..., None] > 1e-9, diff / np.maximum(d[..., None], 1e-9), jitter[:, None, :])
        pos = pos + (np.where(ok, 0.5 * over, 0.0)[..., None] * unit).sum(1)
    return pos


def steer(pos, vel, pref, alive, rad, hp, ticks: int = 2):
    """Move units with SMAClite's own collision avoidance (ORCA, `smaclite.env.rvo2.rvo2`), `ticks` game ticks
    per step as in the game (docs/generalisation/20, fix R). pref: preferred velocities (N, 2)."""
    from types import SimpleNamespace

    from smaclite.env.rvo2.rvo2 import compute_new_velocity

    pos, vel = pos.copy(), vel.copy()
    dt = DT / ticks
    rmax = float(rad.max())
    for _ in range(ticks):
        objs = [
            SimpleNamespace(
                pos=pos[i].copy(),
                velocity=vel[i].copy(),
                pref_velocity=pref[i].copy(),
                radius=float(rad[i]),
                radius_sq=float(rad[i]) ** 2,
                max_velocity=SPEED,
                hp=float(hp[i]) if alive[i] else 0.0,
            )
            for i in range(len(pos))
        ]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        new = np.zeros_like(vel)
        for i in np.flatnonzero(alive):
            js = np.flatnonzero(alive & (d[i] <= rmax + rad[i] + 2 * SPEED * dt))
            new[i] = compute_new_velocity(objs[i], [(objs[j], float(d[i, j])) for j in js if j != i], [])
        pos = pos + np.where(alive[:, None], new * dt, 0.0)
        vel = np.where(alive[:, None], new, 0.0)
    return pos, vel


class HybridWorld:
    """`switch[(side, type)] = (predict_fn, ctrls)` with predict_fn(X) -> walk (bool);
    `k[(attacker, target)]` in health units per enemy-in-reach per second (negative, from the frozen
    pipeline); `wd` the withdraw radius."""

    separation = False  # docs/generalisation/18: no-overlap step after movement
    steering = False  # docs/generalisation/20, fix R: SMAClite's own collision avoidance (ORCA)
    sticky = False  # docs/generalisation/20, fix T: Red keeps its target as attack-move does
    sequence = False  # docs/generalisation/20, fix S: the switch also sees the previous step's mode

    def __init__(self, switch: dict, k: dict, wd: float, max_reach: float = 6.0):
        self.switch, self.k, self.wd, self.max_reach = switch, k, wd, max_reach

    @staticmethod
    def k_from_pipeline(model) -> dict:
        out = {}
        for (_side, t), c in model.coef.items():
            for a in model.L.types:
                out[(a.upper(), t.upper())] = float(c.get(f"R[{a}]*s[{t}]", 0.0))
        return out

    def rollout(self, state: dict, intents: list[tuple[str, str]], alive_given=None) -> dict:
        """state: pos (N, 2), hp (N), alive (N), blue (N bool), type (N str), cd (N). intents: Blue's
        (movement, target) per step. Returns per step: hp_blue, hp_red (sums), reach_blue, reach_red
        (counts). alive_given (steps + 1, N): recorded alive flags; units then leave at their recorded
        deaths (geometry only, docs/generalisation/16 H1)."""
        pos, hp = state["pos"].astype(float).copy(), state["hp"].astype(float).copy()
        alive, blue, typ = state["alive"].copy(), state["blue"], state["type"]
        cd = state["cd"].astype(float).copy()
        n = len(hp)
        vel = np.asarray(state.get("vel", np.zeros((n, 2))), float).copy()
        prev_walk = np.linalg.norm(vel, axis=1) > 2.9
        prev_target = np.asarray(state.get("target", np.full(n, -1)), int).copy()
        R = np.array([[min(pair_reach(a, b), self.max_reach) for b in typ] for a in typ])
        hp_type = np.array([unit_stats(t)["hp"] + unit_stats(t)["shield"] for t in typ], float)
        maxcd = np.array([MAX_CD[t] for t in typ])
        rad = np.array([unit_stats(t)["radius"] for t in typ], float)
        kmat = np.array([[self.k.get((a, b), 0.0) for b in typ] for a in typ])
        out = {k: [] for k in ("hp_blue", "hp_red", "reach_blue", "reach_red")}
        idx = np.arange(n)
        for step, (mv, tg) in enumerate([*intents, intents[-1]]):
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            enemy = (blue[:, None] != blue[None, :]) & alive[None, :] & alive[:, None]
            inreach = enemy & (d <= R)
            reach_now = inreach.any(1) & alive
            out["hp_blue"].append(hp[blue & alive].sum())
            out["hp_red"].append(hp[~blue & alive].sum())
            out["reach_blue"].append(int(reach_now[blue].sum()))
            out["reach_red"].append(int(reach_now[~blue].sum()))
            if step == len(intents) or not (alive & blue).any() or not (alive & ~blue).any():
                for k in out:
                    out[k] += [out[k][-1]] * (len(intents) - step)
                break
            dn = np.where(enemy, d, np.inf)
            nearest = dn.argmin(1)
            marg = np.where(enemy, d - R, np.inf)
            m = marg.min(1)
            # targets
            target = nearest.copy()
            if mv != "withdraw":
                if tg == "focus":
                    reachable = inreach[blue].any(0) & ~blue & alive
                    if reachable.any():
                        f = np.flatnonzero(reachable)[hp[reachable].argmin()]
                        can = blue & inreach[:, f]
                        target[can] = f
                else:
                    cand = np.where(inreach, d, np.inf)
                    has = blue & np.isfinite(cand).any(1)
                    target[has] = cand[has].argmin(1)
            if self.sticky:  # Red keeps a live target that is in reach or attacking it
                keep = (~blue) & (prev_target >= 0)
                pt = np.where(keep, prev_target, 0)
                keep &= alive[pt] & (inreach[idx, pt] | (target[pt] == idx))
                target = np.where(keep, pt, target)
            prev_target = np.where(alive, target, -1)
            tir = inreach[idx, target] & alive
            contact = bool((m[blue & alive] <= 0).any())
            ready = cd / maxcd
            danger = (dn.min(1) < self.wd) * 1.0
            walk = np.zeros(n, bool)
            for (side, t), (fn, ctrls) in self.switch.items():
                sel = alive & (blue if side == "blue" else ~blue) & (typ == t)
                if sel.any():
                    mvs = np.full(sel.sum(), mv if side == "blue" else "advance", dtype=object)
                    tgs = np.full(sel.sum(), tg if side == "blue" else "nearest", dtype=object)
                    X = features(
                        m[sel],
                        ready[sel],
                        danger[sel],
                        tir[sel] * 1.0,
                        contact,
                        mvs,
                        tgs,
                        side,
                        ctrls,
                        prev=prev_walk[sel] * 1.0 if self.sequence else None,
                    )
                    walk[sel] = fn(X)
            # damage (simultaneous)
            fire = tir & ~(blue & (mv == "withdraw"))  # withdrawing Blue units do not fire
            dmg = np.zeros(n)
            np.add.at(dmg, target[fire], -kmat[fire, target[fire]] * hp_type[target[fire]] * DT)
            # movement
            to_t = pos[target] - pos
            to_n = pos[nearest] - pos
            head = to_t.copy()
            if mv == "withdraw":
                head[blue] = -to_n[blue]
            elif mv == "kite":
                back = blue & (ready > KITE_READY)
                head[back] = -to_n[back]
            nrm = np.linalg.norm(head, axis=1, keepdims=True)
            step_v = np.where(nrm > 1e-9, head / np.maximum(nrm, 1e-9), 0.0) * SPEED * DT
            if self.steering:
                pref = np.where((walk & alive)[:, None], step_v / DT, 0.0)
                pos, vel = steer(pos, vel, pref, alive, rad, hp)
            else:
                pos = pos + np.where((walk & alive)[:, None], step_v, 0.0)
                if self.separation:
                    pos = separate(pos, alive, rad)
            prev_walk = walk & alive
            hp = np.maximum(hp - dmg, 0.0)
            alive = alive & (hp > 0)
            if alive_given is not None:
                alive = alive_given[min(step + 1, len(alive_given) - 1)].copy()
            fired = tir & (cd <= 0) & ~((blue) & (mv == "withdraw"))
            cd = np.where(fired, maxcd, np.maximum(cd - DT, 0.0))
        return {k: np.array(v[: len(intents) + 1], float) for k, v in out.items()}


def state_from_snapshot(snap) -> dict:
    """Unit-level state from one step's snapshot rows (all units of the battle)."""
    s = snap.sort_values(["team", "unit_id"])
    index = {k: n for n, k in enumerate(zip(s.team, s.unit_id, strict=True))}
    tgt = [
        index.get((str(t), int(i)), -1) if i >= 0 else -1
        for t, i in zip(s.target_team, s.target_id, strict=True)
    ]
    return {
        "vel": s[["vx", "vy"]].fillna(0).to_numpy(float),
        "target": np.array(tgt, int),
        "pos": s[["x", "y"]].to_numpy(float),
        "hp": (s.hp + s.shield).to_numpy(float) * s.alive.to_numpy(bool),
        "alive": s.alive.to_numpy(bool),
        "blue": (s.team == "blue").to_numpy(),
        "type": s.unit_type.to_numpy(object),
        "cd": s.cooldown.fillna(0).to_numpy(float),
    }


def state_from_game(game) -> dict:
    units = [(True, u) for _, u in sorted(game.agents.items())] + [
        (False, u) for _, u in sorted(game.enemies.items())
    ]
    pos_of = {id(u): n for n, (_, u) in enumerate(units)}

    def _vel(u):
        v = getattr(u, "velocity", None)
        return np.zeros(2) if v is None else np.asarray(v, float)

    return {
        "vel": np.array([_vel(u) for _, u in units], float),
        "target": np.array([pos_of.get(id(getattr(u, "target", None)), -1) for _, u in units], int),
        "pos": np.array([u.pos for _, u in units], float),
        "hp": np.array([u.hp + u.shield for _, u in units], float),
        "alive": np.array([u.hp > 0 for _, u in units]),
        "blue": np.array([b for b, _ in units]),
        "type": np.array([getattr(u.type, "name", str(u.type)).upper() for _, u in units], dtype=object),
        "cd": np.array([u.cooldown for _, u in units], float),
    }


class HybridPlannerPolicy:
    """The Phase B planner with the hybrid world model: every `replan` s, roll each candidate out for
    `commit` s, then advance/nearest to `horizon` s, from the live units; play the best predicted health
    margin (hp)."""

    def __init__(self, world, candidates, withdraw_radius: float, replan=1.0, commit=2.0, horizon=3.0):
        self.world, self.candidates, self.wd = world, tuple(candidates), withdraw_radius
        self.replan, self.commit, self.horizon = replan, commit, horizon

    def reset(self, seed=None):
        from sindy_wm.envs.intents import IntentPolicy

        self._step, self.log = 0, []
        self._current = IntentPolicy(*self.candidates[0], withdraw_radius=self.wd)

    def setting(self):
        return {"movement": self._current.movement, "target": self._current.target}

    def act(self, env):
        from sindy_wm.envs.intents import IntentPolicy

        game = env.unwrapped
        every = max(1, round(self.replan / env.seconds_per_step))
        if self._step % every == 0 and game.agents and game.enemies:
            st = state_from_game(game)
            nc, nh = round(self.commit / DT), round(self.horizon / DT)
            scores = []
            for c in self.candidates:
                r = self.world.rollout(st, [c] * nc + [("advance", "nearest")] * (nh - nc))
                scores.append(r["hp_blue"][-1] - r["hp_red"][-1])
            self.log.append(scores)
            self._current = IntentPolicy(*self.candidates[int(np.argmax(scores))], withdraw_radius=self.wd)
        self._step += 1
        return self._current.act(env)
