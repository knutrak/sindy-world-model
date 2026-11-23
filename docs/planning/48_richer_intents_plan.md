# A richer decision space: per-type roles and target-type priority (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026, as drafted; rules fixed before any battle.**

## Why

The current intents limit what a planner can choose:
- hold equals advance in contact;
- micro (kite) is mixed in with team decisions;
- the whole team does the same thing, although the world model is per unit type;
- targeting is only nearest or focus.

This simulator-only check asks whether a decision space built around the per-type structure gives a planner **more
room** than the five team-level intents, before any world model is extended.

## The new choice set (27 combinations, mixed armies)

| part | options |
|---|---|
| marines' movement | advance, **stand** (a real hold: stay put, fire at what's in range), withdraw |
| marauders' movement | advance, stand, withdraw |
| target priority (both types) | nearest; **marines first** (nearest enemy marine in range, else nearest); **marauders first** |

That is 3 × 3 × 3 = 27 combinations. **No kite**: micro stays plain (shoot what the rule picks), so the planner
competes only with team- and type-level choices.

**Code:** a `RolePolicy` that gives each Blue unit the action of an `IntentPolicy` with its type's movement
(target nearest), then swaps an attack for the priority type's nearest enemy in range, if there is one. The oracle
gets a candidate list of `RolePolicy` combinations. Tests:
- with both types on advance and priority nearest, `RolePolicy` plays exactly as `IntentPolicy` advance/nearest;
- priority picks only enemies of that type when one is in range.

## Runs

| | what | seeds |
|---|---|---|
| fixed | all 27 combinations as fixed policies for the whole battle | `34`'s first 200 per scenario (21 x00 000 +), paired with the existing `oracle5-h3` and best fixed team-intent rows |
| `oracle27-h3` | exact oracle over the 27 combinations, the planners' settings (re-plan 1 s, commit 2 s, horizon 3 s) | the same 200 |

**Scenarios:** S2, S4, S5 (both sides mixed). **Run time:** timed on a smoke run first; estimated 30-60 min.

## Rules (fixed once approved)

| | quantity | how |
|---|---|---|
| **Room27** | `oracle27-h3` − the best of the 27 fixed combinations (picked per scenario on the same seeds, which favours the fixed side) | win points, paired 95% bootstrap |
| Room5 | `oracle5-h3` − the best fixed team intent (`43`), on the same 200 seeds | the same |
| **N1** | Room27 − Room5 | point estimate per scenario |
| N2 | does one fixed combination come within 5 points of the best fixed combination in all three scenarios? | yes / no, and which |
| N3 | `oracle27-h3` − `oracle5-h3` | paired 95% bootstrap: does the richer set raise the ceiling? |

**Statement, fixed now:** "the richer decision space gives a planner more room" holds if **N1 ≥ 5 points in at
least two of the three scenarios**. If it holds, extending the world model's controls to per-type roles and target
priority is worth planning. If not, the team-level intents already hold most of the room.

## Expectations (written before any battle)

- The best fixed combination is "marauders first" with both types advancing, or something close to it; it changes
  between S4 and the balanced scenarios.
- **N3:** the richer oracle beats `oracle5-h3` by 5-15 points.
- **N1:** holds in S2 and S4 (about 55%).
- **N2:** no single combination near-best everywhere (about 50%).

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | No kite at all, or kite as the fixed micro under every role? | no kite here; a kite-micro version can follow if the room is there |
| Q2 | 200 seeds per scenario (27 fixed policies + a slow oracle)? | yes |
