# One-off checks (25 Nov 2026)

Exploratory checks made while answering questions for the thesis; none was pre-registered. Data: train, val, the counterfactual
sets and planner seeds, all used before. **No held-out set was read and no frozen model was refitted.** Scripts:
`experiments/checks/` (run from the repository root with `uv run --extra explore python`). Full outputs: `logs/`.

## Model B (`b_reach.py`, marines val)

- Least-squares k over 4 s, on measured reach: 0.183 (Blue's loss), 0.189 (Red's loss, windows without withdraw). On A's own
  predicted reach: 0.161 / 0.158. B's fitted k: 0.156 / 0.165.
- A predicts +0.66 (Blue) and +0.32 (Red) units too many in reach at 4 s.
- Health error at 4 s, own closure: A 0.055, B 0.052. With the measured share: A 0.032, B 0.040. Margin R²: 0.07 / 0.28 and
  0.64 / 0.65.
- Mean signed health error at 4 s: A −0.011 / −0.016, B +0.022 / +0.013 (Blue / Red).
- Reading (hedged): B's lower k fits the model's noisy, too high reach; it is not a better firepower estimate. The
  weak-form check in `sindyc_07` points the same way.

## B, A and S as planners (`b_planner.py`, `b_planner_read.py`, `b_ranking.py`)

S1 marines, seeds 21 000 000 to 21 000 399 (as docs/planning/35 and 37), 3 s horizon, 2 s commit. The frozen planner reproduces
its recorded wins exactly. The planner's rows are in `data/processed/planning/b_planner_S1.csv` (not in git).

| | wins | minus frozen (95%) |
|---|---|---|
| frozen | 0.652 | |
| A | 0.650 | −0.003 [−0.013, +0.005] |
| B | 0.645 | −0.007 [−0.020, +0.003] |
| S | 0.642 | −0.010 [−0.045, +0.025] |

Each of A, B and S loses to fixed hold/focus (about −0.04) and kite/nearest (about −0.32).

Intent ranking (522 states, regret at 1 / 2 / 4 s): frozen 0.0170 / 0.0225 / 0.0308, A 0.0171 / 0.0230 / 0.0316,
B 0.0167 / 0.0230 / 0.0314, S 0.0097 / 0.0164 / 0.0301, always advance/nearest 0.0052 / 0.0109 / 0.0267.
S is below frozen at 1 and 2 s (paired intervals below 0), not at 4 s.

## The withdraw term (`withdraw_term.py`, `withdraw_shared.py`, train)

- Fitted with a withdraw candidate at threshold 0.15, STLSQ keeps it for marines (0.88 of the fire lost) and marauders (0.80),
  not in mixed or triple, where each attacker type's term is 0.02-0.12 in effect size.
- Forced in by least squares, the fraction lost is 0.72-0.85 in mixed and 0.66-0.97 in triple.
- Measured firing per unit in reach under withdraw: 0.08-0.17, otherwise 0.87-0.93.
- One shared term per equation passes 0.15 for marines, marauders and mixed marines (0.18); it fails for mixed marauders (0.14)
  and all triple equations (0.11-0.12).
- The extra terms that return at lower thresholds are the **attacker's** alive counts.

## Shields (`shield_rises.py`, `shield_regen.py`, exp07 train)

- Stalker health rises in 37-40% of steps in contact (97-99% before it, 50-56% later with no enemy in reach); regained health
  is 9% of the damage.
- A regeneration term (own alive, or own missing health) has the wrong sign or an effect size of 0.05 or less. Only firepower is
  kept at 0.15. k is 0.86 / 0.88 of the game's, in line with marines.

## Medivac (`healing_size.py`, `medivac_battle.py`)

- A side loses 32-45 hp/s while engaged; 9 hp/s of healing is 20-28% of that.
- As a term it would have an effect size of 0.12-0.33 if it all lands on one type, 0.06-0.16 if half does (upper
  bound: healing switching on and off with contact).
- One MMM battle (seed 1): both medivacs start at 50 energy and drain 2.4/s while healing; Red's medivac is shot at the end.
