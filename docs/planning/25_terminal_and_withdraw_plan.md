# A terminal value and the withdraw-firing fix: pre-registration (22 Nov 2026, overnight run)

**Status: written before either arm is fitted, run or analysed, and before any result of the gated-planner test
(`22`) is read. Both arms are exploratory variants of the planner or the model; the frozen pipeline stays the
tested model.**

## Why

- **Terminal value.** The model is reliable for 2-3 s; decisions pay off over about 8 s (`21`, sections 2-3). A
  planner that scores only the margin 3 s ahead ignores what that state means for the rest of the battle.
- **Withdraw firing.** The pipeline's rule is that Blue does not fire while withdrawing (reach × (1 − wd)). On train,
  Blue units with an enemy in reach still fire at 9.3% (marines) and 12.7% (mixed) of their advance rate while
  withdrawing. On a small early-branch set (`24`), the model expects a 2 s withdrawal to cut the damage Red takes far
  more than it does in S1 (0.264 against 0.186), which makes it too pessimistic about withdrawing there.

## The arms

**`terminal:h3` (planner change, frozen model).** As the Phase B planner (horizon 3 s, commit 2 s, re-plan 1 s), but
each candidate is scored by a Lanchester square-law projection of the final margin from the model's state at 3 s,
instead of the margin at 3 s:

```
S_side = k_side · A_side · H_side          (fighting strength: firepower × units × health; team view)
V = H_B·sqrt(max(0, 1 − S_R/S_B)) − H_R·sqrt(max(0, 1 − S_B/S_R))
```

k_B = the mean coefficient of the reach terms in Red's health equations (Blue's fire on Red), k_R likewise from
Blue's; taken from the frozen model, no fitting (marines: k_B 0.1861, k_R 0.1770).

**`wfire:h3` (model change, otherwise frozen).** As the Phase B planner, with Blue's reach in Red's health loss
× (1 − wd·(1 − f)) instead of × (1 − wd), f = the measured firing fraction while withdrawing on train (0.093 marines,
0.127 mixed). Nothing else changes.

## Test

The seeds of the gated test (**18 x00 000 +**, 400 per scenario), where the ungated Phase B planner (`model:h3`) and
the six fixed intents are played; both new arms are added on the same seeds. Read-outs, wins, paired, bootstrap 95%:

- each arm − Phase A's best fixed intent;
- each arm − the ungated planner.

beats / ties / loses per scenario; main scenarios S1, S2, S5, with S3 and S4 flagged. Offline, also reported: on the
early-branch set of `24`, the wfire model's predicted effect of a 2 s withdrawal against the truth (S1, S3).

## Expectations (written before)

- terminal: small changes; a gain in S1 is possible (it values the end state, as hold/focus does), else ties.
- wfire: more early withdrawals in S1; ties or a small gain in S1, S3 unchanged or slightly better.

## Not allowed

Changing the projection, k, f or the arms after seeing these battles.
