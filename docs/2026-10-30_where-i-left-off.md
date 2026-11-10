# Where I left off (30 Oct 2026)

## The story so far, in one breath

SINDy on team health alone predicts, but its terms don't mean anything. The breakthrough was the
right variable, **units within reach** (an enemy closer than 5.75, counted from positions). Given
health, alive and reach, SINDy keeps one term per side:
`(health_blue)' = -0.176 reach_red`, `(health_red)' = -0.183 reach_blue` (80% / 84% of a marine's
maximum firepower, R² 0.73). Reach in turn is an S-curve in the **distance** between the team
centres and each team's **spread** (depth). Its switch lands at 5.76, against the game's 5.75.

**The catch:** every good number so far reads the *real future* distance, spread and alive from the
recording. Health error 4 s ahead:

| | error |
|---|---|
| real reach given (best case) | 0.031 |
| reach from real distance + spread | 0.044 |
| health only, no future data | 0.065 |

So the open problem is predicting **distance, spread and alive** themselves.

## Where things are

- `notebooks/engagement_from_distance.ipynb`: all of the above, sections 1–11, run on 5,000 battles.
  Section 9 = reach law, section 11 = spread closure. The "Next" cell at the end lists open items.
- `notebooks/walkthrough_exp_05.ipynb`: health-only models, Lanchester, first exp05 fits.
- `docs/findings.md`: numbered findings F1–F26. **Doesn't have the reach/spread results yet.**
- Supervisor deck: https://claude.ai/artifact/UEQDbvxXaHPdbLKRawKTiG (9 slides, notes have the detail).
- Working tree was clean; last commit `a4b4034`.

## Supervisor feedback (fill in)

- RQ1 sharpened?
- Movement first?
- Complete model before other unit types?

## Next, in order

1. ~~**Fair test (quick)**~~ **Done 10 Nov, section 12.** With nothing read from the future, reach from
   the spread closure with distance and depth extended along their last-second trend, and alive
   scaled by predicted health, gives **0.057** at 4 s (0.033 at 2 s), against 0.065 (0.039) for
   health only. Holding geometry constant is worse (0.084-0.092); trend without deaths blows up
   (0.117). The offset trick doesn't help.
2. ~~**Equations for distance, spread and alive**~~ **Done 10 Nov, section 13.** SINDy keeps
   `(distance)' = -5.78 gate + 2.18 timid*gate` (timid = 1 - aggression; gate switches off at
   centroid distance 8.3). Depth can simply be held, and alive follows predicted health. Whole
   battles simulated from 1 s in with nothing read from the data: health error 0.059 (0.053 with real
   distance), winner right 92%. Distance error 0.29, against 3.46 for the trend. **The remaining
   error is in contact:** the reach closure underestimates damage at first contact (0.044 against
   0.031 with real reach), and alive-from-health is crude.
   - Next candidates: improve reach in contact (per-unit geometry, or reach as a state with a
     better equation), a real alive equation, and RQ-level metrics (winner accuracy is weak: 90.4%
     from starting health alone).
2b. **Done 10 Nov, section 14: reach in contact.** The closure's late-battle excess came from
   battles where Blue retreats. Added, all fitted on training battles: alive = A0·(H/H0)^0.58
   (damage is spread, F6); a retreat share rho from Blue's `team_below` / `unit_below` triggers;
   retreating units don't fire (s_f = 0.81) and keep out of reach (reach R² 0.84 → 0.93); retreat
   stops closing. **Health error at 4 s: 0.040 with no future data** (health-only 0.065, old oracle
   0.031). Whole battles: 0.049 mean, winner 92.4%. Left: closure ~10% short in the first second
   of contact.
3. **Blue's movement as a control input:** collect battles where the setting changes mid-battle
   (`SwitchingPolicy(segment_seconds=(1, 3))`; exp05 used one setting per battle).
4. **Housekeeping:** add the reach and spread results to `docs/findings.md`; move `geometry`,
   `reach_features` and `in_reach_share` into `src/sindy_wm/` with tests.

Open questions noted along the way: does SINDy still pick reach with degree-2 candidates (not
tested)? The clean one-term law is at threshold 0.05; at 0.02 two small terms survive.
