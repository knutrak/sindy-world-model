# RQ5 so far: planning with the world model (consolidated results, 21 Nov 2026)

A summary of the planning work of 20-21 Nov, for writing up. Each line points to the document with the full tables.
Results only; the interpretation is Knut's. Pre-registered steps are marked **(pre-reg.)**; the rest is exploratory.

## 1. Is there anything to gain? (Phase A, `00`, `01`) (pre-reg.)

The exact oracle plans with copies of the real game (re-plan 1 s, commit 2 s, horizon 8 s, five candidates). Wins,
paired on 200 seeds:

| | best fixed intent | exact oracle | perturbed oracle* |
|---|---|---|---|
| S1 marines | 68.5% hold/focus | 94.0% | 86.5% |
| S2 mixed | 60.2% advance/focus | 96.5% | 86.0% |
| S3 range asymmetry | 54.5% hold/focus | 98.0% | 93.5% |
| S4 underdog | 10.5% hold/focus | 65.0% | 46.5% |
| S5 uneven | 55.2% advance/focus | 79.0% | 70.5% |

\* Each candidate scored in its own perturbed copy; a rerun with one draw per re-plan shared by the candidates is in
section 9.

## 2. The world-model planner (Phase B, `07`, `08`) (pre-reg.), and two replications

Frozen pipeline model, horizon 3 s, commit 2 s, re-plan 1 s. Wins against Phase A's best fixed intent, 400 paired
seeds per set:

| | Phase B (14 x00 000) | replication 1 (15 x00 000, `10`) | replication 2 (16 x00 000, `14`) |
|---|---|---|---|
| S1 | −3.8 [−7.0, −0.5] loses | −2.2 [−5.5, +1.0] ties | −4.0 [−7.2, −0.7] loses |
| S2 | +2.2 [+0.3, +4.3] beats | −1.0 [−2.8, +0.7] ties | +0.2 ties |
| S3 (flagged) | **+16.8 [+11.5, +21.7]** | **+13.5 [+8.3, +18.8]** | **+15.0 [+10.0, +20.3]** |
| S4 (flagged) | −1.2 ties | +0.2 ties | −1.5 ties |
| S5 | 0.0 ties | −0.5 ties | +0.2 ties |

Share of the exact oracle's headroom captured (Phase B): S1 −0.25, S2 +0.08, S3 +0.43, S4 −0.03, S5 0.00.
The planner chooses focus intents in 81-94% of re-plans; horizons 2 s and 4 s give the same read-outs as 3 s.

## 3. Single decisions on the counterfactual sets (`02`-`06`)

- The frozen model ranks intents **worse than always advance/nearest over 1-2 s** (S1 regret 0.017 vs 0.005) and ties
  the best fixed intent at 8 s; it picks focus in about 98% of states.
- **Focus has a short-term cost the model lacks:** fewer Blue units fire at their own target (−11% in S1); damage
  per firing unit unchanged (`03`). A focus-cost term fixes the 1-2 s ranking, but the model then acts as always
  advance/nearest (`04`).
- Other scores at the horizon (alive margin, health + alive, square law) do not help (`03`). A perfect model at 2 s
  would capture 19-29% of the hindsight gap in wins, at 4 s 36-55%.
- Intent terms in the state equations (focus in d', shape dynamics) do not help (`06`); the residuals line up with
  the intents most in Red's formation depth (`05`).

## 4. Hold and advance are twins in contact (`docs/planning`, session notes)

The scripts' hold closes in as soon as any Blue unit has an enemy in range (`envs/intents.py`). On the
counterfactual sets, in contact at the branch (90.5% of states in S1, 91.8% in S2), hold and advance play identical
4 s battles in 99.2% / 99.3% of states. **In contact, the five candidates are three distinct choices** (focus,
nearest, withdraw).

## 5. Alive-state experiments (`09`-`16`)

| variant | train | val rollouts | ranking (cf) | closed loop |
|---|---|---|---|---|
| spread state, evenness (`09`, `10`) (pre-reg.) | CV R² 0.196 → 0.201 | alive error −0.6 to −2.6% | unchanged | same outcome as frozen in 95-100% of battles |
| compartments, bolted on (`13`, `14`) (pre-reg.) | wounded → dead CV R² 0.41-0.70 | S1 worse at 2-4 s; S2 ≈ | unchanged | ties the frozen planner |
| compartments, discovered (`15`, `16`) (pre-reg.) | no compartment term in health; distance keeps w_blue | better only S2 at 1 s; worse S1 and 2-4 s | picks differ in 41-69% of states, same regret | not run (decision) |

Train-only checks (`11`, `12`): kills come mostly from units already wounded; with wounded relative to own start
health, focus creates far fewer new wounded per unit of damage and finishes them about twice as fast (S1 Red:
0.43 → 0.09 and 0.25 → 0.49); the previous second's focus effect is carried 45-74% by the wounded count. Training
battles start damaged (each side's units at one fraction, median 0.73).

## 6. A hold-position intent (`17`, `18`) (decision rule fixed before)

`stand` stays put and fires only at enemies in its own range. Distinct from advance in contact (73-90% of states),
but 2-26% wins as a fixed intent and no gain for the exact oracle (S1 0.0, S2 −1.0, S3 0.0, S5 +1.0 points) →
no recollection.

## 7. When the oracle withdraws (`19`), and the planner (`20`)

Exact oracle replayed with the state at every re-plan (5,412 decisions). Withdrawals are short (median 1-2 s) and
early, at first contact:

| median at withdraw vs other | S1 | S2 | S3 | S5 |
|---|---|---|---|---|
| time (s) | 3 vs 5 | 2 vs 6 | 3 vs 9 | 3 vs 8 |
| Blue units with an enemy in reach | 20% vs 71% | 35% vs 75% | 31% vs 71% | 25% vs 67% |
| Red formation depth | 0.58 vs 0.36 | 0.71 vs 0.45 | 0.63 vs 0.42 | 0.71 vs 0.45 |

Depth-3 trees separate withdraw from other decisions with balanced accuracy 0.66-0.71; when it withdraws, withdraw
beats the next-best intent by a median 24-33 hp (about 4% of both sides' health).

The model planner (Phase B battles, 200 per scenario): in S3 it withdraws in the first 4 s in 179 of 200 battles,
and those battles carry its gain (+17.9 vs hold/focus; +4.8 in the other 21). In S1 it does so in 6 of 200.

## 8. The perturbed oracle with shared noise (`29`)

One noise draw per re-plan shared by the five candidates' copies (Phase A's 200 seeds, paired): wins equal to the
independent-noise version within noise (−0.5 to +6.0 points, all intervals spanning 0), still 3-12.5 points below
the exact oracle, and +16 to +43 points above the best fixed intent (67-93% of the exact oracle's headroom). The
flat intent use is gone (hold 1-6% of steps, as the exact oracle). Even with copies of the game, the top two
candidates are within 1 hp in 61-73% of re-plans.

## 9. Counterfactual picks over distinct choices (`28`)

With hold and advance merged in contact, the model variants' picks differ from the frozen model's in 1-15% of states
(41-69% when the twins are counted apart); regret unchanged.

## 10. Formation and the S3 mechanism (`24`, exploratory)

- The frozen model's 4 s margin error grows with Red's formation depth at the start: S1 0.040 (compact quarter) to
  0.095 (strung-out quarter); S2 0.049 to 0.110. Feeding the true team-level shape (depth, width) at every step
  removes 5% of the error overall, 7-10% in the strung-out quarter.
- Early 2 s withdrawals (small counterfactual set, 150 seeds): in S3 they truly help (+0.019 margin over 3 s, better
  in 61% of states) and the model gets it right (+0.018); in S1 they truly hurt on average (−0.087, better in 18%)
  and the model is too pessimistic (−0.176, better in 0.7%), mostly because it expects more damage on Red while
  advancing than happens. A fitted share of fire kept while withdrawing (9-13%, train) moves S1 only to −0.165.

## 11. A confidence-gated planner (`22`, `23`) (pre-reg.)

Default = Phase A's best fixed intent; deviate only when the model predicts a gain above its typical error
(τ = 0.033 marines, 0.042 mixed). Fresh seeds 18 x00 000 +, 400 per scenario:

| | gated − best fixed | gated − ungated | ungated − best fixed | re-plans off the default |
|---|---|---|---|---|
| S1 | −1.5 [−2.8, −0.5] loses | +2.0 ties | −3.5 ties | 1.0% |
| S2 | +0.5 ties | 0.0 ties | +0.5 ties | 1.9% |
| S3 (flagged) | +8.0 [+4.5, +11.7] beats | −3.5 ties | +11.5 beats | 6.4% |
| S4 (flagged) | −0.2 ties | 0.0 ties | −0.2 ties | 3.3% |
| S5 | 0.0 ties | 0.0 ties | 0.0 ties | 2.9% |

## 12. A kite intent (`26`, `27`) (decision rule fixed before)

Shoot when the weapon is ready, step away while it reloads. Distinct from advance in contact in 95-97% of states.
As a fixed intent: S1 98.5%, S2 88.5%, S3 98.5%, S4 34.0%, S5 59.0% wins (Phase A's best fixed intents 68.5, 60.2,
54.5, 10.5, 55.2%; +3.8 to +44 points, all intervals above 0). The exact oracle with kite added: S4 +12 [+4, +20],
S5 +1 (n.s.). By the pre-set rule a full recollection with kite is worth it; it was not started (needs Knut's
decision). Caveat: SMAClite may make stepping back between shots cheaper than a more detailed simulator would.

## 13. Terminal value and withdraw firing (`25`, `30`) (pre-reg.)

Same seeds as section 11. Against Phase A's best fixed intent / against the ungated planner, win points:

| | terminal value (square-law projection at 3 s) | withdraw firing (Blue keeps 9-13% of fire) |
|---|---|---|
| S1 | −3.8 loses / −0.2 ties | −3.5 ties / 0.0 ties |
| S2 | −4.0 loses / −4.5 loses | +0.8 ties / +0.2 ties |
| S3 (flagged) | −5.2 loses / −16.8 loses | +10.5 beats / −1.0 ties |
| S4 (flagged) | −0.5 ties / −0.2 ties | −0.2 ties / 0.0 ties |
| S5 | −2.5 loses / −2.5 loses | 0.0 ties / 0.0 ties |

The terminal value nearly stops withdrawals (S3: 18% → 3% of re-plans); the withdraw-firing fix changes almost no
decisions.

## 14. Why kite wins (`31`, exploratory)

Pilot seeds 19 7x0 000, 100 per scenario and set-up. Kite's firing rule without the step back plays like advance;
the step back costs Red 0.05-0.17 of its firing duty while Blue's rises. A kiting Red beats an advancing Blue
(S1 Blue 66 → 34%, S2 65 → 27%), so kiting is strong for either side in SMAClite. With both sides kiting Blue still
wins 86-100%, not yet explained (Red's kite is implemented through commands, not an exact copy).

## 15. Kite in the world model (`32`, `33`) (pre-reg.)

Refit with kite as a fifth control (exp08 train + 2,000 new battles with kite, 6.6% of steps). STLSQ kept no kite
term in health; the distance equation got a kite gate equal to advance's. K1 (old movements unchanged) and K2
(kite error ≤ 1.5 × advance) pass; K3 fails: on the counterfactual sets with kite added, the variant ranks like
the frozen model and worse than always kite/focus (regret 0.052 vs 0.046 at 2 s, 0.086 vs 0.067 at 4 s). K4 not
run. The unthresholded kite terms: Red's fire per unit in reach barely changes under kite, Blue's rises 11-76%
(just below the threshold); the reach closure overpredicts Blue's reach on kite steps.

## 16. Closed loop with μ, σ reach states (`34`, `35`) (pre-reg.)

Planner with reach from μ, σ (`generalisation/11`, clipped to the training range), kite among its seven candidates in
mixed. Fresh seeds 21 x00 000 +, 400 per scenario. Against the best fixed kite: S2 −9.5, S4 −9.0, S5 −4.0 (all
lose); against the frozen planner: S2 +17.2, S4 +13.0 (beat), S1, S3, S5 tie. It picks kite in 42-53% of re-plans
and switches 6-7 times per battle. The pre-registered statement "planning with the world model is useful in this
testbed" does not hold.

