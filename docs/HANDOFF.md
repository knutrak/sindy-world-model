# Handoff: where the work stands (22 Nov 2026; sections 7 and 8 are the latest)

A note for starting a new chat. It records **why** things were decided and **what is open**. The results
themselves are in the documents linked below. Branch `main`; everything is committed.

## 1. How to work with Knut

- **Sparring partner.** Ask questions about results and review Knut's drafts; don't hand over conclusions.
  Reports and analysis docs are fine when asked for.
- **Show the equations.** Before every model step, display the current equations with numbers in chat.
- **Ask first** before:
  - a large new data collection;
  - a change to the pipeline that breaks earlier results;
  - a dependency that is heavy to install.
- **No data leakage.** Derived states and normalisation are fitted on train only.
- **Git.** Knut does the pushing; give the commands to run. Explain each in the pattern "git restore … does xyz".
  `docs/final_exploration/results.csv` is re-saved by Knut: leave it uncommitted.
- **Use case.** A 1-2 s prediction horizon is enough.
- **Practical.**
  - Long runs need the laptop lid open; `caffeinate` doesn't stop sleep with the lid closed.
  - Never run multiprocessing code from stdin (`python -`): the pool keeps retrying in the background.
  - After stopping a background run, check that its shell loop is really gone.

## 2. The thesis, and where the work fits

Project description: NTNU/FFI, "master_Knut". The research questions:

| RQ | question | status |
|---|---|---|
| RQ1 | can SINDy identify battle dynamics | yes: interpretable equations, coefficients match the game |
| RQ2 | state representation | per-type states (health, alive, reach per side and unit type, plus distance) |
| RQ3 | predictive performance | beats persistence and Lanchester; reliable 2-3 s |
| RQ4 | applicability across scenarios | the generalisation pipeline, held-out test 19/20 |
| RQ5 | **key question:** does an LLM agent decide better with the world model | headroom shown (Phase A); the model planner ties or loses except S3 (Phase B); kite and S4 open (sections 7-8) |

- Phase 1 runs to the end of 2026. Step 7 is "package the model behind a clean world-model interface".
- FFI (§2.5) wants a **repeatable pipeline from simulator data to a world model**, usable on other simulators
  (e.g. VR-Forces).
- **Knut's decision (20 Nov): the generalisation pipeline is the main product, not a side project.**

## 3. The pipeline (main product)

**Read:**
- [`generalisation/REPORT.md`](generalisation/REPORT.md) — the report;
- [`generalisation/06_frozen_equations.md`](generalisation/06_frozen_equations.md) — the frozen equations;
- `notebooks/05_generalisation/generalisation_01.py` — the notebook.

**Code and config:**
- `src/sindy_wm/models/pipeline.py`;
- `configs/generalisation_pipeline.yaml` (frozen);
- state tables: `scripts/build_typed_states.py`.

**Held-out test:** pre-registered and run once, 19 of 20 expectations met.

- The one miss: in the uneven scenario's refit, marine → marauder firepower came out at 1.02 × the game rate.
- **All four held-out sets are spent:** marines `test4`, marauders `test`, mixed `test3`, uneven `exp10`.
  Never evaluate on them again.

**Decisions and why:**

| decision | why |
|---|---|
| reach capped at 6 (SMAClite's `AGENT_TARGET_RANGE`) | units only fire within 6 regardless of weapon range; without the cap, marauder firepower came out at 57-64% of the game, with it at about 92% |
| withdraw *firing* as a rule, R·(1 − wd) | the scripts don't fire while withdrawing; the fitted alternative tied and adds numbers |
| withdraw *movement* as a rule (danger·wd, walk_R·wd, the scripts' withdraw radius) | the fitted gate was fragile on small sets (closing speed 12.8 on marauders); Knut chose the rule; recorded as an amendment before any held-out read |
| target choice proportional to type shares | a fitted closure was no better in rollouts and adds 12 numbers |
| per-type states in mixed armies | beat the team-level model, because the composition changes during a battle |
| the pipeline's marine model replaces A*-2k (D4) | identical health terms; differs in Red's alive p (+4%) and the withdraw rules |

**Inputs the pipeline is given** (the wording in the report and notebook was corrected on 20 Nov):
- hp, attack range and size per unit type, plus the firing limit 6;
- the scripts' withdraw radius, a setting.

Damage, cooldown, armor, the armored bonus and speed are used only afterwards, in the physics check.

**Weak parts:**
- the per-type alive law (p 0.38-0.58);
- the advance and hold gates;
- Red's marine reach closure;
- not a long-range predictor (an MLP was clearly better at 4 s on marines).

## 4. RQ5: the headroom test

**Read:**
- [`planning/00_headroom_plan.md`](planning/00_headroom_plan.md) — the plan, approved 20 Nov;
- [`planning/01_headroom.md`](planning/01_headroom.md) — the results, with notes on the run.

**Code:**
- `experiments/planning/headroom.py`, `analyse.py` and `reseed_check.py`;
- `src/sindy_wm/envs/planner.py`.

**Why the test was needed:**
- The earlier closed loop (marines, 10 Nov): the SINDy planner won 69.8% against 67% for always advance/focus,
  and the oracle won 94%.
- The open question: is that headroom real, and is it good choices or perfect foresight?

**The oracle** plans with copies of the real game:
- every 1 s it tries 5 candidate intents;
- each candidate is played for 2 s, then advance/nearest up to 8 s ahead;
- it plays the candidate with the best health margin.

The *perturbed* oracle moves positions by ±0.2 and changes health by ±5% in each copy, a stand-in for a good
but imperfect model.

**Phase A result:** all five scenarios GO on O1 (win). The "gain" is the perturbed oracle minus the best fixed
intent, paired on 200 seeds:

| | best fixed | perturbed oracle | exact oracle | gain [95%] |
|---|---|---|---|---|
| S1 marines | 68.5% hold/focus | 86.5% | 94.0% | +16.5 [+9.5, +23.0] |
| S2 mixed | 60.2% advance/focus | 86.0% | 96.5% | +21.5 [+15.5, +28.0] |
| S3 Blue marauders vs Red marines (range asymmetry) | 54.5% hold/focus | 93.5% | 98.0% | +37.5 [+31.0, +44.5] |
| S4 underdog | 10.5% hold/focus | 46.5% | 65.0% | +36.5 [+29.5, +43.5] |
| S5 uneven | 55.2% advance/focus | 70.5% | 79.0% | +14.5 [+9.0, +19.5] |

The foresight share is 0.1-0.4: most of the exact oracle's lead comes from choosing intents well, not from
knowing the future exactly.

**Checked after the run:**
- **Duplicate rows removed.** A stopped run overlapped its restart; the duplicates were identical.
- **Shared shuffle stream.** The perturbed copies shared the game's unit-order shuffle stream, a deviation
  from the plan. A rerun with separate streams on 100 seeds of S1 and S3 found no consistent effect.

**Open questions, to discuss before Phase B:**

1. **Flat intent use.** The perturbed oracle uses all five intents about 20% each, withdraw included. What is
   it doing when it withdraws? Its winner also changes in 17% of S1 battles with different noise draws.
2. **Is S4 an exploit?** It goes from 10% to 47% wins. Is that tactics, or does the oracle exploit the
   scripted Red AI, which runs inside its copies?
3. **O3 is trivially won by always withdrawing.** That keeps 79-92% of Blue's units but never wins a battle.
   O3 needs pairing with O1.
4. **The survival oracle can't value a sustained withdraw,** because advance/nearest follows each 2 s
   commit. Its O3 result for S4 says little. Fixing it means a rerun.
5. **Horizon gap.** The oracle looks 8 s ahead; the model is reliable for 2-3 s. What share of the headroom
   can a short-horizon model planner capture?

**Phase B (needs Knut's own approval):**
- `ModelPlannerPolicy` with the frozen pipeline models (S1 the marine model, S2-S5 the mixed model);
- horizon 2-4 s, commit 1-2 s, re-plan every 1 s;
- new seeds, compared against the fixed intents and both oracles;
- the result is the share of the headroom the model captures, the evidence RQ5 needs before an LLM is added.

## 5. Other open threads

- **The Phase 1 world-model interface (step 7)** and making the parameters less game-specific: offered as a
  design doc, not started.
- **From the generalisation report:**
  - why the uneven refit gives 1.02;
  - a better per-type alive law;
  - does the reach cap explain the stalker transfer loss;
  - inferring Red's hidden mode from d' (Knut's idea).

## 6. Where things are

| what | where |
|---|---|
| rated findings across the project | [`2026-11-20_findings-summary.md`](2026-11-20_findings-summary.md) |
| earlier final model A*-2k | [`final_exploration/REPORT.md`](final_exploration/REPORT.md) |
| marauders, mixed armies, the reach cap | `mixed/PROGRESS.md` |
| notebooks, grouped by stage | `notebooks/README.md` |
| headroom data (not in git) | `data/processed/planning/headroom_S*.csv`, `*_reseed.csv` |
| tags | `final-exploration-2026-11-19`, `generalisation-test-2026-11-21` |

## 7. Update after 21-22 Nov

**(Resolved 22 Nov, see section 8: a small kite collection and refit were run instead.)** The decision then
was a full recollection with the kite intent. By the rule fixed before its pilot (`26`,
`27`), kite justifies one: as a fixed intent it wins 98.5% (S1), 88.5% (S2), 98.5% (S3), 34% (S4), 59% (S5),
beating every earlier fixed intent and, in S1, the old exact oracle (94%). It shifts every RQ5 baseline, and SMAClite
may make kiting cheaper than a more detailed simulator would, so it was not started.

**Where the results are:** `docs/planning/21_rq5_report.md` collects everything from Phase A on, with pointers;
plans and results are `docs/planning/09`-`30` (pre-registered ones say so at the top).

**What is now known (details in `21`):**
- In contact, hold and advance are the same battle (99%): five candidates are three distinct choices.
- The planner's choices are mostly near-ties; even the game's copies put the top two within 1 hp in 61-73% of
  re-plans. Changing the alive law (spread, compartments) changes which twin is picked, not the outcome.
- The exact oracle's key move is a short step back at first contact while Red is strung out. The model planner
  does it in S3 (its +13 to +17 points, replicated three times) and almost never in S1 (2-4 points below hold/focus).
- The model's error grows with Red's depth (up to 2.4×), but the true team-level shape removes only 5-10% of it:
  the missing information is finer than depth and width.
- Tried and not helping the planner: focus cost, alive-weighted scores, shape dynamics, spread state, compartments
  (bolted on and discovered), a confidence gate (S1 −1.5, S3 +8 vs fixed), a square-law terminal value (worse), the
  withdraw-firing share (ties). Hold position (`stand`) adds nothing.
- Phase A's headroom holds with shared noise (`29`).

**Code:** optional hooks in `pipeline.py` (focus_cost, intent_extra, spread_model, compart_model, withdraw_fire,
shape_traj; all off by default), `models/discovered.py`, `states/spread.py`, `states/compartments.py`, planners in
`envs/planner.py` (PipelinePlannerPolicy, GatedPlannerPolicy, `terminal`), intents `stand` and `kite` outside
`INTENTS`. The frozen pipeline is unchanged and stays the tested model.

**Working rules added:** keep experiments short and state run times first; data from seeds 13-18 x00 000 + and
19 x00 000 + is used (see the plans for which).

## 8. Update 22 Nov: the kite intent (read this first)

**Where the results are:** `docs/planning/31` (mechanism), `32` (refit plan, pre-registered, approved as drafted),
`33` (refit results), and `21` sections 14-15.

**Why kite wins (`31`, exploratory, 100 seeds per set-up, pilot seeds 19 7x0 000):**
- The gain is the step back. Kite's firing rule alone (shoot only inside own range, no step back) plays like
  advance.
- Stepping back costs Red 0.05-0.17 of its firing duty: Red's weapons are ready with no Blue unit in range more
  often, because Red walks after Blue. Blue's duty rises.
- Kiting is strong for either side in SMAClite: a kiting Red beats an advancing Blue (S1 Blue 66 → 34%, S2 65 → 27%).
- Not explained: with both sides kiting Blue still wins 86-100%. Red's kite is built from SMAClite commands, not
  an exact copy of Blue's, so this row is not trusted yet.

**What Knut said, and what it decided:**
- RQ5 does not fade; its baseline becomes the strongest simple rule (kite). Knut: beating uneven matches (S4) with
  the world model and planner would be a result in itself.
- S4 so far (`01`, `27`): best fixed 10.5% (hold/focus), fixed kite 34%, exact oracle 65%; on 100 seeds the oracle
  with kite added wins 79% against 67%. The model planners have captured about 0% of S4's headroom. S4 is outside the
  training strength range (0.6-0.8 vs 0.85-1.18).
- Withdraw when weak (`RetreatPolicy`) was only tested in October (F26: 15% vs 63% wins, 60 battles), never in
  S1-S5 or as an oracle candidate.

**The kite refit (`32`, `33`, pre-registered):**
- Data exp11 (mixed only): 4 × 500 train battles with kite in the schedule (weights 0.35 / 0.20 / 0.15 / 0.30),
  300 val, and `cf_kite_nearest`, `cf_kite_focus` on exp08's cf seeds. Kite was 6.6% of training steps with exp08.
- **No kite term survived in the health equations**, even at the halved threshold; the distance equation got
  close·kite −5.47, equal to advance's −5.45. The model predicts kite as advance.
- K1 passes (old movements unchanged), K2 passes (kite error 1.16 × advance at 2 s), **K3 fails** (regret 0.052 vs
  0.046 for always kite/focus at 2 s, 0.086 vs 0.067 at 4 s), so **K4 (closed loop, S4 main) was not run**.
- Diagnostic (no threshold): Red's fire per Red unit in reach barely changes under kite (+5 to +8%), Blue's rises
  11-76% (standardised 0.022-0.063, just below 0.075). The reach closure overpredicts Blue units in reach on kite
  steps (0.38 vs 0.32 measured). A kite candidate is the true best in 51-69% of counterfactual states.

**Open questions for Knut (not decided):**
1. The kite effect seems to sit in reach, which is a closure, not in the sparse health terms. Is a kite term in the
   reach closure still discovery, or a hand-built fix?
2. The Red-health kite terms were just below the threshold. Would a rerun with more kite data be a fair new
   pre-registered variant, or tuning until it passes?
3. Should the new kite data use underdog strength ratios, so S4 is inside training? Deferred in `32` (Q2): two
   changes at once can't be told apart.
4. Can a 1 s team-level model represent a sub-second per-unit tactic at all?
5. Should the opponent be part of the design: a Red that attack-moves and one that kites?

**Code:** `models/kite.py` (`KitePipeline`, `CTRL_K`, `add_kite`, `control_vector_k`); hooks in `pipeline.py` with
unchanged defaults (`batch(ctrl=)`, `ctrl_cols`, `_health_stlsq`, `_fit_extra_gates`, `self._u` per rollout step;
`tests/test_kite_variant.py` checks the frozen path is unchanged); `IntentSchedule` takes a fourth weight (kite);
`PipelinePlannerPolicy` uses the model's own `control_vector` if it has one; `scripts/collect_kite.py`;
`experiments/planning/kite_mechanism.py`, `kite_refit.py`. Kite model: `data/processed/planning/kite_model.pkl`.

**Seeds:** 19 700 000-19 720 099 used (mechanism check); exp11 uses 20 000 000-20 001 999 (train) and
20 100 000-20 100 299 (val); 21 x00 000 + was reserved for K4 and is unused.

**Commits:** `a65cf01` (kite code and plan) and `1724f8c` (refit results), not pushed: Knut pushes.

**Reach as a state from a generic library (`generalisation/07`, `08`, pre-registered, 22 Nov evening):** degree 2
and 3 polynomials of the state, distance gates on a fixed grid, deaths, all × the five intents; two-level STLSQ in
weak form (and one-step). R2 fails everywhere (54-232 groups kept, ≤ 8 asked; 2-22% blow-ups), R3 fails (reach
better at 1-2 s by up to 14%, not at 4 s). R4: given the closure as a feature, the kite reach bias drops from
+0.30 to under 0.1 units. By the plan: no sparse reach equation found from a generic library; the closure stays.
Code: `models/reach_state.py`, `Pipeline._reach` hook (default unchanged), `experiments/generalisation/reach_state.py`.

**μ, σ reach states (`generalisation/09`-`11`, 22 Nov night):** each unit's reach margin (distance to the nearest
enemy minus pair reach); its mean μ and spread σ per side and type. Feasibility (`09`, exploratory): with measured μ,
σ, reach error is 35% (marines) and 50% (mixed) lower than the closure. Model (`10` pre-registered, `11` results):
generic library × intents, weak form, with and without p. M1 (sparse) fails: 26-51 groups. M2 (rollouts) fails:
reach better at 1-2 s, blow-ups 12-18%. M3 (kite bias) fails. **M4 (intent ranking) passes:** regret below frozen at
2 and 4 s in both scenarios, and below always kite/focus at 2 s in mixed; always advance/nearest still better at
1-2 s in marines. By the plan: no pipeline v2; a closed-loop planner test is the next step (not planned yet).
Code: `states/margins.py`, `scripts/build_margin_states.py`, `models/mu_sigma.py`,
`experiments/generalisation/mu_sigma.py`, `reach_distribution.py`.

**Closed loop with μ, σ (`planning/34`, `35`, pre-registered, 22 Nov night):** the μ, σ planner (7 candidates incl.
kite in mixed) loses to fixed kite in S2 (−9.5), S4 (−9.0) and S5 (−4.0); it beats the frozen planner in S2 (+17)
and S4 (+13). RQ5's pre-registered statement does not hold. Better ranking (M4) became more wins than the frozen
planner, but not more than the best simple rule. Code: `experiments/planning/closed34.py`, `kind == "musigma"` in
`phase_b.py`, `PipelinePlannerPolicy.live_margins`.


**Decision (Knut, 22 Nov night): stop developing the world model here.** The frozen pipeline stays the product;
reach is reported as the known limit (μ, σ carry the information, no sparse law found; literature: range-dependent
attrition (Bonder-Farrell), spatial PDE Lanchester models (Protopopescu 1989, Keane 2011), weak-SINDy mean-field
learning (Messenger & Bortz) needs far more units than the 6-14 per side here). Next: commit, update the supervisor
doc, the world-model interface (Phase 1 step 7), and writing.

**Black-box comparison (`planning/36`, `37`, pre-registered, 22 Nov night):** a typed MLP world model, much better
offline (margin R² at 4 s 0.54-0.61), loses to fixed kite in S2/S4/S5 by 19-55 points and to the μ, σ planner
everywhere. By the plan's fixed interpretation: the setup limits planning, not the interpretable model. The SINDy
planners were the more robust ones in closed loop. Code: `models/typed_mlp.py`, `experiments/planning/black_box.py`,
`kind == "mlp"` in `phase_b.py`.

