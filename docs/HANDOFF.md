# Handoff: where the work stands (21 Nov 2026)

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
| RQ5 | **key question:** does an LLM agent decide better with the world model | headroom shown (Phase A); the model planner not yet tested (Phase B) |

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

## 7. Update after 21-22 Nov (read this first)

**Decision waiting for Knut: a full recollection with the kite intent.** By the rule fixed before its pilot (`26`,
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
