# Kite in the world model: a small collection and a refit (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, as drafted; decision rules fixed before any new data was collected.**

## Why

Kite beats every fixed intent in S1-S3 (`27`), and in S4 the exact oracle with kite gains +12 points (`27`, P3).
The gain comes from the step back, which costs Red 0.05-0.17 of its firing duty (`31`). Unlike focus, kite should
show up in **team health**, which is where the model is good. The frozen model can't plan with kite: a kite step
has u = (0, 0, 0, focus), so no movement gate fires.

**The question:** can a model refitted with kite as a control decide better than fixed kite, mainly in S4?

## Scope

- **Mixed scenario only** (`mixed_m_mrd`, `MixedBattleSampler`), since its model serves S2, S4 and S5 (S5 uses the
  model fitted on mixed). S1 and S3 are left out: kite already wins about 98% there.
- The frozen model and every earlier result stay as they are. The refit is a **new model variant**, "kite".
- Strength ratio in training stays 0.85-1.18, as in exp08. S4 (0.6-0.8) stays outside the training range (see Q2).

## Data (new seeds 20 000 000 +, used nowhere else)

| set | what | battles | seeds |
|---|---|---|---|
| `train_kite_0..3` | exp08's schedule (intents switch every 1-4 s), movements advance / hold / withdraw / **kite** with weights 0.35 / 0.20 / 0.15 / 0.30 | 4 × 500 | 20 000 000 + |
| `val_kite` | same schedule, new seeds | 300 | 20 100 000 + |
| `cf_kite_nearest`, `cf_kite_focus` | exp08's counterfactual design (switching, then kite for 4 s, then advance/nearest) on **exp08's cf seeds** (10 600 000 +, 600), so they pair with the six existing cf sets | 2 × 600 | 10 600 000 + |

Total: about 3,500 battles. exp08 ran 1,000 battles in 7.5 min on one core, so with 4 shards in parallel this is
about **8-10 min**, plus about 5 min to build the states. Kite will be about 8% of training steps, together with
exp08's 6,000 battles.

**Code needed first (committed before collecting, since `collect` refuses a dirty tree):**
- `IntentSchedule` accepts a fourth movement weight for kite. A three-weight schedule must draw exactly as before,
  and a test checks that.
- The control vector gets a fifth column, kite, appended at the end, so existing indices stay valid.

## The model (the frozen pipeline, plus kite terms only)

The health equation for side s, type t, attacked by side o (the existing terms first):

    H_st' = Σ_a k_{a,t} · R_oa · σ_{oa,t}  +  Σ_a c_{a,t} · R_oa · σ_{oa,t} · kite  +  (A terms, constant)

- When Blue is attacked (o = red), c is the kite effect on **Red's** fire, which `31` predicts is negative
  (Red loses fire chasing).
- When Red is attacked (o = blue), c is the kite effect on **Blue's** fire (`31`: duty slightly up).
- The kite terms are controls, so they are pruned at **half** the health threshold (0.075, standardised). The state
  terms keep 0.15, following the rule used for controls in `09`.

The other equations:

    alive       A_st' = (p_st + pf_st · focus) · Ĥ_st' · A_st / H_st      unchanged
    distance    the movement library with a kite gate (fitted as the other gates, started from advance's values)
    reach       closure unchanged; checked on kite steps in K2

## Checks and decision rules (fixed once approved)

| step | what | passes if |
|---|---|---|
| K1 | the refit on old movements: exp08 val rollouts at 1, 2, 4 s, kite variant vs frozen | team health and distance error within +5% of frozen's (it must not get worse where it was tested) |
| K2 | kite steps: `val_kite` rollouts, error on kite segments vs advance segments; reach closure on kite steps | error on kite segments ≤ 1.5 × the error on advance segments at 2 s; reported either way |
| K3 | ranking on the counterfactual sets (eight candidates incl. kite/nearest, kite/focus): regret at 1, 2, 4 s, kite variant vs always kite/focus and vs the frozen model | regret below always-kite at 2 s or 4 s |
| K4 | closed loop, only if K1-K3 pass: planner (7 candidates, horizon 3 s, re-plan 1 s, commit 2 s) on fresh seeds 21 x00 000 +, 400 per scenario, paired with fixed kite/focus, fixed kite/nearest and Phase A's best fixed intent | **main: S4** (pre-registered here as the main test, flagged as out of training range); S2, S5 reported |

**K4 read-out:** the planner beats the best fixed kite intent if the paired 95% interval of the win difference is
above 0. It should also report the share of the seven-candidate oracle's S4 headroom it captures, using `27` P3's
100 seeds as the reference.

K4 costs about 1,200 planner battles at the Phase B rate. Its run time will be timed on 10 battles before it
starts.

## Expectations (written before any data)

- K1 passes: kite rows add data without changing what the old terms see.
- K2: a kite term on Red's fire is kept (c < 0 for Blue's health), and the error on kite segments is higher than on
  advance, from the sub-second rhythm.
- K3: it beats the frozen model, since that model can't express kite; whether it beats always-kite is uncertain.
- K4: ties fixed kite in S2 and S5. In S4, a 30-40% chance of beating fixed kite.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Mixed only, or also marines (S1)? | mixed only; kite leaves nothing to gain in S1 |
| Q2 | Widen the strength ratio in the new kite shards so S4 is inside training? | not now: two changes at once can't be told apart; a second variant later if K4 fails |
| Q3 | Kite weight 0.30 in the new shards (about 8% of all training steps)? | yes; if STLSQ drops kite, report that rather than raise it |
| Q4 | Make S4 the main test for this variant? | yes, flagged as out of training range |

## Amendments

- **22 Nov, before any kite data was read:** K3 uses **seven** candidates, not eight. exp08's typed counterfactual
  sets have no withdraw/focus: it plays the same battle as withdraw/nearest, since Blue doesn't fire while
  withdrawing, and the earlier ranking work (`02`-`06`) used the same five sets.
