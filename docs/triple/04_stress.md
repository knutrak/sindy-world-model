# Stress tests of the pipeline (25 Nov 2026)

Scripts: `experiments/triple/stress_fit.py`, `stress_follow.py`, `stress_rollout.py`; full output in `logs/` (before
and after the fixes). Train and val data only, apart from the single exp12 test run in `02_test.md`. Exploratory: the
checks were not pre-registered. Regression tests: `tests/test_pipeline_edge_cases.py`.

**Every fix was checked against the tested models:** refitting marines, marauders and mixed after the fixes gives
all 332 numbers of the frozen models with a difference of exactly 0.0.

## Defects found and fixed

| # | defect | effect | did it touch reported results? | fix |
|---|---|---|---|---|
| 1 | `stlsq`: a constant column was standardised by its spread (0), so the constant could never be kept | a planted constant went into firepower: −0.02 turned k 0.18 into 0.22 | no: with the fix all 16 health equations (five data sets) keep the same terms and numbers | constant scaled by its size |
| 2 | `stlsq`: when no term reached the threshold it returned **all** of them | shuffled, meaningless rates gave a full model | not the five fitted models; a sweep to very high thresholds would look denser than it is | returns zero |
| 3 | wrong time step accepted (dt 0.25 read as 0.125) | health error 4 × larger, no error message | no: all 24 train / val tables pass the new check | refused |
| 4 | two battles sharing an episode id merged silently | k −22%, p +7-11% | no (same check) | refused |
| 5 | a unit type present but not listed counted in the geometry, not in the health | wrong states, no message | no: the tests and experiments that were run still pass | refused |
| 6 | missing hold / advance steps, NaN, inf, absent types | obscure crash deep in scipy | no | clear message |
| 7 | world-model interface (new on 25 Nov): unit with hp NaN silently dropped, hp inf, hp above maximum and x inf accepted, typos in plans crashed | wrong predictions | not used in any result | refused |

Defect 1 also means that the statement "candidates also include a constant" was not true in effect for any earlier
STLSQ fit. With the constant now a real candidate it is still not selected: in 60 battle-bootstrap resamples the
marine, marauder, stalker and mixed equations keep exactly their firepower terms every time. But it is close. Added
by hand, the intercept is −0.001 to −0.009, its effect size is 0.164-0.187 in three single-type equations (threshold
0.15), and firepower is then 3-13% lower. (STLSQ drops it first, while the alive count is still in the library.)

## What held

- **Rollouts:** 10,000 random states per model (5 models; also huge armies, single units, distance 0 and 1000, tiny
  fractions): no NaN, no negative value, no rising health or unit count, no unit type appearing. Batch equals one by
  one exactly, inputs are not changed, repeated calls are identical, a NaN stays in its own row.
- **Symmetries:** swapping the sides (without withdraw) and the order of the types changes nothing (≤ 1e-16); states
  are unchanged by translation, rotation, reflection and relabelling (≤ 1e-13).
- **Planted laws** (real features, residual-sized noise): the right terms in 20 of 20 for four laws (as fitted, a
  missing term, a strong cross term, alive counts instead of reach), coefficients within 10%; not for weak terms (below). **Null** (shuffled rates): only the constant (the mean rate) is kept.
- **Robust settings:** marines and mixed give the same terms for thresholds 0.1-0.25 and windows 3-17.
- **Data:** marines, stalkers and mixed find the structure from about 100 battles.
- **Movement gates:** 12 random starting points agree with the pipeline's fit (no local minima), all scenarios.
- **Noise:** health noise up to 0.01 (about 5 marine hp) and reach noise up to 0.5 units leave the terms unchanged.

## Limits found (not fixed)

1. **Weak terms are invisible.** A true firepower term of 0.03, or a cross term of 5%, was found in 0 of 20 trials.
2. **The three-type structure is at the edge.** Marine → stalker has an effect size of 0.162 against a threshold of
   0.15. Exact structure in **41 of 60** bootstrap resamples (that term kept in 68%) and in 3 of 5 draws of 1000
   battles. The 17 Nov gate G3a asks for 90%. So T1 in `02_test.md` was met, but would not be met reliably. Effect
   sizes shrink as types are added (smallest term: mixed 0.32, triple 0.16), because more terms share one target. A
   fixed threshold of 0.15 does not scale to more types.
3. **T2 is borderline.** Marine → marauder is 1.014 × the game on the train fit and 0.99 in the test refit; the same term
   missed in the uneven test (1.02).
4. **Stalkers alone: the closing speed is wrong.** Advance −3.97 against 2 × 4.13 = 8.26 (0.48 ×); hold in contact
   −3.76. Earlier scenarios' check (within 20%) would fail. Not a fitting failure: the gates agree from every
   starting point. In the data stalkers close at 8.4 only beyond distance 12, and at 3.2-4.6 between 8 and 12, where only
   56-65% of units walk at full speed. Likely reading (not tested): stalker formations are large, so their fronts
   meet while the centres are still 8-12 apart, and one switch on the centre distance can't describe that.
5. **Shields.** Stalker health rises in 34-44% of steps (marines and marauders: 0%); regained health is 8-9% of the
   damage. The model clamps health to fall, and no own-alive term is selected. So stalker firepower is a net rate.
6. **Too few battles:** with 3-10 battles the gates sometimes do not converge (a clear scipy error) and terms are wrong.
7. **Smoothing windows of 33 steps or more** (4 s) bring in wrong terms.
8. **Splitting a rollout** (2 s + 2 s) differs from 4 s in one go by up to 0.03, because the measured reach
   offset decays from the start of each call.
9. **Accepted as they are:** a unit listed twice, battles far outside training (1 against 200), durations rounded to
   0.125 s, negative hp read as dead. "Red loses no health while Blue withdraws" holds by rule.
10. **`Pipeline.complexity()` is wrong** about closure numbers (reported 6 / 24, used 23 / 29), as noted before.

## What limits the triple structure? (25 Nov, exploratory, train; `threshold.py`, `logs/threshold.log`)

- **The threshold.** Exact structure in 60 bootstraps of the triple model: 100% at 0.10-0.14, 65-70% at 0.15, 0% at 0.16.
  All five data sets are at 100% for 0.12-0.14. The largest spurious candidate in the triple fit is 0.078.
- **Not detectability.** The weak term (Blue's stalkers losing health to Red's marines) is k = −0.0485 ± 0.0010, 50 sd from
  0. Dropping it costs 0.03-0.04 of one-step val R² and shifts the two other coefficients by 11% (Blue) and 19-20% (Red).
- **Not mainly data.** Four times the battles (resampled) raises 70% only to 87%: its effect size stays at the threshold.
- **Why 0.162 fails 30% of the time:** elimination is decided in the first pass, with all 13 candidates in, where its
  effect size is 0.153 ± 0.005 (below 0.15 in 30% of resamples). With only the three real terms it is 0.162 and never below.
- **Why it is small:** effect = |k| sd(x) / sd(y). Marines are on average a few units of three types and their fire is
  split over three targets (sd(x) 0.056 here, 0.245 for marines alone), and a marine does little to a stalker (k 0.049).

A lower threshold (0.12-0.14) was found after the fact. It is a hypothesis for new data, not a tested setting, and the
frozen 0.15 stays in the reported models.
