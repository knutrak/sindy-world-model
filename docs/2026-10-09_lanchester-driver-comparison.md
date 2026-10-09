# 2026-10-09 · Lanchester square law on three candidate drivers

Prompted by a question worth recording as-is: classical Lanchester theory models **unit counts**,
not aggregate health. SMAClite's actual damage mechanics agree — a unit deals full damage every
attack regardless of its own remaining HP, right up until it dies — so `alive_*` (or `engaged_*`,
from the newly-added `engagement()` state) is arguably more mechanistically faithful than
`health_*`, which is what every fit so far (including notebook 04) had used.

## New in `src/sindy_wm/models/sindy.py`

- **`LanchesterModel`**: a plain 2-parameter model, `dX1/dt = -a·X2`, `dX2/dt = -b·X1` — no
  constant, no self-term. Implements just enough of `ps.SINDy`'s interface (`simulate`,
  `coefficients`, `get_feature_names`, `equations`) to drop into the existing `simulate`,
  `evaluate`, `summarize_eval`, `plot_predictions`, `save_run` pipeline unchanged.
- **`fit_lanchester(trajectories, train_ids, state_columns, dt, smooth_window)`**: fits it via the
  same `SmoothedFiniteDifference` derivative estimate `fit_sindy` uses (for a fair comparison), but
  the regression step is two closed-form 1-variable least-squares fits through the origin, not a
  SINDy/STLSQ search — the square-law form is fixed, not something sparsification happened to find.
- **`evaluate()` generalized**: it hardcoded `state_columns.index("health_blue"/"health_red")` to
  find which column is Blue's; now uses `state_columns[0]/[1]` by convention (Blue first, Red
  second — already true of every existing caller). Needed so `evaluate`/`summarize_eval` work on
  `alive_*`/`engaged_*` too, not just health.

## New notebook: `06_lanchester_drivers.ipynb`

Fits `fit_lanchester` on `health`, `alive`, and `engaged` (normalized by unit count, to put it on
the same 0-1-ish scale as the other two) on the same episodes from `exp03` (`10m_vs_10m`,
randomized spawn — the only experiment collected with the `vx`/`vy`/`target_*` columns
`engagement()` needs), trimmed once via health so all three drivers are evaluated on exactly the
same time window per episode. Runs a `SMOOTH_WINDOW` sensitivity sweep per driver (same check as
notebook 04's Part 4), and plots simulated-vs-actual for each.

Validated by executing every cell against the real `10m_attack_a070_s1_v1` dataset; the numbers
below are from that run, not hand-typed.

## The result

| driver | a | b | rmse_midway | winner_acc_midway | leader_baseline_midway |
|---|---|---|---|---|---|
| health | 0.171 | 0.167 | 0.071 | 0.89 | 0.89 |
| alive | 0.117 | 0.106 | 0.122 | 0.71 | 0.71 |
| engaged (÷10) | 0.008 | **-0.017** | 0.220 | 0.61 | 0.61 |

All three stable across `SMOOTH_WINDOW ∈ {5, 9, 13, 17}` — none of this is a smoothing artifact.

**`health` and `alive` are legitimate fits that exactly match their own naive baseline.** That's
not a coincidence specific to these two: with `a, b > 0`, the square law's structure (both
quantities monotonically decreasing, faster when the opponent is currently stronger) makes a lead
reversal from the midpoint mathematically rare regardless of which driver is used. Matching
`leader_baseline_midway` is close to the ceiling the *functional form* allows, not a property of
health or alive-count specifically.

**`engaged`'s `b` is consistently negative, not just noisy** — stable-but-wrong-signed across
every window tested. Read as: `engaged_*` is a fluctuating activity measure (units move in and out
of range), not a monotonically depleting stock like health or alive count, so the classical
attrition form (`dX/dt = -a·(opponent)`) isn't the right shape for it at all, independent of
smoothing. This is a stronger, more specific claim than "it's noisy" — worth re-checking on another
dataset before fully trusting it (see README's Next steps), but the sweep already rules out
smoothing-window choice as the explanation.

**Where SINDy actually earns its keep**: the general (non-Lanchester-constrained) `degree=2` SINDy
fit on health in notebook 04 gets `winner_acc_midway=0.84` against `leader_baseline_midway=0.73` —
a real improvement, on a different dataset. The classical square law, constrained to pure
cross-coupling by construction, structurally can't do that regardless of which driver feeds it.
That comparison — general fit vs. its own baseline — is the real test of whether SINDy finds
something beyond 1916 theory, not whether a Lanchester fit's coefficients "look sensible."

## Also fixed in passing

Notebook `05`'s hardcoded path (`exp02_5m_random_position_aggression_stepmul`) had gone stale —
that folder was renamed to `exp02_5m_random_aggression_stepmul` at some point and the notebook
silently pointed at nothing. Updated; confirmed it runs again.
