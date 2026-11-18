# Final exploration: literature and library scan (axis K, 18 Nov 2026)

This is a short scan, not a review. Each entry gives what the work does, what it means for this
project, and what I do about it. The searches found little from 2025-2026 on most topics. Where a
source was only seen as a title or a search snippet, the entry says so, so check it before citing.

## PySINDy itself (installed 2.1.0)

- 2.0 was a breaking release: a simpler `SINDy` API, `AxesArray`, `EnsembleOptimizer`, and new
  sparse optimizers with inequality constraints ([release notes][pysindy-rel], [PyPI][pysindy-pypi]).
- **Correction to the audit:** the constrained optimizers are in 2.1.0. They are hidden unless their
  optional dependency is installed. With `cvxpy` (now in the `explore` extra), the following import:
  `ConstrainedSR3`, `TrappingSR3`, `StableLinearSR3`, `SINDyPI`. Still not usable: `MIOSR` (needs
  `gurobipy` and a licence) and `SBR` (needs `numpyro`).
- **Action:** axis E uses ConstrainedSR3 for the symmetry and sign constraints. It also tries
  StableLinearSR3 / TrappingSR3 only if the blow-up rate is above zero for some candidate (today it
  is 0 for A). SBR is replaced by the bootstrap ensemble below.

## Uncertainty: ensembles and Bayesian SINDy

- **E-SINDy** (Fasel et al. 2022; convergence analysis by Gao et al. 2023, [arXiv 2301.12649][esindy]):
  bagging / library bagging around STLSQ gives inclusion probabilities and coefficient intervals
  cheaply. The 2023 paper shows these estimates converge, and argues they are much cheaper than MCMC.
- **Bayesian SINDy** with Gaussian approximations for speed (2024, [arXiv 2402.15357][bsindy]);
  spike-and-slab variants exist too.
- **Relevance:** high. A failed on G3b because the *alive* term set changes with the derivative
  estimate. Inclusion probabilities from an ensemble state directly how firm each alive term is. An
  ensemble also gives a win probability, which the brief wants calibrated.
- **Action:** axis E/G. EnsembleOptimizer (bagging by battle, which is what matters here, rather
  than by row), inclusion probabilities, and ensemble rollouts → win probability, Brier score and
  calibration.

## Hybrid / switching SINDy

- **Hybrid-SINDy** (Mangan et al. 2019, [arXiv 1808.03251][hybrid]): cluster the data, fit sparse
  models per cluster, and pick models with information criteria. Later work extends it to impact
  oscillators (2025, [Nonlinear Dynamics][hybrid25]) and to latent switching via variational
  inference ("Dynamic SINDy", 2024, [arXiv 2410.02079][dynsindy]).
- **Relevance:** medium. The model already switches smoothly (movement gates, far/contact,
  the reach closure). An explicit phase model could only win if phase boundaries are sharper than
  the gates. The status note's worry 2 (too many branches) argues for a quick screen only.
- **Action:** axis C as a screen. Phases from the engaged fraction and from change points
  (`ruptures`), per-phase STLSQ, compared with A's gating on the harness.

## SINDy as a world model for control / RL

- Arora et al. 2022 (SINDy in model-based RL, few rollouts, [arXiv 2208.14501][mbrl]);
  SINDy-RL (Zolman, Fasel, Kutz 2024; seen as a title only); "Learning from less: SINDy surrogates in
  RL" (ICLR 2025 world-models workshop, [mlanthology][less]): SINDy surrogate environments that train
  agents with fewer real steps.
- **Relevance:** framing for RQ5 (out of scope for Phase 1). These papers work on low-dimensional
  control benchmarks with full state. The combat setting has a hidden state (formation), which the
  intent-ranking result already showed matters. Worth one paragraph in related work.
- **Action:** none in this exploration.

## SINDy vs symbolic regression (PySR)

- Comparisons ([e.g. arXiv 2508.20257][srcmp]) tend to find PySR more robust when the library is
  unknown, and SINDy faster and as good when the library contains the right terms. All of them are
  sensitive to derivative noise. Most are small preprint studies.
- **Relevance:** low for the final model. The library here is designed from the game's rules, which
  is the setting where SINDy does well. PySR could answer the supervisor's question of whether
  SINDy finds terms by itself, but it needs Julia. The decision (18 Nov) was not to install it.
- **Action:** the axis D "open library" screen approximates this. It uses a large generic library,
  ensemble selection, and a check on whether the designed terms come back.

## Weak / integral SINDy

- Weak SINDy is mature, and PySINDy has `WeakPDELibrary`. This project already ran a weak-form fit
  (`sindyc_07` §9, W in the gates). The terms stayed the same, the alive exponents moved, and there
  was no rollout gain.
- **Action:** not repeated as a new axis. The harness re-scores W for comparability, if time allows.

## Combat modelling with system identification

- One directly relevant paper: Lanchester-type ODEs fitted with SINDy to Russia-Ukraine equipment
  losses, with the data partitioned into regimes before fitting (*Mathematics* 2024, 12(6) 851,
  [MDPI][lanchester-sindy]). It works on aggregate real-world losses, has no spatial state, and its
  regime partition is similar in spirit to Hybrid-SINDy.
- No 2023-2025 work turned up on SINDy for simulated unit-level combat (SMAC or similar). That is a
  gap this thesis can name, hedged as "none found in this scan".
- **Action:** cite in related work. The "fitted Lanchester" baselines in the harness (square,
  linear, mixed) are the comparison that paper makes.

## What this adds to the plan

1. Ensemble SINDy for the alive equation and for win probabilities (E, G). Highest priority.
2. ConstrainedSR3 for side symmetry and signs (E).
3. Phase models only as a screen (C).
4. Nothing from the RL or symbolic-regression literature is implemented. They are written up as
   context only.

[pysindy-rel]: https://www.github.com/dynamicslab/pysindy/releases/tag/v2.0.0-rc1
[pysindy-pypi]: https://pypi.org/project/pysindy/
[esindy]: https://arxiv.org/pdf/2301.12649
[bsindy]: https://arxiv.org/html/2402.15357v2
[hybrid]: https://arxiv.org/pdf/1808.03251
[hybrid25]: https://link.springer.com/article/10.1007/s11071-025-11253-6
[dynsindy]: https://ar5iv.labs.arxiv.org/html/2410.02079
[mbrl]: https://arxiv.org/pdf/2208.14501
[less]: https://mlanthology.org/iclrw/2025/dixit2025iclrw-learning
[srcmp]: https://arxiv.org/pdf/2508.20257v2
[lanchester-sindy]: https://www.mdpi.com/2227-7390/12/6/851
