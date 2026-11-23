# Where do the model planners disagree with a perfect model? (plan, 23 Nov 2026)

**Status: run on Knut's standing instruction of 23 Nov ("if worth continuing, run the next best experiment").
Exploratory and descriptive: the read-outs below were fixed before the run; there is no pass/fail.**

## Why

At team level a perfect model with the planners' settings (`oracle5-h3`) beats the best fixed intent by 25-32 points
in S1-S4, but the μ, σ planner captures only 0-31% of that (`43`). To improve the world model where it matters, we
need to know **which decisions** the models get wrong, and **where in a battle**.

## Design

`oracle5-h3` replays the first 100 of `34`'s seeds in each of S1-S5. This gives the same battles as in `43`; the
deterministic replay is checked against the logged wins. At every oracle re-plan, the same live state is also scored
by:
- the μ, σ model (five team-level candidates, horizon 3 s);
- the frozen model (the same candidates and horizon).

Nothing is played differently, so all three rankings are compared on exactly the same states. The oracle's score per
candidate is the real health margin (hp) after the 3 s lookahead.

## Read-outs (fixed now)

1. **Agreement:** the share of re-plans where the model's pick equals the oracle's. In contact, the hold/advance twins
   with the same target count as one choice.
2. **Regret (hp):** the oracle's true score of its own pick minus that of the model's pick, as a mean per re-plan.
3. **By phase:** before first contact; the first 3 s after first contact; later.
4. **By Red's spread:** terciles of Red's σ of reach margins at the re-plan.
5. **Decisive re-plans:** the oracle's top two candidates more than 20 hp apart. Agreement and regret there.
6. **Confusion:** the oracle's pick against the model's pick (five × five), and which oracle choices the models miss
   most.

## Run time

About 500 oracle battles plus model scoring: a few minutes.
