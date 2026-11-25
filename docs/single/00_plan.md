# One model with the scenario parameters as input (plan, 25 Nov 2026, written before the run)

Exploratory, val only; asked for by Knut. **Knut's expectation:** better performance, as more parameters are known.
Pool (equal weight): marines, marauders, mixed, triple train, and the distinct exp13 variants (refit halves). One set of
coefficients for all; distances scaled by 5.75 / reach, speeds by speed / 3.15; pooled alive law, reach closure and gates.
Small library: reach x game rate, alive x game rate, constant. Large library: (reach, alive, reach x focus, reach x hold,
reach^2) x (8 monomials of net damage, 1 / cooldown, 1 / hp; rate x range; rate x speed), and a constant: 51. STLSQ 0.15; the
large one also at 0.05. LO: fitted on marines and marine variants only. Compared with each army's own model.

E1 (Knut): better than the own model in health error at 2 s (paired 95% interval below 0) in at least 3 of marines,
marauders, mixed, triple, stalkers. E2: at most 1.10 x own in all five. E3: LO at most 1.15 x own on marauders and stalkers.
Damage, cooldown and armour are inputs here, so coefficients are no longer a check against the game.
