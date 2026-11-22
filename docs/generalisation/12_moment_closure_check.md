# Do μ and σ follow the game's movement rule? Moment-closure feasibility check (22 Nov 2026)

**Exploratory, not pre-registered; the criterion below was written before anything was computed.**

## Idea

Under advance or attack-move, a unit walks towards the enemy at speed v (3.15 map units/s for marines and
marauders) until it is in reach, then stops and fires. If the margins are roughly normal (`09`), the mean and spread
of a side's margins must then follow:

    μ′ ≈ −v · (1 − p)  (own units still walking)   − v · (1 − p_o)  (enemy units still walking in)
    σ′ ≈ −v · φ(z) · …                               the back keeps walking while the front has stopped

Here p = Φ(z), z = (c − μ)/(k σ), and φ is the normal density. Withdraw and kite add terms of opposite sign. If this
holds, a sparse μ, σ model with physical coefficients is within reach (`11` found only dense, unstable fits).

## The check

**Physics library:** 7 state groups, each × the six intent controls {1, adv, hold, wd, focus, kite} (kite in mixed
only):

| group | meaning |
|---|---|
| 1 | constant |
| q = 1 − p, q_o = 1 − p_o | own and opposing units still walking |
| φ(z), φ(z_o) | density at the reach edge (spread terms) |
| (ln A)′, (ln A_opp)′ | deaths (selection) |

**Dense reference:** the L-p library of `11` (51 groups × intents).

**Fit and evaluation:** both libraries fitted by one-step STLSQ on SG-9 rates on **train**, two-level, threshold
0.05 (as `11`); R² of μ′ and σ′ on **val**. Marines (exp06) and mixed (exp08 + exp11).

**Passes if both hold:**
1. **Fit:** the physics library reaches at least 80% of the dense library's val R², as the median over all μ′ and σ′
   equations.
2. **Physics:** in the μ′ equations (2 marines, 4 mixed), the effective coefficients on q and q_o (constant plus
   advance interaction) are negative with size between 0.5 v and 1.5 v (1.6 to 4.7 map units/s), in at least 4 of
   the 6 equations. The comparison uses own walking for Red (always attack-moves) and for Blue under advance, and
   enemy walking in both.

**Reported alongside:**
- all coefficients;
- the groups kept;
- weak-form (radius 1 s) R² for both libraries.

## Result


**Run 22 Nov.** Val R² of μ′ and σ′ (one-step SG-9 rates; weak form radius 1 s alongside):

```
scenario          equation  physics step val R²  physics groups  eff q (1+adv)  eff q_o (1+adv) physical  physics weak val R²  dense step val R²  dense groups  dense weak val R²
 marines   mu′ blue marine                0.510               5         -2.699           -0.870    False                0.456              0.830            45              0.816
 marines   sd′ blue marine                0.119               5            NaN              NaN      NaN                0.205              0.314            46              0.407
 marines    mu′ red marine                0.502               5         -0.960           -2.621    False                0.448              0.825            45              0.810
 marines    sd′ red marine                0.163               6            NaN              NaN      NaN                0.239              0.317            46              0.428
   mixed   mu′ blue marine                0.418               5         -1.462           -1.242    False                0.414              0.690            44              0.667
   mixed   sd′ blue marine                0.090               6            NaN              NaN      NaN                0.169              0.224            46              0.335
   mixed mu′ blue marauder                0.379               5         -1.038           -1.441    False                0.381              0.654            45              0.638
   mixed sd′ blue marauder                0.075               6            NaN              NaN      NaN                0.175              0.137            46              0.285
   mixed    mu′ red marine                0.408               5         -0.328           -2.559    False                0.403              0.710            45              0.692
   mixed    sd′ red marine                0.097               6            NaN              NaN      NaN                0.152              0.233            46              0.329
   mixed  mu′ red marauder                0.398               5         -0.492           -2.176    False                0.420              0.629            45              0.627
   mixed  sd′ red marauder                0.065               5            NaN              NaN      NaN                0.141              0.099            45              0.236
```

- Median physics / dense val R² (one-step): **0.58** (criterion ≥ 0.80).
- μ′ equations with both walking coefficients in [−4.7, −1.6]: **0 of 6** (criterion ≥ 4).
- **Criterion: fails.**

Physics coefficients (one-step, threshold 0.05; name·control):

```
                          0                1               2               3                4                5                  6                  7               8               9                 10                11
scenario             marines          marines         marines         marines            mixed            mixed              mixed              mixed           mixed           mixed             mixed             mixed
equation     mu′ blue marine  sd′ blue marine  mu′ red marine  sd′ red marine  mu′ blue marine  sd′ blue marine  mu′ blue marauder  sd′ blue marauder  mu′ red marine  sd′ red marine  mu′ red marauder  sd′ red marauder
1·1                    0.108            0.003           0.085          -0.015            0.176           -0.004              0.137              0.011           0.088          -0.012             0.116             0.012
1·adv                 -0.138           -0.016           -0.14          -0.015           -0.152              NaN             -0.101             -0.016          -0.109          -0.012            -0.134            -0.031
1·hold                 0.092           -0.018           0.102          -0.008              NaN              NaN                NaN             -0.013           0.066             NaN               NaN            -0.029
1·wd                   0.623            0.069           0.517             NaN            0.847            0.063              0.832              0.093           0.499             NaN             0.868               NaN
1·focus               -0.044              NaN          -0.041             NaN           -0.064              NaN                NaN                NaN             NaN          -0.015               NaN               NaN
q·1                   -0.481           -0.044          -0.329           0.011           -0.236           -0.005             -0.138             -0.032          -0.042           0.025            -0.297               NaN
q·adv                 -2.218           -0.125           -0.63            0.08           -1.227           -0.058               -0.9                NaN          -0.286           0.024            -0.195               NaN
q·hold                -0.393           -0.074          -0.097           0.035           -0.205           -0.035              -0.18                NaN          -0.145             NaN               NaN               NaN
q·wd                   0.327            0.012          -0.283          -0.048            0.319              NaN              0.366                NaN             NaN             NaN            -0.351               NaN
q·focus                0.157              NaN          -0.103          -0.009              NaN              NaN                NaN                NaN             NaN          -0.023               NaN               NaN
q_o·1                 -0.305            0.012          -0.436          -0.011           -0.457           -0.002             -0.507              0.016          -0.523          -0.044            -0.364             0.013
q_o·adv               -0.565            0.117          -2.186          -0.136           -0.786            0.018             -0.934                NaN          -2.036          -0.083            -1.812            -0.051
q_o·wd                -0.448           -0.105           0.351           0.063           -0.524           -0.102             -0.554             -0.127           0.186           0.027             0.281               NaN
q_o·focus             -0.087              NaN           0.179           0.017              NaN              NaN                NaN                NaN             NaN           0.043               NaN               NaN
phi·1                  0.694             0.02            0.33          -0.041            0.673           -0.034              0.053             -0.122           0.331          -0.123            -0.028            -0.095
phi·adv                1.951              NaN           0.678          -0.107             0.82           -0.066              0.941                NaN             NaN             NaN             0.503               NaN
phi·hold              -0.282           -0.041          -0.479          -0.071           -0.538           -0.086                NaN                NaN          -0.507             NaN               NaN               NaN
phi·wd                 -1.06            0.127           0.427           0.151           -0.559            0.295             -0.496              0.344          -0.614           0.115               NaN             0.118
phi·focus             -0.348              NaN             NaN           -0.03              NaN              NaN                NaN                NaN             NaN             NaN               NaN               NaN
phi_o·1                0.231           -0.123           0.631          -0.024            0.355           -0.097              0.487              -0.15           0.798          -0.048             0.653            -0.095
phi_o·adv               0.61           -0.152           1.961            0.03            0.988            -0.06              1.162                NaN           2.027             NaN             1.792               NaN
phi_o·hold            -0.579           -0.143          -0.393          -0.111           -0.662            -0.08             -0.614                NaN          -0.692          -0.126            -0.688            -0.066
phi_o·wd               0.255           -0.033           -1.23          -0.048           -0.554              NaN             -0.404                NaN          -0.943           0.092            -0.791               NaN
q_o·hold                 NaN            0.095          -0.296          -0.033            0.117            0.027               0.13                NaN          -0.106             NaN               NaN               NaN
phi_o·focus              NaN              NaN          -0.445          -0.033              NaN              NaN                NaN                NaN             NaN          -0.044            -0.293               NaN
dA·1                     NaN              NaN             NaN            0.02              NaN            0.032                NaN              0.027             NaN           0.024               NaN              0.04
dA·focus                 NaN              NaN             NaN          -0.008              NaN              NaN                NaN                NaN             NaN             NaN               NaN               NaN
q·kite                   NaN              NaN             NaN             NaN           -0.788           -0.077             -0.311                NaN          -0.249             NaN            -0.299               NaN
q_o·kite                 NaN              NaN             NaN             NaN           -0.983              NaN             -0.968                NaN          -1.871          -0.078            -1.253            -0.092
phi_o·kite               NaN              NaN             NaN             NaN            1.452             -0.2              1.488                NaN           1.324             NaN             0.575               NaN
1·kite                   NaN              NaN             NaN             NaN              NaN            0.058                NaN                NaN            0.12             NaN             0.196               NaN
phi·kite                 NaN              NaN             NaN             NaN              NaN           -0.151                NaN                NaN             NaN             NaN             0.664               NaN
dA·adv                   NaN              NaN             NaN             NaN              NaN              NaN                NaN              0.018             NaN             NaN               NaN               NaN
dA·hold                  NaN              NaN             NaN             NaN              NaN              NaN                NaN              0.022             NaN             NaN               NaN               NaN
dA·wd                    NaN              NaN             NaN             NaN              NaN              NaN                NaN              0.071             NaN             NaN               NaN               NaN
dA·kite                  NaN              NaN             NaN             NaN              NaN              NaN                NaN               0.05             NaN             NaN               NaN               NaN
```
Facts beside the criterion:
- With 5-6 groups instead of 44-46, the physics library explains about 60% of what the dense one does (one-step).
- In each μ′ equation, **Blue's walking term** is the large one: q under advance in Blue's equations (marines −2.70,
  mixed −1.0 to −1.5) and q_o under advance in Red's (−2.2 to −2.6). Red's own walking term is small in every
  equation (−0.3 to −1.0, effective q in Red's equations and q_o in Blue's). Only one of the two walking terms per
  equation comes near v = 3.15, so 0 of 6 meet the criterion.
- φ terms with the opposite sign appear beside the q terms (q and φ(z) are correlated near the reach edge).
- Kite enters μ′ in every mixed equation (q_o·kite −1.0 to −1.9, φ_o·kite +0.6 to +1.5).
- σ′ is poorly explained by both libraries (one-step val R² 0.07-0.16 physics, 0.10-0.31 dense).

Not checked here, and worth checking before reading the small Red term as physics: Red's attack-move walks towards
the map's attack point until it picks a target within scan range, not towards the nearest Blue unit.
