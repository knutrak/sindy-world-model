# Counters, step 2: the committed flank, payoff screens

Plan: `02_flank_plan.md`.

## M: Blue win rate (rows Blue, columns Red), 150 paired seeds per cell

```
red              advance/nearest  advance/focus  stand/nearest  kite/nearest  spread/nearest  flank/nearest
blue                                                                                                       
advance/nearest            0.460          0.567          1.000         0.067           1.000          0.393
advance/focus              0.433          0.527          1.000         0.020           1.000          0.347
stand/nearest              0.000          0.000          0.000         0.000           0.753          0.000
kite/nearest               0.960          0.987          1.000         0.547           1.000          0.893
spread/nearest             0.000          0.000          0.273         0.000           0.527          0.000
flank/nearest              0.667          0.660          1.000         0.080           1.000          0.467
```

Blue's best reply to each Red column / Red's best reply to each Blue row:

```
red
advance/nearest       kite/nearest
advance/focus         kite/nearest
stand/nearest      advance/nearest
kite/nearest          kite/nearest
spread/nearest     advance/nearest
flank/nearest         kite/nearest

blue
advance/nearest       kite/nearest
advance/focus         kite/nearest
stand/nearest      advance/nearest
kite/nearest          kite/nearest
spread/nearest     advance/nearest
flank/nearest         kite/nearest
```

Timeout share per cell, max 1.00.

## Z2: Blue win rate (rows Blue, columns Red), 150 paired seeds per cell

```
red              advance/nearest  advance/focus  stand/nearest  kite/nearest  spread/nearest  flank/nearest
blue                                                                                                       
advance/nearest            0.567          0.807            1.0         0.580           0.993          0.987
advance/focus              0.993          1.000            1.0         0.993           1.000          1.000
stand/nearest              0.813          0.933            0.0         0.787           1.000          0.913
kite/nearest               1.000          1.000            1.0         1.000           1.000          1.000
spread/nearest             1.000          1.000            1.0         1.000           1.000          1.000
flank/nearest              0.767          0.920            1.0         0.733           1.000          0.993
```

Blue's best reply to each Red column / Red's best reply to each Blue row:

```
red
advance/nearest       kite/nearest
advance/focus        advance/focus
stand/nearest      advance/nearest
kite/nearest          kite/nearest
spread/nearest       advance/focus
flank/nearest        advance/focus

blue
advance/nearest    advance/nearest
advance/focus      advance/nearest
stand/nearest        stand/nearest
kite/nearest       advance/nearest
spread/nearest     advance/nearest
flank/nearest         kite/nearest
```

Timeout share per cell, max 1.00.

## Read-outs

```
match-up          menu best single Blue row  its mean                          near-dominant rows    P1  P2 headroom (points)     P2 95%  gate 1  timeouts
       M     full menu         kite/nearest     0.898                                kite/nearest False                   0.0 [0.0, 0.0]   False     0.028
       M without flank         kite/nearest     0.899                                kite/nearest False                   0.0 [0.0, 0.0]   False     0.040
      Z2     full menu         kite/nearest     1.000 advance/focus, kite/nearest, spread/nearest False                   0.0 [0.0, 0.0]   False     0.028
      Z2 without flank         kite/nearest     1.000 advance/focus, kite/nearest, spread/nearest False                   0.0 [0.0, 0.0]   False     0.040
```

(The timeout maximum of 1.00 is stand against stand: neither side moves. Timeouts are 2.8% overall.)

## Verdict: gate 1 fails in both match-ups; the counters line stops here (as `02` says)

- **Flank works as an intent but sits in the wrong place in the order.** In M it *beats* advance (0.67 against
  0.46 for advance against advance), the opposite of the Lanchester prediction. Concentration didn't punish the
  split: the advancing team targets the nearest half, and the other half arrives on its flank.
- **Kite beats flank** (0.89, and flank against kite 0.08), as it beats everything else. The predicted
  kite > advance > flank > kite cycle has the wrong arrow at the end: flank > advance. So the order is kite > flank
  > advance, with no cycle.
- **Z2:** kite, spread and advance/focus each win 99-100% against everything, as in stage 1. The longer spawn
  distance didn't change this.
- **Overall reading (stages 1 and 2):**
  - Kite is a pure efficiency gain: the kiting unit fires as often as a standing one and takes fewer hits.
  - No intent tried so far (focus, stand, withdraw, spread, a committed flank), and no unit mix, removes that gain.
  - A counter to kite would need something that *punishes the step back*: faster ranged pursuers, a longer-range
    opponent (stage 1's stalkers do exactly this, but then Red kites and dominates), or terrain that blocks
    retreat. None of these is a counter in the menu sense; each changes who has the dominant intent.
