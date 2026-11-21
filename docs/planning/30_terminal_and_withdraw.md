# A terminal value and the withdraw-firing fix: results

Plan: `25_terminal_and_withdraw_plan.md` (pre-registered before either arm ran or any gated result was read).
Seeds 18 x00 000 + (400 per scenario), the same battles as the gated test (`23`); wins, paired, bootstrap 95%.

```
                                    arm won  against won  difference               95% read-out    n
scenario arm         against                                                                        
S1       terminal:h3 hold/focus       0.652        0.690      -0.038  [-0.072, -0.005]    loses  400
                     model:h3         0.652        0.655      -0.002  [-0.015, +0.007]     ties  400
         wfire:h3    hold/focus       0.655        0.690      -0.035  [-0.070, +0.000]     ties  400
                     model:h3         0.655        0.655       0.000  [+0.000, +0.000]     ties  400
S2       terminal:h3 advance/focus    0.582        0.622      -0.040  [-0.068, -0.015]    loses  400
                     model:h3         0.582        0.628      -0.045  [-0.075, -0.015]    loses  400
         wfire:h3    advance/focus    0.630        0.622       0.008  [-0.013, +0.028]     ties  400
                     model:h3         0.630        0.628       0.002  [+0.000, +0.007]     ties  400
S3       terminal:h3 hold/focus       0.540        0.592      -0.052  [-0.087, -0.020]    loses  400
                     model:h3         0.540        0.708      -0.168  [-0.223, -0.110]    loses  400
         wfire:h3    hold/focus       0.698        0.592       0.105  [+0.050, +0.160]    beats  400
                     model:h3         0.698        0.708      -0.010  [-0.035, +0.015]     ties  400
S4       terminal:h3 hold/focus       0.095        0.100      -0.005  [-0.025, +0.015]     ties  400
                     model:h3         0.095        0.098      -0.002  [-0.015, +0.007]     ties  400
         wfire:h3    hold/focus       0.098        0.100      -0.002  [-0.022, +0.018]     ties  400
                     model:h3         0.098        0.098       0.000  [+0.000, +0.000]     ties  400
S5       terminal:h3 advance/focus    0.472        0.498      -0.025  [-0.043, -0.010]    loses  400
                     model:h3         0.472        0.498      -0.025  [-0.045, -0.005]    loses  400
         wfire:h3    advance/focus    0.498        0.498       0.000  [-0.013, +0.013]     ties  400
                     model:h3         0.498        0.498       0.000  [-0.007, +0.007]     ties  400
```

Withdraw share of re-plans (ungated planner → terminal → wfire): S1 2.4 / 0.7 / 3.1%, S2 8.9 / 3.8 / 10.2%,
S3 18.4 / 3.3 / 20.4%, S4 24.8 / 0.6 / 27.3%, S5 16.8 / 4.4 / 18.5%.

Against the expectations written before: terminal was expected to change little or gain in S1; it loses in
S1, S2, S3 and S5 and almost stops withdrawing. wfire was expected to withdraw more in S1 and tie or gain; it
ties everywhere, with slightly more withdrawals.

