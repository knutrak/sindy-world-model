# Curve shapes and what shifts them (descriptive, 26 Nov 2026)

Script: `experiments/checks/curve_shapes.py`. Marine train battles (exp06) with one movement all through (669 battles), from first contact until one side has no units. Each fit uses the battle's whole future: this describes shapes, it is not a predictive model.

## Which shape fits (R² per battle, k = free numbers per curve)

```
                                           median R²  10% R²
quantity side shape                     k                   
alive    blue delayed linear            3      0.945   0.882
              exponential               2      0.928   0.865
              linear                    2      0.941   0.874
              quadratic                 3      0.954   0.905
         red  delayed linear            3      0.960   0.903
              exponential               2      0.925   0.839
              linear                    2      0.949   0.889
              quadratic                 3      0.966   0.928
distance team linear                    2      0.676   0.324
              quadratic                 3      0.869   0.685
              settling                  3      0.928   0.612
health   blue Lanchester square (joint) 1      0.960   0.933
              exponential               2      0.982   0.943
              linear                    2      0.976   0.950
              quadratic                 3      0.994   0.982
         red  Lanchester square (joint) 1      0.970   0.945
              exponential               2      0.960   0.936
              linear                    2      0.983   0.947
              quadratic                 3      0.994   0.984
share    blue logistic                  3      0.756   0.001
              saturating                3      0.755   0.103
         red  logistic                  3      0.843   0.044
              saturating                3      0.847   0.156
```

Lanchester square (joint): two numbers a, b for both health curves together, start fixed.

## What shifts the parameters between battles (5-fold CV R² across battles; 2% tails trimmed)

```
                                   median             10-90%  CV R² all features best single feature  its CV R²     slope  battles
parameter                                                                                                                         
lanchester a (Red's fire on Blue)   0.232     0.181 to 0.302               0.265              n_blue      0.228    -0.008      641
lanchester b (Blue's fire on Red)   0.245     0.193 to 0.321               0.236               n_red      0.212    -0.009      641
alive slope Blue                   -1.025    -1.47 to -0.669               0.372               n_red      0.326    -0.071      641
alive slope Red                    -1.227    -1.87 to -0.779               0.457              n_blue      0.337    -0.084      641
share tau Blue                      1.537       0.735 to 9.7               0.016           d_contact      0.020 -1449.444      641
share tau Red                       2.153  0.943 to 1.29e+04               0.043           d_contact      0.027 -1870.290      641
distance tau                        1.260      0.655 to 7.56               0.008           d_contact      0.018 -2427.758      641
distance d_eq                       5.481        4.88 to 5.7               0.053           d_contact      0.045   127.269      641
half-life Blue                      3.375        2.38 to 4.5               0.343              n_blue      0.262     0.130      636
half-life Red                       3.000       2.25 to 3.88               0.242               n_red      0.181     0.106      640
duration                            6.375           5 to 8.5               0.155               n_red      0.121     0.202      644
```

## One time scale

Health of both sides / start, against time and against time / the battle's own half-life: the median curve explains 0.93 of the spread against time and 0.98 against scaled time. Half-lives:

```
       half-life Blue  half-life Red
count          655.00         666.00
mean             3.43           3.06
std              0.83           0.69
min              1.38           1.38
25%              2.88           2.62
50%              3.38           3.00
75%              3.88           3.50
max              6.62           5.75
```

Figures: `figures/curve_collapse.png`, `figures/curve_shift.png`.
