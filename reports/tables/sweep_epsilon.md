### sweep_epsilon

| variant | total savings % | energy savings % | peak reduction % | violations / week | cycles / week | train time (s) | seeds |
|---|---|---|---|---|---|---|---|
| epsilon_schedule=linear | -2.87 ± 2.03 | 4.10 ± 1.82 | -13.56 ± 3.56 | 0.02 ± 0.03 | 5.00 ± 0.15 | 19.46 ± 0.12 | 5 |
| epsilon_schedule=exponential_fast | -1.67 ± 0.25 | 4.19 ± 0.80 | -10.64 ± 1.02 | 0.06 ± 0.07 | 4.39 ± 0.25 | 20.15 ± 0.35 | 5 |
| epsilon_schedule=exponential | -2.21 ± 3.29 | 5.19 ± 1.29 | -13.57 ± 7.06 | 0.03 ± 0.05 | 4.76 ± 0.25 | 20.52 ± 0.14 | 5 |
| epsilon_schedule=exponential_slow | -3.93 ± 2.87 | 3.22 ± 1.42 | -14.89 ± 5.51 | 0.01 ± 0.01 | 4.77 ± 0.27 | 15.26 ± 2.71 | 5 |
