### Validation (2009)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | total savings, terminal-adjusted % | train time (s) | % of oracle savings | % of oracle (terminal-adjusted) | % of 5-level oracle savings | % of 5-level oracle (terminal-adjusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -29.59 | 29.59 | 48.89 | 0.00 | 0.00 | 3.86 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (no training) | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | -79.80 | 24.11 | 48.87 | 18.51 | 0.03 | 4.95 | -28.29 | 4.96 | 3.13 | 0.74 | 0 (no training) | 0.1 | 2.0 | 0.1 | 2.6 |
| Greedy threshold | -48.92 | 24.20 | 44.21 | 18.20 | 9.57 | 4.00 | -3.68 | 2.34 | 3.05 | 10.21 | 0 (no training) | 25.7 | 28.0 | 32.3 | 35.5 |
| Q-learning (plan settings) | -31.90 ± 1.48 | 28.05 ± 0.38 | 49.97 ± 1.61 | 5.19 ± 1.29 | -2.21 ± 3.29 | 4.38 ± 0.27 | -13.57 ± 7.06 | 0.03 ± 0.05 | 4.76 ± 0.25 | -2.12 ± 3.32 | 10.3 | -5.9 ± 8.8 | -5.8 ± 9.1 | -7.5 ± 11.1 | -7.3 ± 11.5 |
| Q-learning (tuned) | -30.39 ± 1.02 | 27.42 ± 0.36 | 48.30 ± 0.91 | 7.31 ± 1.21 | 1.20 ± 1.86 | 4.17 ± 0.13 | -8.17 ± 3.39 | 0.04 ± 0.04 | 5.15 ± 0.14 | 1.13 ± 1.87 | 10.6 | 3.2 ± 5.0 | 3.1 ± 5.1 | 4.1 ± 6.3 | 3.9 ± 6.5 |
| Q-learning (selected: coarse grid) | -28.70 ± 0.87 | 28.65 ± 0.67 | 47.58 ± 0.84 | 3.18 ± 2.27 | 2.66 ± 1.72 | 3.79 ± 0.08 | 1.87 ± 2.13 | 0.00 ± 0.00 | 2.06 ± 0.58 | 2.39 ± 1.78 | 9.9 | 7.1 ± 4.6 | 6.5 ± 4.9 | 9.0 ± 5.8 | 8.3 ± 6.2 |
| Oracle, 5 power levels (best plan found) | -16.23 | 22.23 | 34.39 | 24.86 | 29.65 | 2.43 | 37.01 | 0.00 | 5.71 | 28.80 | 0 (no training) | 79.7 | 78.8 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | -12.49 | 22.38 | 30.69 | 24.36 | 37.23 | 1.66 | 56.95 | 0.00 | 5.50 | 36.54 | 0 (no training) | 100.0 | 100.0 | 125.5 | 126.9 |
