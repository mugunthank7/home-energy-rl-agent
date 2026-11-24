### Test (2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | total savings, terminal-adjusted % | train time (s) | % of oracle savings | % of oracle (terminal-adjusted) | % of 5-level oracle savings | % of 5-level oracle (terminal-adjusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -28.04 | 28.04 | 45.31 | 0.00 | 0.00 | 3.45 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (no training) | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | -86.55 | 22.74 | 47.51 | 18.88 | -4.86 | 4.95 | -43.41 | 5.57 | 3.05 | -4.07 | 0 (no training) | -13.7 | -11.7 | -17.7 | -15.1 |
| Greedy threshold | -48.90 | 22.87 | 41.78 | 18.44 | 7.79 | 3.78 | -9.51 | 2.38 | 2.95 | 8.51 | 0 (no training) | 21.9 | 24.4 | 28.4 | 31.5 |
| Q-learning (plan settings) | -31.43 ± 0.82 | 26.73 ± 0.30 | 47.53 ± 0.98 | 4.65 ± 1.07 | -4.92 ± 2.16 | 4.16 ± 0.15 | -20.45 ± 4.33 | 0.02 ± 0.02 | 4.82 ± 0.24 | -4.72 ± 2.18 | 10.3 | -13.8 ± 6.1 | -13.5 ± 6.3 | -17.9 ± 7.9 | -17.5 ± 8.1 |
| Q-learning (tuned) | -30.01 ± 0.85 | 26.12 ± 0.32 | 46.06 ± 0.76 | 6.85 ± 1.14 | -1.66 ± 1.67 | 3.99 ± 0.16 | -15.48 ± 4.49 | 0.02 ± 0.03 | 5.11 ± 0.16 | -1.65 ± 1.67 | 10.6 | -4.7 ± 4.7 | -4.7 ± 4.8 | -6.0 ± 6.1 | -6.1 ± 6.2 |
| Q-learning (selected: coarse grid) | -27.71 ± 0.64 | 27.24 ± 0.53 | 44.61 ± 0.62 | 2.85 ± 1.88 | 1.53 ± 1.37 | 3.47 ± 0.07 | -0.60 ± 2.05 | 0.00 ± 0.00 | 1.84 ± 0.52 | 1.26 ± 1.41 | 9.9 | 4.3 ± 3.8 | 3.6 ± 4.0 | 5.6 ± 5.0 | 4.7 ± 5.2 |
| Oracle, 5 power levels (best plan found) | -16.69 | 21.27 | 32.86 | 24.13 | 27.47 | 2.32 | 32.88 | 0.00 | 5.49 | 27.04 | 0 (no training) | 77.1 | 77.5 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | -12.95 | 21.41 | 29.17 | 23.63 | 35.61 | 1.55 | 55.06 | 0.00 | 5.24 | 34.87 | 0 (no training) | 100.0 | 100.0 | 129.7 | 129.0 |
