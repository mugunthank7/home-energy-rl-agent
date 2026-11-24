### Summer week (Jul 5-11, 2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | total savings, terminal-adjusted % | train time (s) | % of oracle savings | % of oracle (terminal-adjusted) | % of 5-level oracle savings | % of 5-level oracle (terminal-adjusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -19.57 | 19.57 | 32.54 | 0.00 | 0.00 | 2.59 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (no training) | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | -56.71 | 15.67 | 39.21 | 19.91 | -20.51 | 4.71 | -81.47 | 3.00 | 2.36 | -19.39 | 0 (no training) | -53.3 | -51.8 | -70.4 | -69.0 |
| Greedy threshold | -29.40 | 15.86 | 31.91 | 18.93 | 1.95 | 3.21 | -23.66 | 1.00 | 2.32 | 3.24 | 0 (no training) | 5.1 | 8.7 | 6.7 | 11.5 |
| Q-learning (plan settings) | -25.91 ± 1.51 | 18.82 ± 0.58 | 37.90 ± 1.55 | 3.82 ± 2.97 | -16.48 ± 4.77 | 3.82 ± 0.23 | -47.10 ± 8.68 | 0.00 ± 0.00 | 4.88 ± 0.48 | -16.03 ± 4.77 | 10.3 | -42.9 ± 12.4 | -42.8 ± 12.7 | -56.5 ± 16.4 | -57.0 ± 17.0 |
| Q-learning (tuned) | -24.64 ± 2.20 | 17.35 ± 0.86 | 36.55 ± 2.19 | 11.31 ± 4.38 | -12.33 ± 6.74 | 3.84 ± 0.34 | -48.00 ± 13.08 | 0.00 ± 0.00 | 5.32 ± 0.14 | -12.76 ± 7.07 | 10.6 | -32.1 ± 17.5 | -34.1 ± 18.9 | -42.3 ± 23.1 | -45.4 ± 25.1 |
| Q-learning (selected: coarse grid) | -19.92 ± 0.86 | 18.77 ± 0.62 | 32.47 ± 0.83 | 4.06 ± 3.19 | 0.21 ± 2.55 | 2.74 ± 0.17 | -5.60 ± 6.39 | 0.00 ± 0.00 | 2.10 ± 0.74 | -0.37 ± 2.47 | 9.9 | 0.5 ± 6.6 | -1.0 ± 6.6 | 0.7 ± 8.7 | -1.3 ± 8.8 |
| PPO (deep RL) | -16.75 ± 0.31 | 19.14 ± 0.27 | 29.60 ± 0.33 | 2.16 ± 1.40 | 9.02 ± 1.00 | 2.09 ± 0.02 | 19.37 ± 0.70 | 0.00 ± 0.00 | 0.59 ± 0.18 | 8.76 ± 1.01 | 319.8 | 23.5 ± 2.6 | 23.4 ± 2.7 | 30.9 ± 3.4 | 31.2 ± 3.6 |
| DQN (deep RL) | -16.03 ± 1.32 | 17.54 ± 0.76 | 28.43 ± 1.32 | 10.37 ± 3.89 | 12.64 ± 4.04 | 2.18 ± 0.24 | 16.07 ± 9.28 | 0.00 ± 0.00 | 2.86 ± 1.06 | 12.59 ± 3.73 | 337.6 | 32.9 ± 10.5 | 33.6 ± 10.0 | 43.4 ± 13.9 | 44.8 ± 13.3 |
| PPO, peak input hidden (ablation) | -16.62 ± 0.61 | 18.93 ± 0.85 | 29.44 ± 0.75 | 3.25 ± 4.34 | 9.51 ± 2.31 | 2.10 ± 0.02 | 18.95 ± 0.95 | 0.00 ± 0.00 | 0.75 ± 0.69 | 9.16 ± 2.21 | 234.3 | 24.7 ± 6.0 | 24.5 ± 5.9 | 32.6 ± 7.9 | 32.6 ± 7.9 |
| DQN, peak input hidden (ablation) | -23.04 ± 4.05 | 18.51 ± 1.08 | 33.29 ± 1.42 | 5.38 ± 5.54 | -2.30 ± 4.35 | 2.95 ± 0.28 | -13.88 ± 10.62 | 0.20 ± 0.40 | 3.63 ± 1.88 | -2.22 ± 4.07 | 253.4 | -6.0 ± 11.3 | -5.9 ± 10.9 | -7.9 ± 14.9 | -7.9 ± 14.5 |
| Oracle, 5 power levels (best plan found) | -11.06 | 13.41 | 23.05 | 31.49 | 29.15 | 1.93 | 25.62 | 0.00 | 4.89 | 28.10 | 0 (no training) | 75.8 | 75.1 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | -7.85 | 15.40 | 20.03 | 21.30 | 38.45 | 0.93 | 64.32 | 0.00 | 3.99 | 37.42 | 0 (no training) | 100.0 | 100.0 | 131.9 | 133.2 |
