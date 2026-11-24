### Winter weeks (Jan-Feb 2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | total savings, terminal-adjusted % | train time (s) | % of oracle savings | % of oracle (terminal-adjusted) | % of 5-level oracle savings | % of 5-level oracle (terminal-adjusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -37.70 | 37.70 | 60.32 | 0.00 | 0.00 | 4.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (no training) | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | -96.95 | 30.28 | 59.36 | 19.68 | 1.58 | 5.82 | -28.58 | 5.94 | 4.02 | 2.05 | 0 (no training) | 4.6 | 6.0 | 5.9 | 7.7 |
| Greedy threshold | -64.68 | 30.35 | 53.74 | 19.50 | 10.90 | 4.68 | -3.42 | 3.28 | 3.81 | 11.11 | 0 (no training) | 31.4 | 32.5 | 40.4 | 41.9 |
| Q-learning (plan settings) | -38.26 ± 0.73 | 35.71 ± 0.41 | 59.66 ± 0.96 | 5.26 ± 1.10 | 1.09 ± 1.59 | 4.79 ± 0.14 | -5.87 ± 2.98 | 0.02 ± 0.03 | 4.88 ± 0.28 | 0.95 ± 1.57 | 10.3 | 3.1 ± 4.6 | 2.8 ± 4.6 | 4.0 ± 5.9 | 3.6 ± 5.9 |
| Q-learning (tuned) | -37.19 ± 1.75 | 35.19 ± 0.38 | 58.50 ± 1.71 | 6.66 ± 1.01 | 3.01 ± 2.84 | 4.66 ± 0.27 | -3.08 ± 6.05 | 0.02 ± 0.04 | 5.35 ± 0.22 | 2.73 ± 2.85 | 10.6 | 8.7 ± 8.2 | 8.0 ± 8.4 | 11.1 ± 10.5 | 10.3 ± 10.8 |
| Q-learning (selected: coarse grid) | -36.56 ± 0.84 | 36.51 ± 0.66 | 58.81 ± 0.85 | 3.16 ± 1.76 | 2.51 ± 1.41 | 4.46 ± 0.08 | 1.42 ± 1.80 | 0.00 ± 0.00 | 1.86 ± 0.43 | 2.14 ± 1.39 | 9.9 | 7.2 ± 4.1 | 6.3 ± 4.1 | 9.3 ± 5.2 | 8.1 ± 5.2 |
| PPO (deep RL) | -32.45 ± 0.35 | 36.09 ± 0.29 | 54.35 ± 0.36 | 4.27 ± 0.77 | 9.90 ± 0.60 | 3.65 ± 0.02 | 19.28 ± 0.47 | 0.04 ± 0.01 | 1.81 ± 0.20 | 9.92 ± 0.58 | 319.8 | 28.5 ± 1.7 | 29.0 ± 1.7 | 36.7 ± 2.2 | 37.4 ± 2.2 |
| DQN (deep RL) | -33.07 ± 2.78 | 34.97 ± 0.93 | 54.07 ± 1.62 | 7.23 ± 2.47 | 10.36 ± 2.69 | 3.82 ± 0.19 | 15.58 ± 4.29 | 0.06 ± 0.10 | 4.90 ± 1.77 | 10.34 ± 2.68 | 337.6 | 29.8 ± 7.7 | 30.3 ± 7.8 | 38.4 ± 9.9 | 39.0 ± 10.1 |
| PPO, peak input hidden (ablation) | -33.43 ± 1.44 | 36.06 ± 0.92 | 55.17 ± 1.62 | 4.36 ± 2.43 | 8.54 ± 2.69 | 3.82 ± 0.15 | 15.51 ± 3.23 | 0.06 ± 0.05 | 1.63 ± 0.66 | 8.40 ± 2.68 | 234.3 | 24.6 ± 7.8 | 24.6 ± 7.8 | 31.6 ± 10.0 | 31.7 ± 10.1 |
| DQN, peak input hidden (ablation) | -35.85 ± 1.93 | 35.37 ± 1.01 | 56.01 ± 1.88 | 6.18 ± 2.68 | 7.15 ± 3.11 | 4.13 ± 0.18 | 8.76 ± 4.04 | 0.16 ± 0.07 | 4.52 ± 2.08 | 7.10 ± 3.29 | 253.4 | 20.6 ± 9.0 | 20.8 ± 9.6 | 26.5 ± 11.5 | 26.8 ± 12.4 |
| Oracle, 5 power levels (best plan found) | -22.58 | 29.42 | 44.03 | 21.95 | 27.00 | 2.92 | 35.42 | 0.00 | 5.85 | 26.52 | 0 (no training) | 77.8 | 77.7 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | -17.99 | 29.34 | 39.38 | 22.18 | 34.71 | 2.01 | 55.60 | 0.00 | 6.15 | 34.15 | 0 (no training) | 100.0 | 100.0 | 128.6 | 128.8 |
