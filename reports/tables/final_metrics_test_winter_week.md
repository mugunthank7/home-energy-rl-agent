### Winter week (Jan 18-24, 2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | total savings, terminal-adjusted % | train time (s) | % of oracle savings | % of oracle (terminal-adjusted) | % of 5-level oracle savings | % of 5-level oracle (terminal-adjusted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -42.85 | 42.85 | 69.18 | 0.00 | 0.00 | 5.27 | 0.00 | 0.00 | 0.00 | 0.00 | 0 (no training) | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | -118.61 | 35.04 | 64.06 | 18.22 | 7.39 | 5.80 | -10.24 | 8.00 | 4.38 | 8.00 | 0 (no training) | 21.1 | 23.2 | 26.0 | 28.4 |
| Greedy threshold | -76.10 | 35.26 | 61.59 | 17.71 | 10.97 | 5.27 | 0.00 | 4.00 | 4.19 | 11.48 | 0 (no training) | 31.3 | 33.2 | 38.6 | 40.7 |
| Q-learning (plan settings) | -42.93 ± 3.62 | 41.20 ± 0.48 | 68.28 ± 3.69 | 3.85 ± 1.12 | 1.30 ± 5.34 | 5.42 ± 0.73 | -2.85 ± 13.96 | 0.00 ± 0.00 | 4.88 ± 0.68 | 1.13 ± 5.35 | 10.3 | 3.7 ± 15.2 | 3.3 ± 15.5 | 4.6 ± 18.8 | 4.0 ± 19.0 |
| Q-learning (tuned) | -39.96 ± 4.27 | 40.54 ± 1.08 | 65.22 ± 4.26 | 5.39 ± 2.52 | 5.72 ± 6.16 | 4.94 ± 0.75 | 6.27 ± 14.22 | 0.00 ± 0.00 | 5.33 ± 0.67 | 5.57 ± 6.19 | 10.6 | 16.3 ± 17.6 | 16.1 ± 17.9 | 20.1 ± 21.7 | 19.7 ± 21.9 |
| Q-learning (selected: coarse grid) | -40.24 ± 2.58 | 41.47 ± 0.95 | 66.17 ± 2.62 | 3.22 ± 2.22 | 4.34 ± 3.79 | 4.94 ± 0.40 | 6.17 ± 7.55 | 0.00 ± 0.00 | 1.98 ± 0.46 | 4.01 ± 3.78 | 9.9 | 12.4 ± 10.8 | 11.6 ± 10.9 | 15.3 ± 13.3 | 14.2 ± 13.4 |
| Oracle, 5 power levels (best plan found) | -24.44 | 34.26 | 49.51 | 20.04 | 28.43 | 3.05 | 42.09 | 0.00 | 6.32 | 28.21 | 0 (no training) | 81.2 | 81.6 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | -19.92 | 33.61 | 44.94 | 21.55 | 35.03 | 2.27 | 56.97 | 0.00 | 6.52 | 34.55 | 0 (no training) | 100.0 | 100.0 | 123.2 | 122.5 |
