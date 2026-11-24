### Winter weeks (Jan-Feb 2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines are deterministic. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | train time (s) | % of oracle savings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -37.70 | 37.70 | 60.32 | 0.00 | 0.00 | 4.52 | 0.00 | 0.00 | 0.00 | 0 (no training) | n/a (oracle not built yet) |
| Rule-based | -96.95 | 30.28 | 59.36 | 19.68 | 1.58 | 5.82 | -28.58 | 5.94 | 4.02 | 0 (no training) | n/a (oracle not built yet) |
| Greedy threshold | -64.68 | 30.35 | 53.74 | 19.50 | 10.90 | 4.68 | -3.42 | 3.28 | 3.81 | 0 (no training) | n/a (oracle not built yet) |
| Q-learning (plan settings) | -38.26 ± 0.73 | 35.71 ± 0.41 | 59.66 ± 0.96 | 5.26 ± 1.10 | 1.09 ± 1.59 | 4.79 ± 0.14 | -5.87 ± 2.98 | 0.02 ± 0.03 | 4.88 ± 0.28 | 9.4 | n/a (oracle not built yet) |
| Q-learning (tuned) | -37.19 ± 1.75 | 35.19 ± 0.38 | 58.50 ± 1.71 | 6.66 ± 1.01 | 3.01 ± 2.84 | 4.66 ± 0.27 | -3.08 ± 6.05 | 0.02 ± 0.04 | 5.35 ± 0.22 | 9.5 | n/a (oracle not built yet) |
| Q-learning (selected: coarse grid) | -36.56 ± 0.84 | 36.51 ± 0.66 | 58.81 ± 0.85 | 3.16 ± 1.76 | 2.51 ± 1.41 | 4.46 ± 0.08 | 1.42 ± 1.80 | 0.00 ± 0.00 | 1.86 ± 0.43 | 9.3 | n/a (oracle not built yet) |
