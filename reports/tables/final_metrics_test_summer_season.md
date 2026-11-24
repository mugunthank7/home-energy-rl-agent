### Summer weeks (Jun-Jul 2010)

Q-learning rows: mean ± std over 5 training seeds. Baselines are deterministic. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | train time (s) | % of oracle savings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -20.81 | 20.81 | 35.51 | 0.00 | 0.00 | 2.94 | 0.00 | 0.00 | 0.00 | 0 (no training) | n/a (oracle not built yet) |
| Rule-based | -78.25 | 17.05 | 41.83 | 18.08 | -17.78 | 4.95 | -68.58 | 5.07 | 2.34 | 0 (no training) | n/a (oracle not built yet) |
| Greedy threshold | -35.82 | 17.06 | 36.14 | 18.02 | -1.77 | 3.82 | -29.80 | 1.39 | 2.22 | 0 (no training) | n/a (oracle not built yet) |
| Q-learning (plan settings) | -25.91 ± 0.80 | 20.15 ± 0.23 | 39.60 ± 0.78 | 3.18 ± 1.11 | -11.50 ± 2.19 | 3.89 ± 0.12 | -32.30 ± 4.11 | 0.01 ± 0.01 | 4.75 ± 0.19 | 9.4 | n/a (oracle not built yet) |
| Q-learning (tuned) | -24.10 ± 1.20 | 18.92 ± 0.22 | 37.49 ± 0.95 | 9.10 ± 1.05 | -5.59 ± 2.68 | 3.71 ± 0.21 | -26.39 ± 7.04 | 0.03 ± 0.04 | 5.01 ± 0.23 | 9.5 | n/a (oracle not built yet) |
| Q-learning (selected: coarse grid) | -20.82 ± 0.62 | 20.36 ± 0.59 | 35.17 ± 0.56 | 2.21 ± 2.85 | 0.95 ± 1.57 | 2.96 ± 0.06 | -0.82 ± 2.12 | 0.00 ± 0.00 | 1.74 ± 0.75 | 9.3 | n/a (oracle not built yet) |
