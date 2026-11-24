### Validation (2009)

Q-learning rows: mean ± std over 5 training seeds. Baselines are deterministic. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | avg episode reward | energy cost ($/wk) | total cost incl. demand charge ($/wk) | energy savings % | total savings % | peak (kW) | peak reduction % | violations / wk | full cycles / wk | train time (s) | % of oracle savings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No battery | -29.59 | 29.59 | 48.89 | 0.00 | 0.00 | 3.86 | 0.00 | 0.00 | 0.00 | 0 (no training) | n/a (oracle not built yet) |
| Rule-based | -79.80 | 24.11 | 48.87 | 18.51 | 0.03 | 4.95 | -28.29 | 4.96 | 3.13 | 0 (no training) | n/a (oracle not built yet) |
| Greedy threshold | -48.92 | 24.20 | 44.21 | 18.20 | 9.57 | 4.00 | -3.68 | 2.34 | 3.05 | 0 (no training) | n/a (oracle not built yet) |
| Q-learning (plan settings) | -31.90 ± 1.48 | 28.05 ± 0.38 | 49.97 ± 1.61 | 5.19 ± 1.29 | -2.21 ± 3.29 | 4.38 ± 0.27 | -13.57 ± 7.06 | 0.03 ± 0.05 | 4.76 ± 0.25 | 9.4 | n/a (oracle not built yet) |
| Q-learning (tuned) | -30.39 ± 1.02 | 27.42 ± 0.36 | 48.30 ± 0.91 | 7.31 ± 1.21 | 1.20 ± 1.86 | 4.17 ± 0.13 | -8.17 ± 3.39 | 0.04 ± 0.04 | 5.15 ± 0.14 | 9.5 | n/a (oracle not built yet) |
| Q-learning (selected: coarse grid) | -28.70 ± 0.87 | 28.65 ± 0.67 | 47.58 ± 0.84 | 3.18 ± 2.27 | 2.66 ± 1.72 | 3.79 ± 0.08 | 1.87 ± 2.13 | 0.00 ± 0.00 | 2.06 ± 0.58 | 9.3 | n/a (oracle not built yet) |
