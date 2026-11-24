### Total savings vs no battery (%) by split

Q-learning rows: mean ± std over 5 training seeds. Baselines are deterministic. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | Validation (2009) | Test (2010) | Winter week (Jan 18-24, 2010) | Summer week (Jul 5-11, 2010) | Winter weeks (Jan-Feb 2010) | Summer weeks (Jun-Jul 2010) | change val to test (pts) |
|---|---|---|---|---|---|---|---|
| No battery | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Rule-based | 0.0 | -4.9 | 7.4 | -20.5 | 1.6 | -17.8 | -4.9 |
| Greedy threshold | 9.6 | 7.8 | 11.0 | 2.0 | 10.9 | -1.8 | -1.8 |
| Q-learning (plan settings) | -2.2 ± 3.3 | -4.9 ± 2.2 | 1.3 ± 5.3 | -16.5 ± 4.8 | 1.1 ± 1.6 | -11.5 ± 2.2 | -2.7 |
| Q-learning (tuned) | 1.2 ± 1.9 | -1.7 ± 1.7 | 5.7 ± 6.2 | -12.3 ± 6.7 | 3.0 ± 2.8 | -5.6 ± 2.7 | -2.9 |
| Q-learning (selected: coarse grid) | 2.7 ± 1.7 | 1.5 ± 1.4 | 4.3 ± 3.8 | 0.2 ± 2.5 | 2.5 ± 1.4 | 1.0 ± 1.6 | -1.1 |

### Demand statistics by split (REAL data)



| split | hours | mean demand (kW) | 95th pct (kW) | max (kW) | mean weekly peak (kW) |
|---|---|---|---|---|---|
| train | 17472 | 1.09 | 2.99 | 6.56 | 4.30 |
| validation | 8664 | 1.08 | 2.73 | 6.52 | 3.85 |
| test | 7536 | 1.06 | 2.58 | 5.63 | 3.50 |
| test_winter | 168 | 1.54 | 3.29 | 5.27 | 5.27 |
| test_summer | 168 | 0.75 | 1.70 | 2.59 | 2.59 |
