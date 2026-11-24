### Total savings vs no battery (%) by split

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | Validation (2009) | Test (2010) | Winter week (Jan 18-24, 2010) | Summer week (Jul 5-11, 2010) | Winter weeks (Jan-Feb 2010) | Summer weeks (Jun-Jul 2010) | change val to test (pts) |
|---|---|---|---|---|---|---|---|
| No battery | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Rule-based | 0.0 | -4.9 | 7.4 | -20.5 | 1.6 | -17.8 | -4.9 |
| Greedy threshold | 9.6 | 7.8 | 11.0 | 2.0 | 10.9 | -1.8 | -1.8 |
| Q-learning (plan settings) | -2.2 ± 3.3 | -4.9 ± 2.2 | 1.3 ± 5.3 | -16.5 ± 4.8 | 1.1 ± 1.6 | -11.5 ± 2.2 | -2.7 |
| Q-learning (tuned) | 1.2 ± 1.9 | -1.7 ± 1.7 | 5.7 ± 6.2 | -12.3 ± 6.7 | 3.0 ± 2.8 | -5.6 ± 2.7 | -2.9 |
| Q-learning (selected: coarse grid) | 2.7 ± 1.7 | 1.5 ± 1.4 | 4.3 ± 3.8 | 0.2 ± 2.5 | 2.5 ± 1.4 | 1.0 ± 1.6 | -1.1 |
| PPO (deep RL) | 9.9 ± 0.7 | 7.1 ± 0.6 | 13.2 ± 0.5 | 9.0 ± 1.0 | 9.9 ± 0.6 | 7.0 ± 0.8 | -2.8 |
| DQN (deep RL) | 11.7 ± 2.5 | 10.5 ± 2.2 | 12.2 ± 4.0 | 12.6 ± 4.0 | 10.4 ± 2.7 | 12.1 ± 2.0 | -1.2 |
| PPO, peak input hidden (ablation) | 7.9 ± 2.8 | 5.8 ± 2.2 | 13.4 ± 1.3 | 9.5 ± 2.3 | 8.5 ± 2.7 | 6.3 ± 1.7 | -2.1 |
| DQN, peak input hidden (ablation) | 5.6 ± 2.6 | 3.4 ± 2.7 | 10.4 ± 5.5 | -2.3 ± 4.3 | 7.1 ± 3.1 | 1.2 ± 4.0 | -2.1 |
| Oracle, 5 power levels (best plan found) | 29.7 | 27.5 | 28.4 | 29.2 | 27.0 | 28.7 | -2.2 |
| Oracle (perfect foresight, upper bound) | 37.2 | 35.6 | 35.0 | 38.5 | 34.7 | 38.1 | -1.6 |

### Savings as a % of the oracle's savings, by split

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | Validation (2009) | Test (2010) | Winter week (Jan 18-24, 2010) | Summer week (Jul 5-11, 2010) | Winter weeks (Jan-Feb 2010) | Summer weeks (Jun-Jul 2010) |
|---|---|---|---|---|---|---|
| No battery | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | 0.1 | -13.7 | 21.1 | -53.3 | 4.6 | -46.7 |
| Greedy threshold | 25.7 | 21.9 | 31.3 | 5.1 | 31.4 | -4.7 |
| Q-learning (plan settings) | -5.9 ± 8.8 | -13.8 ± 6.1 | 3.7 ± 15.2 | -42.9 ± 12.4 | 3.1 ± 4.6 | -30.2 ± 5.8 |
| Q-learning (tuned) | 3.2 ± 5.0 | -4.7 ± 4.7 | 16.3 ± 17.6 | -32.1 ± 17.5 | 8.7 ± 8.2 | -14.7 ± 7.0 |
| Q-learning (selected: coarse grid) | 7.1 ± 4.6 | 4.3 ± 3.8 | 12.4 ± 10.8 | 0.5 ± 6.6 | 7.2 ± 4.1 | 2.5 ± 4.1 |
| PPO (deep RL) | 26.6 ± 1.8 | 20.0 ± 1.8 | 37.6 ± 1.5 | 23.5 ± 2.6 | 28.5 ± 1.7 | 18.5 ± 2.0 |
| DQN (deep RL) | 31.4 ± 6.7 | 29.6 ± 6.2 | 34.9 ± 11.5 | 32.9 ± 10.5 | 29.8 ± 7.7 | 31.7 ± 5.2 |
| PPO, peak input hidden (ablation) | 21.1 ± 7.5 | 16.2 ± 6.1 | 38.4 ± 3.8 | 24.7 ± 6.0 | 24.6 ± 7.8 | 16.6 ± 4.5 |
| DQN, peak input hidden (ablation) | 14.9 ± 6.9 | 9.7 ± 7.5 | 29.8 ± 15.8 | -6.0 ± 11.3 | 20.6 ± 9.0 | 3.3 ± 10.6 |
| Oracle, 5 power levels (best plan found) | 79.7 | 77.1 | 81.2 | 75.8 | 77.8 | 75.5 |
| Oracle (perfect foresight, upper bound) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Same, with leftover battery energy valued (terminal-adjusted)

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | Validation (2009) | Test (2010) | Winter week (Jan 18-24, 2010) | Summer week (Jul 5-11, 2010) | Winter weeks (Jan-Feb 2010) | Summer weeks (Jun-Jul 2010) |
|---|---|---|---|---|---|---|
| No battery | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | 2.0 | -11.7 | 23.2 | -51.8 | 6.0 | -44.8 |
| Greedy threshold | 28.0 | 24.4 | 33.2 | 8.7 | 32.5 | -2.5 |
| Q-learning (plan settings) | -5.8 ± 9.1 | -13.5 ± 6.3 | 3.3 ± 15.5 | -42.8 ± 12.7 | 2.8 ± 4.6 | -30.6 ± 6.0 |
| Q-learning (tuned) | 3.1 ± 5.1 | -4.7 ± 4.8 | 16.1 ± 17.9 | -34.1 ± 18.9 | 8.0 ± 8.4 | -15.8 ± 7.2 |
| Q-learning (selected: coarse grid) | 6.5 ± 4.9 | 3.6 ± 4.0 | 11.6 ± 10.9 | -1.0 ± 6.6 | 6.3 ± 4.1 | 1.4 ± 4.0 |
| PPO (deep RL) | 27.7 ± 1.9 | 21.1 ± 1.8 | 38.5 ± 1.7 | 23.4 ± 2.7 | 29.0 ± 1.7 | 19.2 ± 1.9 |
| DQN (deep RL) | 32.6 ± 7.0 | 31.0 ± 6.5 | 35.2 ± 11.6 | 33.6 ± 10.0 | 30.3 ± 7.8 | 32.8 ± 5.3 |
| PPO, peak input hidden (ablation) | 21.6 ± 7.9 | 16.7 ± 6.6 | 38.9 ± 3.7 | 24.5 ± 5.9 | 24.6 ± 7.8 | 16.7 ± 4.8 |
| DQN, peak input hidden (ablation) | 15.7 ± 7.8 | 10.4 ± 8.2 | 30.5 ± 16.8 | -5.9 ± 10.9 | 20.8 ± 9.6 | 3.3 ± 10.7 |
| Oracle, 5 power levels (best plan found) | 78.8 | 77.5 | 81.6 | 75.1 | 77.7 | 75.3 |
| Oracle (perfect foresight, upper bound) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Savings as a % of the 5-power-level oracle's savings, by split

Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. 'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. '% of oracle' is savings divided by the oracle's savings under the same accounting. All numbers are simulation results (REAL demand, SIMULATED battery and tariff).

| controller | Validation (2009) | Test (2010) | Winter week (Jan 18-24, 2010) | Summer week (Jul 5-11, 2010) | Winter weeks (Jan-Feb 2010) | Summer weeks (Jun-Jul 2010) |
|---|---|---|---|---|---|---|
| No battery | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Rule-based | 0.1 | -17.7 | 26.0 | -70.4 | 5.9 | -61.9 |
| Greedy threshold | 32.3 | 28.4 | 38.6 | 6.7 | 40.4 | -6.2 |
| Q-learning (plan settings) | -7.5 ± 11.1 | -17.9 ± 7.9 | 4.6 ± 18.8 | -56.5 ± 16.4 | 4.0 ± 5.9 | -40.0 ± 7.6 |
| Q-learning (tuned) | 4.1 ± 6.3 | -6.0 ± 6.1 | 20.1 ± 21.7 | -42.3 ± 23.1 | 11.1 ± 10.5 | -19.5 ± 9.3 |
| Q-learning (selected: coarse grid) | 9.0 ± 5.8 | 5.6 ± 5.0 | 15.3 ± 13.3 | 0.7 ± 8.7 | 9.3 ± 5.2 | 3.3 ± 5.5 |
| PPO (deep RL) | 33.4 ± 2.3 | 25.9 ± 2.3 | 46.3 ± 1.8 | 30.9 ± 3.4 | 36.7 ± 2.2 | 24.5 ± 2.7 |
| DQN (deep RL) | 39.4 ± 8.5 | 38.4 ± 8.0 | 43.0 ± 14.2 | 43.4 ± 13.9 | 38.4 ± 9.9 | 42.0 ± 7.0 |
| PPO, peak input hidden (ablation) | 26.5 ± 9.4 | 21.0 ± 7.9 | 47.3 ± 4.7 | 32.6 ± 7.9 | 31.6 ± 10.0 | 22.0 ± 6.0 |
| DQN, peak input hidden (ablation) | 18.8 ± 8.7 | 12.5 ± 9.7 | 36.7 ± 19.5 | -7.9 ± 14.9 | 26.5 ± 11.5 | 4.3 ± 14.1 |
| Oracle, 5 power levels (best plan found) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Oracle (perfect foresight, upper bound) | 125.5 | 129.7 | 123.2 | 131.9 | 128.6 | 132.5 |

### Demand statistics by split (REAL data)



| split | hours | mean demand (kW) | 95th pct (kW) | max (kW) | mean weekly peak (kW) |
|---|---|---|---|---|---|
| train | 17472 | 1.09 | 2.99 | 6.56 | 4.30 |
| validation | 8664 | 1.08 | 2.73 | 6.52 | 3.85 |
| test | 7536 | 1.06 | 2.58 | 5.63 | 3.50 |
| test_winter | 168 | 1.54 | 3.29 | 5.27 | 5.27 |
| test_summer | 168 | 0.75 | 1.70 | 2.59 | 2.59 |
