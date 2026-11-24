# Sweep findings (validation split, 2009)

All numbers come from `results/sweep_*/` (235 runs, 5 seeds each, mean ± std over seeds) and are
tabulated in `reports/tables/`. Figures are in `reports/figures/`. Demand is REAL; battery and tariff
are SIMULATED. Every sweep changes one factor around the plan's baseline (alpha 0.1, gamma 0.95, epsilon
decaying exponentially, 3000 episodes, 10 kWh battery, 10x24x5x3 grid, bill-aligned reward) except the
alpha x gamma grid.

**How to read the noise:** with 5 seeds the std of total savings is typically 1 to 3 percentage points.
Differences smaller than about 2 points between two variants are not reliable. Larger, consistent
effects are marked below. Train times were measured with 8 jobs running in parallel, so they are
inflated compared with a single sequential run (about 9.5 s for 3000 episodes).

## Headline

- The plan's baseline Q-learning does **not** beat doing nothing on the total bill: **-2.2 ± 3.3%** (5 seeds).
  It does cut energy cost (+5.2 ± 1.3%) but raises the weekly peak (13.6 ± 7.1% higher), and the demand
  charge cancels the energy saving.
- The best settings found reach roughly **+1 to +3%** total savings (coarse grid +2.7 ± 1.7%, alpha 0.01 with
  gamma 0.95 +1.5 ± 1.8%). All are far below the simple greedy-threshold controller (+9.6%).
- The earlier "tuned" setting (alpha 0.02, gamma 0.9) scored +2.6 ± 0.7% over 3 seeds (42-44). Over all 5
  seeds, the same cell of the grid scores **+1.2 ± 1.9%**. The 3-seed number was optimistic (selection bias
  plus a small sample), so quote the 5-seed figure.

## One factor at a time

| Sweep | What the data shows | Confidence |
|---|---|---|
| **Learning rate alpha** (0.01, 0.05, 0.1, 0.3) | Total savings fall as alpha rises: +1.5, -1.2, -2.2, -2.0 %. The alpha x gamma grid shows the same direction at every gamma: alpha 0.1 and 0.3 are negative in every column, alpha 0.01 and 0.02 are the best rows. | Direction is consistent; individual gaps are within noise |
| **Discount gamma** (0.8, 0.9, 0.95, 0.99) | Energy savings rise with gamma up to 0.95 (1.4, 2.8, 5.2%) and cycling rises with it (1.6, 3.1, 4.8 cycles/week), then energy savings drop at 0.99 (3.4%). Total savings do not improve with gamma (-1.0, -1.2, -2.2, -3.3 %). | Energy-savings trend is clear; total-savings differences are within noise |
| **Epsilon schedule** (linear, fast, exponential, slow) | Convergence speed differs a lot: training reward reaches its plateau after about 1000 episodes (fast), about 2000 (exponential) and about 2500 (slow and linear). The final greedy policies are close: -2.9, -1.7, -2.2, -3.9 %. | Convergence speed: clear. Final quality: not distinguishable |
| **Training episodes** (500, 1500, 3000, 6000) | More training helps: -5.6, -2.1, -2.2, -0.1 %. At 500 episodes violations rise to 0.21/week and the peak is 20.9% higher. Savings were still improving at 3000 episodes. | Clear at 500 vs 6000; 1500 vs 3000 is a tie |
| **Reward weights** | Cost-focused (peak weight 0.5): -9.2 ± 0.4% total, energy +7.3% but peak +34.5% higher. Peak-focused (15) -1.8%; cycle-averse (0.2) -1.6% with the fewest cycles (2.4/week); default -2.2%. | Cost-focused being worst: clear. The other three: tie |
| **Battery capacity** (5, 10, 13.5, 20 kWh) | No clear trend in total savings (-0.7, -2.2, -1.9, -1.1 %). Energy savings are highest at 20 kWh (6.6%) and cycles per week fall as capacity grows (4.7, 4.8, 3.8, 3.0). | Cycles trend: clear. Savings: not distinguishable |
| **State discretization** (coarse, medium, fine) | Coarser is better: +2.7 ± 1.7, -2.2 ± 3.3, -3.3 ± 0.6 %. The fine grid also has the most violations (0.20/week vs 0.00 coarse). | Coarse vs fine: clear |

## Why (explanations consistent with the numbers)

These are interpretations. Where a number backs them up it is quoted; the rest are hypotheses that this
project did not test directly.

- **Fine grids learn worse because of sparse visits.** Training uses 504,000 steps. The coarse grid visited
  150 states (about 3,400 steps per visited state), the medium grid 1,447 (about 350) and the fine grid
  5,696 (about 90). Each table entry in the fine grid is estimated from far less data, and it has the most
  violations.
- **High alpha is unstable under noisy demand.** Weekly demand is noisy, so each reward is a noisy sample.
  A large alpha lets single episodes drag the table around, and the greedy policy then chases noise.
  This fits the grid: the two highest alphas lose in every gamma column. The cost of low alpha is slower
  learning, which is why savings improve with more episodes.
- **Gamma matters for energy cost, not for the bill.** Charging overnight pays off hours later, so a
  higher gamma lets the agent value it: energy savings rise from 1.4% to 5.2% as gamma goes from 0.8 to
  0.95. But more cycling also means more charging at 3 kW on top of the home's own demand, which raises
  peaks. At 0.99 the learning target is dominated by distant, noisy estimates and energy savings drop.
- **The peak is the whole problem.** The agent never sees how close it is to the peak it has already set,
  so it charges at full rate in hours that raise the weekly peak. When the reward barely penalizes peaks
  (peak weight 0.5), it does this the most (peak +34.5%) and the bill gets worse by 9.2%. Greedy-threshold
  avoids this with a fixed gentle charge rate and gets +9.6%.
- **Exploration speed vs final quality.** A faster epsilon decay gets to a good policy sooner, because
  the agent spends more episodes exploiting. All four schedules end at similar quality, so the schedule
  changes how quickly the agent learns, not where it ends up (within this noise).
- **More capacity does not buy more savings here.** The power limit stays at 3 kW, so a bigger battery
  gives longer duration, not stronger peak shaving, and the agent cycles it less (3.0 cycles/week at 20 kWh).

## Limits

- Everything above is the validation split. The 2010 test split has not been used.
- One simulated household, one simulated tariff, one battery power limit. Savings are simulation results.
- Sweeps change one factor at a time around a baseline that itself is poor, so interactions are only
  covered for alpha x gamma.
- The best single settings were picked by looking at these same validation results, so their numbers
  are optimistic. Only the 2010 test numbers will be unbiased.


## Does seeing the peak help the tabular agent? (`sweep_peak_bins`)

The deep agents see the highest grid import so far this week, and the earlier sweeps suggested the peak is what
the tabular agent gets wrong. This sweep adds that input to the Q-table as extra state bins (train-demand
quantiles), 5 seeds each, on the validation split:

| Variant | Total savings | Violations / week | Cycles / week |
|---|---|---|---|
| no peak input (3,600 states) | -2.2 ± 3.3% | 0.03 ± 0.05 | 4.8 |
| 3 peak bins (10,800 states) | -1.8 ± 1.5% | 0.29 ± 0.11 | 6.1 |
| 5 peak bins (18,000 states) | -3.0 ± 1.7% | 0.54 ± 0.11 | 6.2 |

No improvement, and more violations. Consistent with the sparse-visits finding above: every added bin splits the
data thinner. The deep agents, which can generalize across similar peak levels, do benefit from the same input
(`docs/GENERALIZATION.md`).
