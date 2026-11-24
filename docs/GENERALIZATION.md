# Generalization: train on 2007-2008, test on 2010

All numbers come from `results/final/records.json` (tables in `reports/tables/final_metrics_*.md` and
`reports/tables/generalization.md`). Demand is REAL; battery and tariff are SIMULATED, so these are
simulation results. Q-learning rows are mean ± std over 5 training seeds. "Total savings" includes the
demand charge. The 2010 test split was used only after the three Q-learning configs were fixed from the
validation sweeps (the `selected` config follows a rule written down beforehand: the best mean validation
total savings across all sweep variants, which was the coarse state grid).

## Headline (test = 100 sampled weeks of 2010)

| Controller | Total cost ($/week) | Total savings | Peak (kW) | Violations/wk |
|---|---|---|---|---|
| No battery | 45.31 | 0% | 3.45 | 0 |
| Rule-based | 47.51 | -4.9% | 4.95 | 5.6 |
| Greedy threshold | 41.78 | **+7.8%** | 3.78 | 2.4 |
| Q-learning, plan settings | 47.53 ± 0.98 | -4.9 ± 2.2% | 4.16 ± 0.15 | 0.02 |
| Q-learning, tuned | 46.06 ± 0.76 | -1.7 ± 1.7% | 3.99 ± 0.16 | 0.02 |
| Q-learning, selected (coarse grid) | 44.61 ± 0.62 | +1.5 ± 1.4% | 3.47 ± 0.07 | 0.00 |
| Oracle (perfect foresight, upper bound) | 29.17 | +35.6% | 1.55 | 0 |

- The best learned controller saves about $0.70 a week (+1.5 ± 1.4%, roughly one standard deviation
  from zero). The greedy-threshold rule saves about $3.53 a week. On this simulation, RL does not beat
  the simple greedy rule.
- Only the coarse-grid Q-learning agent beats doing nothing on test, and by a small margin.
- Measured against the oracle, the best learned controller captures 4.3 ± 3.8% of the achievable saving and
  the greedy rule 21.9% (see the next section).

## Validation to test: the drop

| Controller | Validation (2009) | Test (2010) | Change (points) |
|---|---|---|---|
| Rule-based | 0.0% | -4.9% | -4.9 |
| Greedy threshold | 9.6% | 7.8% | -1.8 |
| Q-learning, plan settings | -2.2 ± 3.3% | -4.9 ± 2.2% | -2.7 |
| Q-learning, tuned | 1.2 ± 1.9% | -1.7 ± 1.7% | -2.9 |
| Q-learning, selected | 2.7 ± 1.7% | 1.5 ± 1.4% | -1.1 |

Every controller loses ground on 2010, including the two that never learn. So part of the drop is a
property of the data and tariff, not of learning. Two things are consistent with the numbers:

- **Distribution shift in the demand itself.** Mean weekly peak demand falls from 4.30 kW (train) to
  3.85 kW (validation) to 3.50 kW (test), and the 95th percentile from 2.99 to 2.73 to 2.58 kW. Lower,
  flatter demand makes a 3 kW charge more likely to set a new peak, which the demand charge punishes.
- **Selection effect.** The selected config was chosen because it was best on validation, so some of its
  validation score is luck. Its drop (-1.1 points) is about what that would produce.

## Held-out seasons

| Controller | Winter week (Jan 18-24) | Summer week (Jul 5-11) | Winter weeks (Jan-Feb) | Summer weeks (Jun-Jul) |
|---|---|---|---|---|
| Rule-based | 7.4% | -20.5% | 1.6% | -17.8% |
| Greedy threshold | 11.0% | 2.0% | 10.9% | -1.8% |
| Q-learning, plan settings | 1.3 ± 5.3% | -16.5 ± 4.8% | 1.1 ± 1.6% | -11.5 ± 2.2% |
| Q-learning, tuned | 5.7 ± 6.2% | -12.3 ± 6.7% | 3.0 ± 2.8% | -5.6 ± 2.7% |
| Q-learning, selected | 4.3 ± 3.8% | 0.2 ± 2.5% | 2.5 ± 1.4% | 1.0 ± 1.6% |

Single weeks are very noisy (seed std of 4 to 7 points). The season rows average about 60 windows that
start at midnight on each day, so they are steadier, but those windows overlap heavily and are not
independent samples.

**Summer is where the controllers that charge hard break down.** The summer week has mean demand of
0.75 kW and a no-battery peak of 2.59 kW, below the 3 kW maximum charge power. Any full-rate charge
therefore sets a new, higher peak. The data show it: the rule-based controller's weekly peak is 4.71 kW
against 2.59 kW with no battery, giving -20.5% total savings. The greedy rule charges at half power and
stays near break-even (+2.0%).

**Winter is easier.** Demand is higher (mean 1.54 kW, week peak 5.27 kW), so charging rarely sets a new
peak and every controller saves: rule-based +7.4%, greedy +11.0%, selected Q-learning +4.3 ± 3.8%.

## What the learned policies look like

Greedy-action shares over the states each agent visited in training (weighted by visits, mean over
5 seeds). These come from the saved Q-tables, not from a separate experiment:

| Config | discharge high | discharge low | idle | charge low | charge high |
|---|---|---|---|---|---|
| plan settings | 12.9% | 22.1% | 49.7% | 13.8% | 1.4% |
| tuned | 14.6% | 26.3% | 43.4% | 14.5% | 1.3% |
| selected (coarse) | 6.7% | 18.4% | 65.4% | 9.4% | 0.0% |

The selected agent is the most conservative one: idle in two thirds of its states, no full-rate charging,
and 1.8 full cycles a week against 4.8 for the plan settings. That is consistent with it generalizing
best: it does little, so it rarely makes a costly mistake, and it also leaves most of the possible saving
untouched (energy savings of only 2.9% against 18.4% for the greedy rule). The policy heatmaps
(`reports/figures/policy_heatmap_*.png`) show the coarse policy as large readable blocks, while the
plan-settings policy on the 3,600-state grid looks noisy with no clear daily pattern.

## Why performance drops, and limits of this analysis

- **Distribution shift and seasons:** supported by the demand statistics and the summer-week peaks above.
- **Discretization limits:** a coarse grid is stable but cannot express fine timing; a fine grid has too
  few visits per state (see `docs/SWEEP_FINDINGS.md`) and its policies fail on the unusual summer demand.
  This is an interpretation consistent with the data, not a controlled test.
- **Only one training window and one test year.** The test year is one household and one made-up tariff.
  The result could look different with other tariffs, battery power limits, or years.
- **Training on 2007-2008 only.** Retraining on 2009 as well (which a real deployment would do) was not
  tried.
- **The simple rules were not tuned.** Greedy and rule-based settings come straight from
  `configs/controllers.yaml`, so the comparison understates how good tuned rules could be too.


## The oracle: how much is left on the table

The oracle (`src/home_energy_rl_agent/controllers/oracle.py`) is a linear program. For each week it is
given the full demand and price series in advance and chooses continuous charge and discharge power to
minimize the real bill (energy plus the demand charge), under the same battery limits as the environment.
Replaying its plans through the simulator's own battery model reproduces the same charge levels and costs, and
tests check that its cost is never above any baseline's on the same weeks. It is not deployable, because it
needs the future.

| Test (2010) | Oracle | Greedy threshold | Q-learning, selected |
|---|---|---|---|
| Total cost ($/week) | 29.17 | 41.78 | 44.61 ± 0.62 |
| Total savings | +35.6% | +7.8% | +1.5 ± 1.4% |
| Share of the oracle's savings | 100% | 21.9% | 4.3 ± 3.8% |
| Weekly peak (kW) | 1.55 | 3.78 | 3.47 ± 0.07 |
| Full cycles per week | 5.2 | 2.9 | 1.8 ± 0.5 |

All shares by split are in `reports/tables/generalization.md`. On validation the greedy rule captures 25.7%
and the selected Q-learner 7.1 ± 4.6%. In the held-out summer week the oracle still saves 38.5%, while the
learned controllers save about nothing, so there is room even where fixed rules fail.

**Where the oracle's saving comes from.** On test, the no-battery weekly peak of 3.45 kW costs $17.25 in
demand charge ($5 per kW) and the energy costs $28.04. The oracle cuts the peak to 1.55 kW (about $9.5 a
week) and the energy cost by 23.6% (about $6.6 a week). So roughly 59% of its saving is peak shaving and 41%
is shifting energy to cheap hours. It does this by keeping grid import almost flat (about 0.9 kW through the
summer test day), which `reports/figures/representative_days.png` shows as a dashed line. This agrees with the
earlier finding that the peak is what the other controllers get wrong: the greedy rule and the learners keep
the peak at 3.5 to 4.2 kW.

**Caveats on the bound.**
- It uses continuous power, while every learned controller has 5 power levels. The next section separates the
  effect of the action set from the effect of foresight.
- It assumes exact knowledge of the whole week. No real controller can reach 100%.
- The shares for Q-learning have wide error bars (a few points) because the savings themselves are small.

## The oracle restricted to the 5 power levels

The continuous oracle can pick any power. Every controller here picks one of 5 levels (discharge or charge at
full or half power, or idle). `controllers/discrete_oracle.py` solves the same perfect-foresight problem with
that restriction as a mixed-integer program: each hour it must choose one of the 5 actions, modelled exactly as
the environment treats them (a discharge is cut to the current demand; a request the battery cannot cover or
fit counts as a violation, so it is excluded). Replaying its plans through the real environment reproduces the
predicted cost to four decimals with zero violations, and tests check this and that small changes to a plan
never make it cheaper.

| Test (2010, 100 weeks) | Continuous oracle | 5-level oracle | Greedy threshold | Q-learning, selected |
|---|---|---|---|---|
| Total cost ($/week) | 29.17 | 32.86 | 41.78 | 44.61 ± 0.62 |
| Total savings | +35.6% | +27.5% (proven ceiling 28.7%) | +7.8% | +1.5 ± 1.4% |
| Share of the 5-level oracle's savings | 129.7% | 100% | 28.4% | 5.6 ± 5.0% |
| Weekly peak (kW) | 1.55 | 2.32 | 3.78 | 3.47 ± 0.07 |
| Full cycles per week | 5.2 | 5.5 | 2.9 | 1.8 ± 0.5 |

- **The action set costs about 8 points of savings** (35.6% to 27.5%, or 6.9 points if the true 5-level optimum
  is at its ceiling). That is roughly a quarter (24%) of the 34.1 points between the best learner and the
  continuous oracle. The other roughly three quarters is what learning with no foresight costs.
- **Even with 5 levels, a lot is possible.** The 5-level oracle saves 27.5% against the greedy rule's 7.8% and
  the best learner's 1.5%. In the validation split the figures are 29.7%, 9.6% and 2.7%. So the best learned
  controller reaches only about 5% of what its own action set allows.
- **It still shaves the peak** (3.45 kW to 2.32 kW), so the peak part of the problem is largely reachable with
  5 levels, which is the part the learners and the greedy rule miss.
- **Solution quality.** The plan is the best the solver found, so its savings are a lower estimate of the 5-level
  optimum; the proven ceiling is the solver's dual bound. On test the two differ by 1.3 points (27.5% and 28.7%).
  Within the 30-second limit per week, 50% of the test weeks solved to the 0.2% tolerance (76% on validation); the
  rest are within the plan-to-ceiling range above. The shares quoted are against the plan found, so they could be
  overstated by up to about 5% in relative terms. The same MILP with leftover battery energy valued gives 27.0%
  on test, consistent with the continuous case.
- Per split (plan found): validation 29.7%, test 27.5%, winter week 28.4%, summer week 29.2%. All shares by
  split are in `reports/tables/generalization.md`.

## Leftover battery energy (terminal value)

The environment does not value the energy left in the battery at the end of a week, and the controllers end
very differently. Mean change in stored energy per test week, from a 5 kWh start: rule-based +4.25 kWh, greedy
+3.84, Q-learning plan settings +1.07, tuned +0.04, selected -1.47. A controller that ends full has paid for
energy it did not use; one that ends emptier used its free starting charge. As a check, every controller and
the oracle were also scored with the leftover energy valued at the cost of buying it back off-peak
($0.08 per kWh over the 0.949 charging efficiency, about $0.084 per stored kWh).

| Test (2010) total savings | Plain | Terminal-adjusted |
|---|---|---|
| Rule-based | -4.9% | -4.1% |
| Greedy threshold | +7.8% | +8.5% |
| Q-learning, plan settings | -4.9 ± 2.2% | -4.7 ± 2.2% |
| Q-learning, tuned | -1.7 ± 1.7% | -1.7 ± 1.7% |
| Q-learning, selected | +1.5 ± 1.4% | +1.3 ± 1.4% |
| Oracle | +35.6% | +34.9% |

No ranking changes. The adjustment trims the selected Q-learner's margin over doing nothing from 1.5 to 1.3
points, which is still only about one standard deviation. The adjusted oracle is a separate linear program
that also pays for energy it uses up, so it remains a valid bound under the adjusted metric. The plain metric
is the headline throughout because it is the one the plan defines; the adjusted one is a robustness check.
