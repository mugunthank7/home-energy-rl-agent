# Reinforcement Learning for Home Battery Energy Management

**ELET 6303, Final Project 3** | Mugunthan Kesavan | Repository: github.com/mugunthank7/home-energy-rl-agent

> **Real vs simulated.** Household demand is REAL: the UCI Individual Household Electric Power
> Consumption dataset (one French household, December 2006 to November 2010). The battery, the
> electricity tariff and the demand charge are SIMULATED. Every saving in this report is a simulation
> result, not a field result.

*Draft notes: every number below comes from files in `results/` and `reports/tables/`. Items marked
[YOU] need your input. Convert to Word or PDF for submission and insert the figures listed in the
appendix.*

---

## 1. Problem and motivation

Households that pay time-of-use prices and a demand charge can save money with a battery: charge when
electricity is cheap, discharge when it is expensive, and avoid setting a high monthly peak. The control
problem is sequential and uncertain. Charging now only pays off hours later, and the future demand is
unknown. Reinforcement learning (RL) is a natural fit, because an agent can learn a charging policy from
experience without a model of the household.

This project asks a practical question: **can tabular Q-learning learn a battery controller that beats
simple hand-written rules on real household demand, and how well does it generalize to a later year and
to other seasons?** We built the data pipeline, a Gymnasium environment, four controllers (no battery,
rule-based, greedy threshold, Q-learning), seven parameter sweeps, and a held-out test on 2010.

The main finding is a negative one, and we report it plainly: in this simulation, Q-learning with a
discretized state beats doing nothing only by a small margin and is clearly worse than a simple greedy
price-threshold rule.

## 2. Data and preprocessing

**Source.** UCI dataset id 235: 2,075,259 minute-level readings of global active power (kW) from one
household, December 2006 to November 2010.

**Cleaning** (`src/home_energy_rl_agent/data.py`, log in `data/processed/cleaning_log.json`):

1. The file uses `;` separators, dd/mm/yyyy dates and `?` for missing values. Date and time are combined
   into one timestamp, and `?` becomes missing.
2. 25,979 minutes (1.25%) were missing. Gaps of up to 3 hours were forward-filled; longer gaps were not
   guessed.
3. Minutes were averaged to hourly values. The mean kW over an hour equals the kWh used in that hour. An
   hour with any missing minute stays missing.
4. Calendar days without all 24 hours were dropped: 24 of 1,442 days, leaving 1,418 days (34,032 hours).

**Splits, by time and never shuffled.** Train: 2007-2008 (17,472 hours). Validation: 2009 (8,664 hours).
Test: 2010 (7,536 hours; the dataset ends in November 2010). Because dropped days leave holes, episodes
are only drawn from fully contiguous 7-day windows. Two held-out windows of 2010 are used for the seasonal
test: a winter week (January 18 to 24) and a summer week (July 5 to 11). July is used for summer
because the household was away for much of August, and the week of January 11 to 17 was avoided because
three of its days were missing.

**Exploratory analysis** (`reports/figures/eda_*.png`). Weekday demand peaks at about 07:00 and again at
19:00 to 21:00, and weekend demand is higher through the middle of the day. The evening peak overlaps the
simulated on-peak price window, which is what a battery can exploit.

**Distribution shift across splits** (table 1). Demand falls from year to year, and the peaks fall faster
than the average. This matters later for generalization.

| Split | Mean demand (kW) | 95th percentile (kW) | Mean weekly peak (kW) |
|---|---|---|---|
| Train (2007-08) | 1.09 | 2.99 | 4.30 |
| Validation (2009) | 1.08 | 2.73 | 3.85 |
| Test (2010) | 1.06 | 2.58 | 3.50 |
| Winter test week | 1.54 | 3.29 | 5.27 |
| Summer test week | 0.75 | 1.70 | 2.59 |

*Table 1. Demand statistics of the real data by split (`reports/tables/generalization.md`).*

## 3. Environment design

`EnergyEnv` is a Gymnasium environment. One step is one hour and one episode is 7 days (168 steps),
starting at a random valid point in the chosen split. All settings are in `configs/env.yaml` and
`configs/tariff.yaml`.

**Battery (simulated).** 10 kWh capacity, 3 kW maximum power, 90% round-trip efficiency applied as the
square root in each direction, charge level kept between 10% and 100%, starting at 50%. Power is measured on
the grid (AC) side. Discharge can never exceed the home's current demand, because there is no export. Any
discharge the home cannot use is cut and logged, but it is not penalized.

**Tariff (simulated, not a real utility).** Off-peak $0.08/kWh from 22:00 to 06:00; on-peak $0.30/kWh from
16:00 to 21:00 on weekdays; mid-peak $0.15/kWh otherwise. A demand charge of $5 per kW applies to the
highest hourly grid import in the episode.

**State.** Hour of day (as sine and cosine), day of week, battery charge level, current demand, tariff
period, demand in the last three hours, and the highest grid import so far in the episode.

**Actions (discrete, 5).** Discharge high, discharge low, idle, charge low, charge high, at 100% and 50%
of the 3 kW limit. Grid import is demand plus charging minus discharging.

**Reward.** Per step: minus the energy cost, minus a penalty on the peak the battery adds, minus a penalty
for violations, minus a small penalty for battery throughput. A violation is a request beyond the charge
limits or the power limit; the environment clips it and penalizes it.

**A deliberate change to the reward.** The plan's literal peak penalty (each rise in the episode's peak,
weight 0.5, violation weight 1) trained a poor agent. The default is now:

- the peak penalty counts only the peak the battery adds on top of the home's own demand peak (peak import
  minus peak demand, counted step by step). The home's own peak does not depend on any action, so this
  differs from the plain penalty by an action-independent constant. It leaves the best policy unchanged and
  removes a lot of noise.
- the peak weight equals the demand charge ($5 per kW), so the reward tracks the real bill.
- the violation weight is 10. At 1, the agent learned to break battery limits on purpose, because a clipped
  discharge that shaved a peak earned more than the penalty cost. This was found by inspecting where the
  violations occurred.

The reward also contains violation and throughput terms, so it is not exactly the bill. Controllers are
therefore compared on bill metrics, not on reward.

**Tests** (`tests/`, 71 passing). Energy conservation, charge level always in bounds, an idle battery
equals demand times price, identical results under the same seed, the equivalence of the reward variants,
and the data-cleaning rules.

## 4. Methods: controllers

All controllers share one interface (`act(observation) -> action`) and one evaluation function, which
runs the controller and a no-battery reference on identical episodes.

1. **No battery.** Always idle. All savings are measured against it.
2. **Rule-based.** At full rate, charge in off-peak hours while the battery is below 90%, and discharge in
   on-peak hours while it is above 15%.
3. **Greedy threshold.** Compare the current price with percentiles of the training prices. Charge gently
   (half rate) when cheap, and discharge in expensive hours at a rate that follows current demand.
4. **Tabular Q-learning.** One table entry per discretized situation and action. The plan's state grid is
   10 battery levels x 24 hours x 5 demand levels (quantiles of training demand) x 3 tariff periods = 3,600
   states. Updates use alpha for the step size and gamma for the discount, with epsilon-greedy exploration
   decaying from 1.0 to 0.05. Two details matter. A pair's first value is set from what was observed,
   not blended with a made-up zero. And untried actions are ignored when choosing, because with mostly
   negative rewards an untried action (value 0) would otherwise look better than every tried one.

Three Q-learning versions are compared in the final test:

| Version | alpha | gamma | State grid |
|---|---|---|---|
| Plan settings | 0.1 | 0.95 | 10 x 24 x 5 x 3 |
| Tuned | 0.02 | 0.9 | 10 x 24 x 5 x 3 |
| Selected | 0.1 | 0.95 | 5 x 6 x 3 x 3 (coarse) |

"Selected" follows a rule written down before the test split was touched: the sweep variant with the
highest mean validation total savings. "Tuned" came from single-seed validation scans. Both choices were
made on validation data only. The oracle upper bound (a linear program with perfect foresight) and deep RL
are not built yet.

## 5. Experiments

Everything runs from YAML configs in `configs/experiments/`, with results saved as JSON under `results/`.
Training uses the training split only; a run takes about 9.5 seconds for 3,000 episodes.

**Baseline.** alpha 0.1, gamma 0.95, epsilon decaying exponentially from 1.0 to 0.05 (reaching the floor
60% of the way through training), 3,000 episodes, 10 kWh battery, 10 x 24 x 5 x 3 grid.

**Seven sweeps, 5 seeds each (235 runs in total, validation split).** One factor changes at a time around
the baseline: learning rate (0.01, 0.05, 0.1, 0.3), discount (0.8, 0.9, 0.95, 0.99), epsilon schedule
(linear, exponential fast, exponential, exponential slow), training episodes (500, 1,500, 3,000, 6,000),
battery capacity (5, 10, 13.5, 20 kWh), state grid (coarse, medium, fine) and reward weights (four
combinations of cost, peak and cycle weights). A 5 x 4 grid of alpha and gamma gives the sensitivity
heatmap. All tables and figures are rebuilt with `scripts/make_report.py`.

**Final test.** Each Q-learning version is trained on 2007-2008 with 5 seeds, then scored on validation, on
100 sampled weeks of 2010, on the held-out winter and summer weeks, and on every Jan-Feb and Jun-Jul 2010
week that starts at midnight. Every controller sees the same weeks.

## 6. Results

### 6.1 Validation split (2009, 50 weeks)

| Controller | Total savings | Energy savings | Peak change | Violations / week |
|---|---|---|---|---|
| No battery | 0% | 0% | 0% | 0 |
| Rule-based | 0.03% | 18.5% | 28.3% higher | 4.96 |
| Greedy threshold | 9.6% | 18.2% | 3.7% higher | 2.34 |
| Q-learning, plan settings | -2.2 ± 3.3% | 5.2 ± 1.3% | 13.6 ± 7.1% higher | 0.03 |
| Q-learning, tuned | 1.2 ± 1.9% | 7.3 ± 1.2% | 8.2 ± 3.4% higher | 0.04 |
| Q-learning, selected | 2.7 ± 1.7% | 3.2 ± 2.3% | 1.9 ± 2.1% lower | 0.00 |

*Table 2. "Total savings" includes the demand charge. Q-learning rows are mean ± std over 5 seeds
(`reports/tables/final_metrics_validation.md`).*

The rule-based controller cuts the energy bill by 18.5% but raises the weekly peak by 28%, so the demand
charge cancels the saving. The greedy rule charges gently and nets 9.6%.

### 6.2 Parameter sweeps

Total savings on the validation split, mean ± std over 5 seeds (`reports/tables/sweep_*.md`):

| Factor | Values and total savings (%) |
|---|---|
| Learning rate alpha | 0.01: 1.5 ± 1.8 / 0.05: -1.2 ± 1.5 / 0.1: -2.2 ± 3.3 / 0.3: -2.0 ± 1.8 |
| Discount gamma | 0.8: -1.0 ± 1.2 / 0.9: -1.2 ± 1.3 / 0.95: -2.2 ± 3.3 / 0.99: -3.3 ± 1.1 |
| Epsilon schedule | linear: -2.9 ± 2.0 / fast: -1.7 ± 0.3 / exponential: -2.2 ± 3.3 / slow: -3.9 ± 2.9 |
| Training episodes | 500: -5.6 ± 1.7 / 1,500: -2.1 ± 1.6 / 3,000: -2.2 ± 3.3 / 6,000: -0.1 ± 0.9 |
| Battery capacity | 5 kWh: -0.7 ± 1.0 / 10: -2.2 ± 3.3 / 13.5: -1.9 ± 1.2 / 20: -1.1 ± 1.9 |
| State grid | coarse: 2.7 ± 1.7 / medium: -2.2 ± 3.3 / fine: -3.3 ± 0.6 |
| Reward weights | default: -2.2 ± 3.3 / cost-focused: -9.2 ± 0.4 / peak-focused: -1.8 ± 0.8 / cycle-averse: -1.6 ± 1.7 |

*Table 3.* The 5-seed std is 1 to 3 points, so many differences between neighbouring values are within
noise. The clear effects are: lower alpha is better (the alpha x gamma grid shows the same direction at every
gamma); more training helps; the coarse grid beats the fine grid; and a cost-focused reward is clearly worst
(energy cost down 7.3%, but peak 34.5% higher). Figures: `reports/figures/sweep_*_convergence.png`,
`sweep_*_final.png`, `sweep_grid_alpha_gamma_heatmap.png`.

Under the plan's literal reward the plan-settings agent scored -9.8 ± 0.8% total savings (3 seeds), with
peak 38% higher and 1.6 violations a week (`results/qlearning/plan_reward/`). That ablation is why the
reward was changed.

### 6.3 Test split (2010, 100 sampled weeks)

| Controller | Total cost ($/week) | Total savings | Peak (kW) | Violations / week | Full cycles / week |
|---|---|---|---|---|---|
| No battery | 45.31 | 0% | 3.45 | 0 | 0 |
| Rule-based | 47.51 | -4.9% | 4.95 | 5.57 | 3.05 |
| Greedy threshold | 41.78 | **7.8%** | 3.78 | 2.38 | 2.95 |
| Q-learning, plan settings | 47.53 ± 0.98 | -4.9 ± 2.2% | 4.16 ± 0.15 | 0.02 | 4.82 ± 0.24 |
| Q-learning, tuned | 46.06 ± 0.76 | -1.7 ± 1.7% | 3.99 ± 0.16 | 0.02 | 5.11 ± 0.16 |
| Q-learning, selected | 44.61 ± 0.62 | 1.5 ± 1.4% | 3.47 ± 0.07 | 0.00 | 1.84 ± 0.52 |

*Table 4 (`reports/tables/final_metrics_test.md`; figure `reports/figures/controllers_test.png`). Error
figures are std over 5 training seeds. The "% of oracle savings" column is not reported because the oracle
has not been built.*

The best learned controller saves about $0.70 a week, roughly one standard deviation from zero. The greedy
rule saves about $3.53 a week.

## 7. Analysis

**Why the learned controller struggles: the peak.** The demand charge depends on the highest hour of the
week, and the agent cannot see how close it is to the peak it has already set. It therefore charges at
full rate in hours that raise the peak. The evidence: the reward-weight sweep, where a cost-focused reward
gives the lowest total savings (-9.2%) and the highest peak; the rule-based controller, whose full-rate
charging raises the validation peak 28%; and the greedy rule, which avoids the problem with a fixed gentle
charge rate and wins. Adding peak bins to the Q-table state was tried during development and did not help at
this training length.

**Learning rate, noise and sparse data.** Weekly demand is noisy, so each reward is a noisy sample. A large
alpha lets single episodes drag the table around, and the greedy policy then chases noise. The alpha x gamma
grid is consistent with this: alpha 0.1 and 0.3 lose in every gamma column. The fine grid has about 90
training visits per visited state, against about 3,400 for the coarse grid, so each entry is estimated from
far less data. It also has the most violations (0.20 a week against 0.00).

**Discount factor.** Charging overnight only pays off hours later, so a higher gamma should help the agent
value it. Energy savings do rise with gamma (1.4%, 2.8%, 5.2% for gamma 0.8, 0.9, 0.95) and so does cycling,
but total savings do not improve, because the extra charging also raises peaks. At gamma 0.99 energy
savings drop (3.4%).

**Exploration versus exploitation.** The four epsilon schedules differ clearly in how quickly training
reward converges: about 1,000 episodes for the fast decay, 2,000 for the exponential, and 2,500 for the slow
and linear schedules. The final greedy policies are close (-1.7% to -3.9%, within the noise). The schedule
changes how fast the agent learns, not where it ends up.

**Generalization.** Every controller loses ground from validation to test, including the two that never
learn (rule-based 0.0% to -4.9%, greedy 9.6% to 7.8%). Part of the drop is therefore a property of the data,
not of learning: mean weekly peak demand falls from 4.30 kW in training to 3.50 kW in 2010, which makes it
easier for a 3 kW charge to set a new peak. The Q-learning versions drop by 2.7 (plan), 2.9 (tuned) and 1.1
(selected) points. The selected version was chosen because it was best on validation, so some of its
validation score was luck, and its drop is about what that would produce.

**Seasons.** The summer week has mean demand of 0.75 kW and a no-battery peak of 2.59 kW, below the 3 kW
charge power, so any full-rate charge sets a new peak. The rule-based controller's summer-week peak is 4.71
kW and it loses 20.5%; the plan-settings and tuned agents lose 16.5% and 12.3%; the greedy rule gains 2.0%;
the selected agent gains 0.2 ± 2.5%. Winter demand is higher (mean 1.54 kW), charging rarely sets a new
peak, and every controller saves (rule-based 7.4%, greedy 11.0%, selected 4.3 ± 3.8%). Single weeks are
noisy; the Jan-Feb and Jun-Jul rows in `reports/tables/generalization.md` are steadier but their windows
overlap and are not independent.

**What the learned policies look like** (`reports/figures/policy_heatmap_*.png`,
`representative_days.png`). The selected agent is idle in 65% of the states it visited, never charges at
full rate, and cycles 1.8 times a week against 4.8 for the plan settings. It generalizes best because it
does little, so it rarely makes a costly mistake, and it also leaves most of the possible saving unused
(energy savings of 2.9%, against 18.4% for the greedy rule). The plan-settings policy on the 3,600-state
grid looks noisy with no clear daily pattern.

**Limitations.**

- One household, one made-up tariff, one battery power limit, one test year. The result could look
  different with other tariffs, power limits or years.
- Tabular Q-learning with a hand-chosen discretization, trained on 2007-2008 only.
- The configs were chosen on validation data, so validation numbers for the chosen settings are optimistic.
  The test numbers are unbiased but come from one run of the protocol.
- The rule-based and greedy controllers were not tuned, so the comparison understates how good tuned rules
  could be.
- The oracle upper bound and deep RL are not built, so we cannot say how much of the achievable saving each
  controller captures.

## 8. Agentic extension

Not built in this submission. The planned design is a language-model layer that turns a plain-English
request ("keep a 40% reserve and cut my evening peak") into validated reward weights and limits, with an
offline fallback, and an MCP server that exposes backtests and comparisons as tools. [YOU: keep this
section as future work, or delete it and renumber.]

## 9. Contributions

This is an individual project. All design, implementation, experiments and writing were done by Mugunthan
Kesavan. [YOU: edit this paragraph to say in your own words who did what and what you verified.]

## 10. Conclusions and future work

On real household demand with a simulated battery and tariff, tabular Q-learning learned to cut the energy
bill but, with the plan's settings, did not beat doing nothing once the demand charge was included
(-2.2 ± 3.3% on validation, -4.9 ± 2.2% on test). Fixing the reward (bill-aligned, noise-reduced, stricter
violation penalty) and choosing a coarser state grid gave a small gain (+1.5 ± 1.4% on test), still far
behind a simple greedy price-threshold rule (+7.8%). The sweeps show where the learner is sensitive:
learning rate, grid resolution and the peak-related reward weight. Generalization to 2010 is weaker for every
controller because demand peaks are lower, and summer weeks are hardest for controllers that charge at full
power.

Future work, in order of expected value: (1) build the linear-programming oracle to turn every result
into a percentage of the achievable saving; (2) add the highest import so far to the learning state in a way
the table can use, or use a function approximator (DQN, PPO) that can handle it; (3) tune the simple rules
fairly; (4) test other tariffs, power limits and years; (5) add the language-model layer from section 8.

---

## Appendix A. Reproduce everything

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/make_eda.py
uv run python scripts/run_baselines.py
uv run python scripts/run_experiment.py all --workers 8
uv run python scripts/run_final.py
uv run python scripts/make_report.py
uv run pytest
```

## Appendix B. Figures to insert

| Figure | File |
|---|---|
| Daily and weekly demand profile, monthly mean, load duration, missing data | `reports/figures/eda_*.png` |
| Environment check episode | `reports/figures/env_check_episode.png` |
| Reward curves with seed bands, per sweep | `reports/figures/sweep_*_convergence.png` |
| Final savings, peak, violations per sweep | `reports/figures/sweep_*_final.png` |
| Alpha x gamma sensitivity heatmap | `reports/figures/sweep_grid_alpha_gamma_heatmap.png` |
| Cost, peak and savings by controller (validation, test) | `reports/figures/controllers_validation.png`, `controllers_test.png` |
| Winter and summer representative days | `reports/figures/representative_days.png` |
| Q-table policy heatmaps | `reports/figures/policy_heatmap_baseline.png`, `policy_heatmap_selected.png` |
| Parameter record | `reports/tables/parameter_record.md` |
