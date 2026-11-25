# Reinforcement Learning for Home Battery Energy Management

**ELET 6303, Final Project 3** | Mugunthan Kesavan | Repository: github.com/mugunthank7/home-energy-rl-agent

> **Real vs simulated.** Household demand is REAL: the UCI Individual Household Electric Power
> Consumption dataset (one French household, December 2006 to November 2010). The battery, the
> electricity tariff and the demand charge are SIMULATED. Every saving in this report is a simulation
> result, not a field result.

## 1. Problem and motivation

Households that pay time-of-use prices and a demand charge can save money with a battery: charge when
electricity is cheap, discharge when it is expensive, and avoid setting a high weekly peak. The problem is
sequential and uncertain, since charging now only pays off hours later and future demand is unknown, so
reinforcement learning (RL) is a natural fit.

This project asks: **can RL learn a battery controller that beats simple hand-written rules on real household
demand, and how well does it generalize to a later year and to other seasons?** We built a data pipeline, a
Gymnasium environment, three non-learning controllers, tabular Q-learning, two deep RL agents (PPO and DQN),
eight parameter sweeps, two optimal-control oracles as upper bounds, and a held-out test on 2010.

The answer depends on the learner. **Tabular Q-learning**, with the plan's 4-feature state table, beats doing
nothing only by a small margin and is clearly worse than a simple greedy price-threshold rule. **Deep RL does
much better**: on the 2010 test weeks DQN saves 10.5 ± 2.2% against 7.8% for the greedy rule (four of its five
seeds are above the rule) and PPO saves 7.1 ± 0.6%, roughly matching it. Both keep the weekly peak below the
no-battery level and stay positive in summer, where the rule-based controller loses money. Hiding the
peak-so-far input costs DQN about 7 points, so seeing the peak is a large part of why it wins. An oracle with
perfect foresight saves about 36% (27.5% when limited to the same 5 power levels), so the best learned
controller is well short of the achievable saving.

## 2. Data and preprocessing

The data is UCI dataset id 235: 2,075,259 minute-level readings of global active power (kW) from one household.
Missing values (25,979 minutes, 1.25%) were handled by forward-filling gaps of up to 3 hours, averaging to hourly
kW (equal to kWh per hour), and dropping the 24 of 1,442 calendar days that still had gaps, leaving 1,418 days
(34,032 hours). The cleaning log is in `data/processed/cleaning_log.json`.

Splits are by time and never shuffled: train 2007-2008 (17,472 hours), validation 2009 (8,664) and test 2010
(7,536; the data ends in November 2010). Episodes are drawn only from fully contiguous 7-day windows. A winter
week (January 18 to 24) and a summer week (July 5 to 11) of 2010 are held out for the seasonal test; July is used
because the household was away for much of August, and January 11 to 17 had three missing days.

**Exploratory analysis.** Weekday demand peaks at about 07:00 and again at 19:00 to 21:00 (Figure 1), and the
evening peak overlaps the simulated on-peak price window, which a battery can exploit. Demand falls from year to
year and the peaks fall faster than the average (Table 1), which matters later for generalization.

| Split | Mean demand (kW) | 95th percentile (kW) | Mean weekly peak (kW) |
|---|---|---|---|
| Train (2007-08) | 1.09 | 2.99 | 4.30 |
| Validation (2009) | 1.08 | 2.73 | 3.85 |
| Test (2010) | 1.06 | 2.58 | 3.50 |
| Winter test week | 1.54 | 3.29 | 5.27 |
| Summer test week | 0.75 | 1.70 | 2.59 |

*Table 1. Demand statistics of the real data by split.*

## 3. Environment design

`EnergyEnv` is a Gymnasium environment: one step is one hour and one episode is 7 days (168 steps), starting at
a random valid point of the chosen split. Settings are in `configs/env.yaml` and `configs/tariff.yaml`.

**Battery (simulated).** 10 kWh, 3 kW maximum power, 90% round-trip efficiency (square root each way), charge
kept between 10% and 100%, starting at 50%. Discharge cannot exceed the home's current demand (no export).

**Tariff (simulated, not a real utility).** Off-peak $0.08/kWh from 22:00 to 06:00, on-peak $0.30/kWh from 16:00
to 21:00 on weekdays, mid-peak $0.15/kWh otherwise, plus a demand charge of $5 per kW on the highest hourly grid
import of the episode.

**State and actions.** The observation holds hour (sine and cosine), day of week, charge level, current demand,
tariff period, the last three hours of demand and the highest grid import so far this week. There are five
discrete actions: discharge high, discharge low, idle, charge low and charge high (100% and 50% of 3 kW). Grid
import is demand plus charging minus discharging.

**Reward.** Minus the energy cost, minus a penalty on the peak the battery adds, minus a penalty for limit
violations (a request beyond the charge or power limits, which the environment clips), minus a small penalty
for battery throughput. The plan's literal peak penalty (each rise in the episode's peak, weight 0.5, violation
weight 1) trained a poor agent, so the default was changed. The peak penalty now counts only the peak the battery
adds on top of the home's own demand peak. That peak does not depend on any action, so the best policy is
unchanged but the signal is far less noisy. Its weight equals the demand charge ($5 per kW), so the reward
tracks the real bill. The violation weight is 10: at 1, the agent learned to break battery limits on purpose,
because a clipped discharge that shaved a peak earned more than the penalty cost. Because the reward also holds
violation and throughput terms, controllers are compared on bill metrics, not on reward.

**Leftover energy.** The bill does not value the energy left in the battery at the end of a week, and
controllers end very differently (per test week, from a 5 kWh start: rule-based +4.25 kWh, greedy +3.84, selected
Q-learning -1.47). We therefore also report a terminal-adjusted cost that values leftover energy at the off-peak
price of buying it back (about $0.084 per stored kWh). It changes no ranking, so the plain bill stays the
headline. Tests (103 passing) cover energy conservation, charge bounds, an idle battery equalling demand times
price, and reproducibility under a seed.

## 4. Methods: controllers

All controllers share one interface and one evaluation function, which runs the controller and a no-battery
reference on identical episodes.

1. **No battery.** Always idle; all savings are measured against it.
2. **Rule-based.** At full rate, charge in off-peak hours while the battery is below 90% and discharge in on-peak
   hours while it is above 15%.
3. **Greedy threshold.** Compare the current price with percentiles of training prices; charge gently (half rate)
   when cheap and discharge when expensive at a rate that follows current demand.
4. **Tabular Q-learning.** One table entry per discretized situation and action. The plan's grid is 10 battery
   levels x 24 hours x 5 demand levels x 3 tariff periods (3,600 states), with epsilon-greedy exploration decaying
   from 1.0 to 0.05. A pair's first value is set from what was observed, and untried actions are ignored when
   choosing, because with mostly negative rewards an untried action (value 0) would otherwise look best. Three
   versions are compared: *plan settings* (alpha 0.1, gamma 0.95), *tuned* (alpha 0.02, gamma 0.9) and *selected*
   (plan settings with a coarse 5 x 6 x 3 x 3 grid). "Selected" follows a rule written down before the test split
   was touched, the sweep variant with the highest mean validation savings; "tuned" came from single-seed
   validation scans.
5. **Deep RL: PPO and DQN** (stable-baselines3, two hidden layers of 64). They train on the same environment,
   reward, actions and sample budget (504,000 steps = 3,000 episodes) as the table, and are scored with the same
   evaluation. They see the full observation, including the highest grid import so far, the last 3 hours of
   demand and the day of week, whereas the table was limited to the plan's 4 discretized features. So the
   comparison is "tabular with the plan's state" against "deep RL with the full observation", not a controlled
   test of the algorithm alone. Hyperparameters are common defaults, fixed before the 2010 split was evaluated
   and not tuned. A peak ablation retrains both with the peak input hidden.
6. **Oracles (upper bounds).** A linear program is given the whole week of demand and prices in advance and picks
   continuous charge and discharge power to minimize the real bill under the environment's battery limits; any
   controller's plan is one feasible plan of this problem, so its cost is a lower bound on theirs. A second oracle
   must pick one of the 5 environment actions each hour (a mixed-integer program, violation-free plans only), which
   separates the cost of the limited action set from the cost of not knowing the future. Replaying its plans
   through the real environment reproduces their cost exactly with zero violations. The solver returns the best
   plan found and a proven bound, which differ by 1.3 points of savings on the test year (27.5% and 28.7%).

## 5. Experiments

Everything runs from YAML configs under `configs/experiments/`, with results saved as JSON under `results/`;
`scripts/make_report.py` rebuilds every table and figure. Training uses 2007-2008 only.

**Sweeps.** Eight sweeps, 5 seeds each (250 runs, validation split), change one factor at a time around the
plan's baseline (alpha 0.1, gamma 0.95, 3,000 episodes, 10 kWh battery, 10 x 24 x 5 x 3 grid): learning rate,
discount, epsilon schedule, training episodes, battery capacity, state grid, reward weights, and giving the
table the peak so far as extra state bins. A 5 x 4 grid of alpha and gamma gives a sensitivity heatmap.

**Final test.** Each Q-learning and deep RL version is trained with 5 seeds (PPO and DQN take about 5 minutes
per run, the table about 10 seconds), then scored on validation, on 100 sampled weeks of 2010, on the held-out
winter and summer weeks, and on every Jan-Feb and Jun-Jul 2010 week that starts at midnight. Every controller
sees the same weeks.

## 6. Results

### 6.1 Validation split (2009, 50 weeks)

The rule-based controller cuts the energy bill by 18.5% but raises the weekly peak by 28%, so the demand charge
cancels the saving (0.03% total). The greedy rule charges gently and saves 9.6%. Tabular Q-learning saves -2.2 ±
3.3% with the plan settings and 2.7 ± 1.7% with the coarse grid, while PPO saves 9.9 ± 0.7% and DQN 11.7 ± 2.5%
(mean ± std over 5 seeds).

### 6.2 Parameter sweeps

| Factor | Values and total savings (%), validation split, mean ± std over 5 seeds |
|---|---|
| Learning rate alpha | 0.01: 1.5 ± 1.8 / 0.05: -1.2 ± 1.5 / 0.1: -2.2 ± 3.3 / 0.3: -2.0 ± 1.8 |
| Discount gamma | 0.8: -1.0 ± 1.2 / 0.9: -1.2 ± 1.3 / 0.95: -2.2 ± 3.3 / 0.99: -3.3 ± 1.1 |
| Epsilon schedule | linear: -2.9 ± 2.0 / fast: -1.7 ± 0.3 / exponential: -2.2 ± 3.3 / slow: -3.9 ± 2.9 |
| Training episodes | 500: -5.6 ± 1.7 / 1,500: -2.1 ± 1.6 / 3,000: -2.2 ± 3.3 / 6,000: -0.1 ± 0.9 |
| Battery capacity | 5 kWh: -0.7 ± 1.0 / 10: -2.2 ± 3.3 / 13.5: -1.9 ± 1.2 / 20: -1.1 ± 1.9 |
| State grid | coarse: 2.7 ± 1.7 / medium: -2.2 ± 3.3 / fine: -3.3 ± 0.6 |
| Reward weights | default: -2.2 ± 3.3 / cost-focused: -9.2 ± 0.4 / peak-focused: -1.8 ± 0.8 / cycle-averse: -1.6 ± 1.7 |
| Peak input for the table | none: -2.2 ± 3.3 / 3 bins: -1.8 ± 1.5 / 5 bins: -3.0 ± 1.7 (violations rise to 0.29 and 0.54 a week) |

*Table 2. Sweep results.*

The 5-seed std is 1 to 3 points, so many differences between neighbouring values are within noise. The clear
effects are: lower alpha is better (the alpha x gamma grid, Figure 2, shows the same direction at every gamma);
more training helps; the coarse grid beats the fine grid; and a cost-focused reward is clearly worst (energy
cost down 7.3% but peak 34.5% higher). Figure 3 shows that the epsilon schedules differ in how fast training
converges but end at similar quality. Under the plan's literal reward the plan-settings agent scored -9.8 ± 0.8%
(3 seeds), with peak 38% higher and 1.6 violations a week, which is why the reward was changed.

### 6.3 Test split (2010, 100 sampled weeks)

| Controller | Total cost ($/week) | Total savings | Peak (kW) | Violations / week | Full cycles / week |
|---|---|---|---|---|---|
| No battery | 45.31 | 0% | 3.45 | 0 | 0 |
| Rule-based | 47.51 | -4.9% | 4.95 | 5.57 | 3.05 |
| Greedy threshold | 41.78 | **7.8%** | 3.78 | 2.38 | 2.95 |
| Q-learning, plan settings | 47.53 ± 0.98 | -4.9 ± 2.2% | 4.16 ± 0.15 | 0.02 | 4.82 ± 0.24 |
| Q-learning, tuned | 46.06 ± 0.76 | -1.7 ± 1.7% | 3.99 ± 0.16 | 0.02 | 5.11 ± 0.16 |
| Q-learning, selected | 44.61 ± 0.62 | 1.5 ± 1.4% | 3.47 ± 0.07 | 0.00 | 1.84 ± 0.52 |
| **PPO (deep RL)** | 42.08 ± 0.28 | **7.1 ± 0.6%** | 2.98 ± 0.02 | 0.03 | 1.17 ± 0.18 |
| **DQN (deep RL)** | 40.53 ± 0.99 | **10.5 ± 2.2%** | 2.94 ± 0.09 | 0.03 | 3.88 ± 1.08 |
| PPO, peak input hidden | 42.69 ± 0.99 | 5.8 ± 2.2% | 3.12 ± 0.07 | 0.01 | 1.19 ± 0.58 |
| DQN, peak input hidden | 43.75 ± 1.20 | 3.4 ± 2.7% | 3.52 ± 0.16 | 0.19 | 4.28 ± 2.00 |
| Oracle, 5 power levels | 32.86 | 27.5% | 2.32 | 0 | 5.49 |
| Oracle, continuous power | 29.17 | 35.6% | 1.55 | 0 | 5.24 |

*Table 3. Test split. Learned rows are mean ± std over 5 training seeds.*

| Controller | Share of continuous oracle's savings | Share of 5-level oracle's savings | Total savings, terminal-adjusted |
|---|---|---|---|
| Rule-based | -13.7% | -17.7% | -4.1% |
| Greedy threshold | 21.9% | 28.4% | 8.5% |
| Q-learning, selected | 4.3 ± 3.8% | 5.6 ± 5.0% | 1.3 ± 1.4% |
| PPO (deep RL) | 20.0 ± 1.8% | 25.9 ± 2.3% | 7.4 ± 0.6% |
| DQN (deep RL) | 29.6 ± 6.2% | 38.4 ± 8.0% | 10.8 ± 2.3% |
| PPO, peak input hidden | 16.2 ± 6.1% | 21.0 ± 7.9% | 5.8 ± 2.3% |
| DQN, peak input hidden | 9.7 ± 7.5% | 12.5 ± 9.7% | 3.6 ± 2.9% |
| Oracle, 5 power levels | 77.1% | 100% | 27.0% |
| Oracle, continuous power | 100% | 129.7% | 34.9% |

*Table 4. A share is savings divided by the oracle's savings (plain bill). Plan-settings and tuned Q-learning are
-13.8% and -4.7% of the continuous oracle. Adjusted savings value leftover battery energy. The 5-level oracle's
savings are the best plan found; its proven ceiling is 28.7%, so shares against it could be overstated by about
5% in relative terms.*

The best tabular controller saves about $0.70 a week, roughly one standard deviation from zero. The greedy rule
saves about $3.53 a week, PPO $3.23, DQN $4.78 and the continuous oracle $16.14 (Figure 4).

### 6.4 Deep RL: seasons and the peak ablation

| Controller | Winter week | Summer week | Jan-Feb weeks | Jun-Jul weeks |
|---|---|---|---|---|
| Rule-based | 7.4% | -20.5% | 1.6% | -17.8% |
| Greedy threshold | 11.0% | 2.0% | 10.9% | -1.8% |
| Q-learning, selected | 4.3 ± 3.8% | 0.2 ± 2.5% | 2.5 ± 1.4% | 1.0 ± 1.6% |
| PPO | 13.2 ± 0.5% | 9.0 ± 1.0% | 9.9 ± 0.6% | 7.0 ± 0.8% |
| DQN | 12.2 ± 4.0% | 12.6 ± 4.0% | 10.4 ± 2.7% | 12.1 ± 2.0% |
| DQN, peak input hidden | 10.4 ± 5.5% | -2.3 ± 4.3% | 7.1 ± 3.1% | 1.2 ± 4.0% |

*Table 5. Total savings on held-out seasons of 2010 (mean ± std over 5 seeds).* DQN's five test seeds score 12.5,
11.0, 7.1, 9.2 and 13.0%, and PPO's 6.5, 6.9, 6.6, 8.2 and 7.3%. Figure 5 shows the validation learning curves:
DQN reaches good reward within about 150,000 steps and is noisy afterwards, and PPO is smoother and plateaus a
little lower.

## 7. Analysis

**Why the tabular controller struggles.** The demand charge depends on the highest hour of the week, and the table
cannot see how close it is to the peak it has already set, so it charges at full rate in hours that raise the peak.
The evidence is the reward sweep (a cost-focused reward gives the lowest savings, -9.2%, and the highest peak), the
rule-based controller (full-rate charging raises the validation peak 28%), and the greedy rule, which avoids the
problem with a fixed gentle charge rate and wins. Noise and sparse data add to it. Weekly demand is noisy, so a
large alpha lets single episodes drag the table around (alpha 0.1 and 0.3 lose in every gamma column), and the
fine grid has about 90 training visits per visited state against about 3,400 for the coarse grid, with the most
violations (0.20 a week against 0.00). A higher gamma raises energy savings (1.4%, 2.8%, 5.2% for gamma 0.8, 0.9,
0.95) and cycling, but not total savings, because the extra charging raises peaks. The epsilon schedules change how
fast the agent learns (about 1,000 episodes for the fast decay, 2,000 for the exponential, 2,500 for the slow and
linear), not where it ends up.

**Why deep RL does better, and the ablation.** The deep agents beat the table by 5 to 9 points on test and bring
the weekly peak to 2.94 to 2.98 kW, 14% below the no-battery peak, which no other non-oracle controller manages.
Two differences could explain this: they see more (the peak so far, recent demand, day of week), and a network
generalizes across similar situations where a table cannot. With the peak input hidden, DQN falls from +10.5% to
+3.4% (weekly peak back to 3.52 kW, violations 0.03 to 0.19, summer week +12.6% to -2.3%), so seeing the peak is
a large part of its advantage. PPO falls only from +7.1% to +5.8%, within the seed noise, and still beats every
tabular agent, so something besides the peak input also helps. Giving the same peak information to the table as
extra state bins did not help (-1.8% and -3.0% against -2.2%), consistent with a table starved for data. This is an
interpretation: we did not isolate function approximation from the other inputs. The edge over the greedy rule is
real but modest: DQN is about 2.7 points ahead on average (roughly one seed standard deviation) and one of its
five seeds is below the rule, while PPO is about level with it.

**How far from the best possible.** The continuous oracle saves 35.6% on test, about 59% of it from peak shaving
(the weekly peak falls from 3.45 kW to 1.55 kW, worth about $9.50 a week) and the rest from shifting energy to
cheap hours. It keeps grid import almost flat, around 0.9 kW across the summer test day (Figure 6). Limiting power
to the 5 environment levels costs about 8 points (35.6% to 27.5%), roughly a quarter of the 34 points between the
best tabular agent and the continuous oracle. The 5-level oracle still saves 27.5% while DQN reaches 38% of that,
PPO 26%, the greedy rule 28% and the best table about 6%, so a large gap remains for any learner, and the learners
miss mainly the peak.

**Generalization and seasons.** Every controller loses ground from validation to test, including the two that
never learn (rule-based 0.0% to -4.9%, greedy 9.6% to 7.8%), so part of the drop is a property of the data: mean
weekly peak demand falls from 4.30 kW in training to 3.50 kW in 2010, which makes it easier for a 3 kW charge to
set a new peak. The learned versions drop by 2.7 (plan), 2.9 (tuned), 1.1 (selected), 2.8 (PPO) and 1.2 (DQN)
points. The summer week has mean demand of 0.75 kW and a no-battery peak of 2.59 kW, below the 3 kW charge power,
so any full-rate charge sets a new peak: the rule-based controller's summer-week peak is 4.71 kW and it loses
20.5%. The deep agents stay positive in summer (Table 5). Winter demand is higher, charging rarely sets a new peak
and every controller saves. Single weeks are noisy; the Jan-Feb and Jun-Jul rows are steadier but their windows
overlap and are not independent.

**What the learned policies look like.** The selected table agent is idle in 65% of the states it visited, never
charges at full rate and cycles 1.8 times a week against 4.8 for the plan settings (Figure 7). It generalizes best
of the tabular agents because it does little, so it rarely makes a costly mistake, and it leaves most of the
possible saving unused (energy savings of 2.9%, against 18.4% for the greedy rule).

**Limitations.**

- One household, one made-up tariff, one battery power limit and one test year; results could differ with other
  tariffs, power limits or years.
- The deep agents see more information than the table, so the comparison is not a controlled test of the algorithm
  alone. No learner was tuned, so a tuned comparison could differ. DQN varies from 7.1% to 13.0% across seeds, and
  deep training costs about 30 times more compute than the table.
- Configs were chosen on validation data, so validation numbers for chosen settings are optimistic; the test
  numbers are unbiased but come from one run of the protocol. The rule-based and greedy controllers were not tuned.
- Both oracles assume exact knowledge of the week, so no real controller can reach 100%. Half of the test weeks did
  not reach the 5-level solver's 0.2% target within 30 seconds, so shares against it carry that uncertainty.
- The terminal-adjusted check shifts savings by under one point and changes no ranking, but it depends on the
  choice of an off-peak replacement price.

## 8. Agentic extension

Not built in this submission. The planned design is a language-model layer that turns a plain-English request
("keep a 40% reserve and cut my evening peak") into validated reward weights and limits, with an offline
fallback, and an MCP server that exposes backtests and comparisons as tools. [YOU: keep this section as future
work, or delete it and renumber.]

## 9. Contributions

This is an individual project. All design, implementation, experiments and writing were done by Mugunthan
Kesavan. [YOU: edit this paragraph to say in your own words who did what and what you verified.]

## 10. Conclusions and future work

On real household demand with a simulated battery and tariff, tabular Q-learning cut the energy bill but, with the
plan's settings, did not beat doing nothing once the demand charge was included (-2.2 ± 3.3% on validation,
-4.9 ± 2.2% on test). A bill-aligned, less noisy reward and a coarser state grid gave a small gain (+1.5 ± 1.4% on
test), still far behind a simple greedy rule (+7.8%). Deep RL with the same environment, actions, reward and sample
budget did much better: DQN +10.5 ± 2.2% and PPO +7.1 ± 0.6% on the 2010 test weeks, both with a weekly peak below
the no-battery level and both positive in summer. DQN beats the greedy rule on average (four of five seeds), PPO
roughly matches it, and hiding the peak input costs DQN about 7 points, while the same information given to the
table did not help. Generalization to 2010 is weaker for every controller because demand peaks are lower, and
summer weeks are hardest for controllers that charge at full power. Against the oracle, DQN captures about 30% of
the achievable saving (38% of the 5-level oracle's), PPO 20% (26%), the greedy rule 22% (28%) and the best table
about 4% (6%).

Future work, in order of expected value: (1) tune the deep agents and the simple rules fairly, and run the deep
agents on the same 4-feature state as the table to isolate the algorithm from the inputs; (2) richer inputs and a
demand forecaster, to narrow the gap to the foresight oracle; (3) test other tariffs, power limits and years;
(4) add the language-model layer from section 8.

## Appendix A. Reproduce everything

```bash
uv sync --extra oracle --extra deeprl
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/run_baselines.py
uv run python scripts/run_experiment.py all --workers 8
uv run python scripts/run_final.py
uv run python scripts/run_oracle.py && uv run python scripts/run_oracle_discrete.py
uv run python scripts/run_deep_rl.py
uv run python scripts/run_deep_rl.py --mask peak_kw --tag nopeak --out results/final/deep_rl_nopeak.json
uv run python scripts/make_report.py && uv run pytest
```
