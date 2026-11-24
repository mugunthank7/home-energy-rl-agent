# Module explainers

Plain-English notes so each module can be explained on video. One entry per module.

## data.py (Phase 1)
- **What it does:** turns the raw UCI file (about 2 million minute-level rows) into a clean hourly demand series in kW. This is the REAL data.
- **Parsing:** the file uses `;` separators, dd/mm/yyyy dates, and `?` for missing values. We combine Date and Time into one timestamp and turn `?` into NaN.
- **Gap handling:** missing stretches of 3 hours or less are forward-filled, because a short sensor dropout probably looks like the minutes before it. Longer gaps are not guessed.
- **Hourly resample:** the mean kW over an hour equals the kWh used in that hour. An hour with any missing minute stays missing, so partial hours never look complete.
- **Dropping days:** any calendar day that does not have all 24 hours is dropped, and the counts are logged in `data/processed/cleaning_log.json`.
- **Splits:** by calendar year and never shuffled: train 2007-08, validation 2009, test 2010. Shuffling would leak the future into training.
- **Held-out weeks:** a winter week (Jan 18-24, 2010) and a summer week (Jul 5-11, 2010) are kept for the generalization test. Jan 12-14 have long gaps, and August is avoided because the household was away.
- **Episode starts:** because dropped days leave holes, `valid_episode_starts` only allows 7-day windows that are fully contiguous.
- **Why it matters:** the RL agent trains on the train split only. Everything else (battery, tariff) is simulated on top of this real demand.

## tariff.py (Phase 2)
- **What it does:** gives each hour a price, using a SIMULATED time-of-use tariff (not a real utility).
- **Three periods:** off-peak 22:00-06:00 (cheapest), on-peak 16:00-21:00 on weekdays (most expensive), and mid-peak for everything else.
- **Codes:** periods are numbers (0 off, 1 mid, 2 on) so controllers and the Q-table can use them directly.
- **Demand charge:** a price per kW on the single highest hourly grid import in the episode. It pushes the agent to flatten peaks, not just shift energy.
- **Config:** all prices and hours live in `configs/tariff.yaml`, so changing the tariff needs no code change.

## battery.py (Phase 2)
- **What it does:** the SIMULATED battery: 10 kWh, 3 kW max, 90% round-trip efficiency, SOC kept between 10% and 100%.
- **Efficiency:** split evenly between charging and discharging (square root each way), so a charge-then-discharge cycle loses about 10% of the energy.
- **Power is measured at the wall (AC side):** charging 2 kW stores 2 x sqrt(0.9) kWh; delivering 2 kW removes 2 / sqrt(0.9) kWh from storage.
- **Violations:** asking to charge a full battery or discharge an empty one is clipped to what is possible and flagged as a violation, which the reward penalizes.
- **No export:** the battery can never discharge more than the home is using. That excess is just recorded as "curtailed" and is not penalized.
- **Why it matters:** these rules make the problem realistic. The agent cannot cheat physics, and efficiency losses make useless cycling cost money.

## env.py (Phase 2)
- **What it does:** a Gymnasium environment. Each step is one hour and each episode is 7 days (168 steps), starting at a random valid point in the chosen split.
- **Real vs simulated:** demand is REAL data; battery and tariff are SIMULATED.
- **Observation:** hour (as sin/cos), day of week, battery SOC, current demand, tariff period, and the last 3 hours of demand. An optional 3-hour "perfect" forecast exists for experiments only.
- **Actions:** 5 choices: discharge high, discharge low, idle, charge low, charge high (fractions of the 3 kW limit).
- **Grid import:** demand + charging - discharging. This is what the home buys.
- **Reward:** minus the energy cost, minus a penalty when the episode peak rises, minus a penalty for violations, minus a small penalty for battery throughput (wear).
- **Logging:** every step returns an info dict (demand, action, SOC, grid import, price, cost, peak, violation). All the plots are drawn from this.
- **Episode summary:** at the end it reports energy cost, demand charge, total cost, peak, violations, and equivalent full cycles.
- **Tests:** energy conservation, SOC always in bounds, an idle battery equals demand times price, and identical results for the same seed.

## controllers/ and evaluate.py (Phase 3)
- **What they do:** the non-learning baselines the Q-learning agent must beat, plus one function that scores any controller fairly.
- **Shared interface:** every controller has `act(obs) -> action` and `reset()`. The env, `evaluate()` and plots never need to know which controller they are using.
- **No-battery:** always idle. Every "savings %" in the project is measured against this.
- **Rule-based:** charge at full rate in off-peak hours while SOC is under 90%; discharge at full rate in on-peak hours while SOC is above 15%; otherwise idle.
- **Greedy threshold:** compares the current price to percentiles of TRAIN prices. It charges gently (low rate) when cheap, and discharges at a rate that follows current demand when expensive.
- **evaluate():** picks the episode start points once from a seed, then runs the controller and a no-battery reference on those exact same episodes. That is what makes comparisons fair.
- **Metrics:** reward, energy cost, total cost (energy plus demand charge), peak kW, violations, equivalent full cycles, and savings vs no-battery. Savings use the ratio of mean costs so one cheap week cannot dominate.
- **Key lesson from the first run:** charging at full rate can create a new demand peak, so a controller can lower the energy bill and still lose on the demand charge. This is the trade-off the RL agent has to learn.

## controllers/qlearning.py and scripts/train_qlearning.py (Phase 4)
- **What it does:** tabular Q-learning. It keeps a table with one row per discretized situation and one column per action, and learns how good each action is by trial and error.
- **State (3,600 situations):** battery level (10 bins) x hour of day (24) x demand level (5 quantile bins from TRAIN demand) x tariff period (3).
- **Learning rule:** after each hour, nudge the table entry toward "reward just received + gamma x best value of the next situation". Alpha is the step size and gamma is how much future savings count.
- **Exploration:** epsilon-greedy. With probability epsilon take a random action, otherwise the best known one. Epsilon decays from 1.0 to 0.05, so the agent explores early and exploits late.
- **Two fixes that mattered:** (1) the first time an action is tried, its value is set to what was observed, not blended with a made-up zero; (2) untried actions are ignored when choosing. Otherwise a zero (which looks great when all rewards are negative) beats every action that was actually tried.
- **Reward fix:** the plan's peak penalty was too noisy to learn from. We charge only the peak the battery adds on top of the home's own demand peak. This gives the same best policy, since the home's own peak does not depend on actions, with much less noise. The plan's version is kept as the `plan_reward` ablation.
- **Violation penalty:** at 1 point, the agent happily broke battery limits whenever a half-delivered discharge shaved a peak. At 10 points it stops.
- **Evaluation:** greedy policy (no exploration) on validation weeks the agent never trained on, 3 seeds, against the baselines on the same weeks.
- **Honest outcome:** Q-learning beats doing nothing but loses clearly to the simple greedy rule (see README). The sweeps in Phase 5 explore why.

## experiments.py, reporting.py, run_experiment.py, make_report.py (Phase 5)
- **What they do:** run many Q-learning experiments from config files, save each run, then turn the saved runs into tables and plots.
- **Sweep configs:** each `configs/experiments/sweep_*.yaml` names a base experiment (the plan's baseline) and what to change: one factor at a time, a two-factor grid, or a list of named variants. Changes use dotted paths such as `qlearning.alpha` or `env_overrides.battery.capacity_kwh`.
- **Why one factor at a time:** if only alpha changes, any difference in results can be blamed on alpha.
- **Runs:** each (variant, seed) trains on the train split and is scored on the validation split. Every run is saved as `results/<sweep>/<run_id>.json` with its config, metrics, and learning curve.
- **Seeds:** 5 seeds per variant. Reporting mean ± std shows whether a difference is real or just seed luck.
- **Parallel and resumable:** runs go in parallel across CPU cores, and finished runs are skipped on a re-run.
- **One training function:** the sweeps and `train_qlearning.py` share it. We checked that the refactor reproduces the earlier numbers exactly.
- **Aggregation:** `make_report.py` groups runs by variant, computes mean and std over seeds, and writes tables to `reports/tables/` and figures to `reports/figures/`.
- **Figures per sweep:** convergence curves with seed bands, and final savings, peak and violations with error bars. The alpha x gamma grid gets a heatmap.
- **Reward weights:** that sweep changes the reward itself, so compare its variants on the bill metrics, not on reward.

## final_eval.py and final_report.py (Phase 6)
- **What they do:** the final, fair comparison of every controller, then all the tables and figures for the report.
- **Test discipline:** the 2010 test data is only used here, after the Q-learning settings were chosen on the 2009 validation data. Choosing on the test data would make the results look better than they are.
- **The three Q-learning versions:** `baseline` (the plan's settings), `tuned` (lower alpha and gamma) and `selected` (coarse state grid). The rule for `selected` was fixed beforehand: best mean validation savings across all sweep variants.
- **Training:** each version is trained on 2007-2008 with 5 seeds, one at a time, so the recorded training times are honest.
- **Splits scored:** validation, 2010 test, a winter week and a summer week, plus all Jan-Feb weeks and all Jun-Jul weeks of 2010 (one window per day).
- **Fairness:** all controllers get the same episodes, and every score is measured against a no-battery run on those same weeks.
- **Error bars:** Q-learning bars show the std over seeds. The rule-based and greedy controllers are deterministic, so their bars show the standard error over weeks.
- **Representative days:** the Wednesday of the held-out winter and summer weeks, showing demand, grid import, battery level and the Q-learning actions.
- **Policy heatmap:** for each battery level and hour, the action the Q-table prefers. Light grey is idle, red is charge, blue is discharge.
- **Parameter record:** a table of every setting used, regenerated from the config files and result counts.
- **Missing on purpose:** the "% of oracle savings" column says n/a until the Tier 1 oracle exists.

## controllers/oracle.py and scripts/run_oracle.py (Phase 7, Tier 1)
- **What it is:** a best-case reference. It is told the whole week of demand and prices in advance and finds the cheapest possible battery plan. No real controller can do this, because nobody knows next week's demand.
- **How:** a linear program (cvxpy). The unknowns are charge power, discharge power and the weekly peak for each hour. The rules are the same as the simulator: battery size and power limit, efficiency, charge between 10% and 100%, no selling back to the grid.
- **Objective:** minimize the real bill: energy cost plus the demand charge on the weekly peak.
- **Why it is an upper bound:** every plan any controller could follow is allowed by these rules, so the oracle's cost can never be higher than theirs.
- **How we checked it:** tests confirm the plan obeys every rule, matches the simulator's battery when replayed, and is never worse than any baseline on the same weeks.
- **% of oracle savings:** a controller's savings divided by the oracle's savings on the same weeks. 100% would be perfect, 0% is no better than doing nothing, and negative means worse than doing nothing.
- **What it taught us:** about 59% of the oracle's saving is flattening the weekly peak and 41% is moving energy to cheap hours. The learners and the greedy rule mostly miss the peak part.
- **Leftover energy:** the oracle also comes in an adjusted version that pays for battery energy it uses up, so the comparison stays fair when we credit or charge for energy left in the battery.
- **Limit:** it uses smooth power levels, while the learned controllers pick from 5 levels, so part of the gap to the oracle is the action set, not just the missing foresight.
