# Methodology

## Real vs simulated
- REAL: household demand (UCI dataset id 235).
- SIMULATED: battery model, time-of-use tariff, demand charge, any PV.

## Environment (Phase 2)
- Step = 1 hour; episode = 7 days drawn from contiguous windows of the chosen split.
- Battery power is on the AC side; round-trip efficiency is split as sqrt per direction.
- Violation = requested charge/discharge beyond SOC bounds or the power limit (clipped and penalized).
- Discharge is capped at current demand (no export). The cut amount is logged as `curtailed_kw` and is not penalized.
- Reward = -(price x grid import) - w_peak x (increase in episode peak) - w_viol x violation - w_cycle x throughput.
- The demand charge (price per kW of the episode's peak import) is added to the reported total cost, while the reward only uses the peak-increase term as shaping.

## Reward design (Phase 4 finding)
The plan's literal reward penalizes each rise in the episode's peak import (`peak_term: increase`, `w_peak: 0.5`, `w_viol: 1.0`). With tabular Q-learning this learned badly (see `results/qlearning/plan_reward/`):
- The episode peak is not in the Q-table state, so the penalty looks like random noise to the agent.
- It lumps together peaks the battery caused and peaks the home's own demand would have set anyway.
- A 1-point violation penalty is cheaper than the peak shaving a clipped discharge earns, so the agent learned to break battery limits on purpose.

The default is now `peak_term: excess`, `w_peak: 5.0` (equal to the demand charge in $/kW), `w_viol: 10.0`. "Excess" charges each rise in (peak import - peak demand). Peak demand does not depend on actions, so summed over an episode this differs from the plain peak penalty by an action-independent constant, so the best policy is unchanged. Idling then costs exactly zero in the peak term.

The reward is not the bill: it also contains violation and cycling penalties, so a controller with a lower bill can have a lower reward (the greedy baseline does). Compare controllers on the bill metrics.

## Tuning and selection bias
The `tuned` hyperparameters (alpha 0.02, gamma 0.9) were chosen from single-seed scans on the VALIDATION split, so validation numbers for `tuned` are slightly optimistic. The 2010 test split has not been used and gives the clean comparison.

## Test protocol (Phase 6)
- Q-learning configs `baseline`, `tuned` and `selected` were fixed using the validation sweeps (2009) before the 2010 test split was evaluated. `selected` follows a rule written beforehand: the sweep variant with the highest mean validation total savings.
- Each config is trained on 2007-2008 with 5 seeds and scored on validation, on 100 sampled test weeks of 2010, on one held-out winter week (Jan 18-24) and one summer week (Jul 5-11), and on all Jan-Feb and Jun-Jul 2010 weeks that start at midnight. July is used for summer because the household was away for much of August.
- All controllers see identical episodes and are compared with a no-battery run on those same weeks.
- The test split was not used for tuning. Results on it are reported once, whatever they show.

## Oracle and terminal value (Phase 7)
- The oracle is a linear program per 7-day episode (cvxpy): continuous AC-side charge and discharge power, stored energy that follows the same round-trip efficiency and bounds as the simulator, discharge capped at demand (no export), and a peak variable that the demand charge multiplies. The objective is the plain bill plus a negligible throughput term that breaks ties.
- Because any 5-level controller is one feasible plan of this problem, the oracle's cost is a lower bound on theirs. This is checked by tests (never above any baseline on the same weeks) and by replaying plans through the simulator's `Battery` class.
- The oracle sees the same weeks as every other controller (same start selection) and is evaluated with the same summary function.
- The environment's reward and bill ignore energy left in the battery at the end of an episode. As a robustness check, a terminal-adjusted cost adds `(start energy - end energy) x (off-peak price / charging efficiency)`. A second LP (`oracle_adj`) optimizes that adjusted objective. The plain metric stays the headline; both are reported.

## Oracle restricted to 5 power levels
- A mixed-integer program per 7-day episode (HiGHS through scipy): one binary per hour and action, stored energy that follows the simulator's efficiency and bounds, discharge cut to the current demand, and a peak variable priced at the demand charge.
- It only allows plans a controller can follow with no limit violation: a discharge request must be coverable by the stored energy and a charge request must fit in the free capacity, exactly as the battery model flags violations. It is therefore a bound for violation-free 5-level controllers. A controller that violates limits gets partial charges or discharges outside this set, so it could in principle beat the bound; none of the controllers evaluated here came close.
- The solver returns the best plan found (its cost is achievable, replayed through the real environment with identical cost and zero violations) and a dual bound (a proven lower bound on cost). Each week has a 30-second limit and a 0.2% relative gap target; weeks that hit the limit still return their best plan and bound. Both are reported.
- Variants: plain bill (`oracle_discrete`) and with leftover battery energy valued (`oracle_discrete_adj`).

## Deep RL protocol
- PPO and DQN (stable-baselines3, MlpPolicy with two hidden layers of 64) are trained on the train split only, with 5 seeds each, on the same environment, reward and 5 discrete actions as the tabular agent. The sample budget is 504,000 steps, equal to 3,000 episodes of 168 steps.
- The network sees the env's observation (hour sin/cos, day of week, charge level, demand, tariff period, peak so far, last 3 hours of demand) divided by fixed scales (a high quantile of TRAIN demand for kW values); training rewards are multiplied by 0.1. Scoring uses the unscaled environment and the same evaluation weeks as every other controller.
- Hyperparameters (`configs/experiments/ppo.yaml`, `dqn.yaml`) are common defaults fixed before the 2010 split was evaluated. They were not tuned on validation or test. A short smoke run on validation checked that the code works; no setting was changed afterwards.
- The peak ablation (`--mask peak_kw --tag nopeak`) divides the peak input by infinity so the network always sees 0 for it; everything else is identical.
- The tabular agent's state is the plan's 4 discretized features; `sweep_peak_bins` adds optional peak bins to it.

(Further details are filled in as each phase lands.)
