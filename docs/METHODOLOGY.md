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

(Further details are filled in as each phase lands.)
