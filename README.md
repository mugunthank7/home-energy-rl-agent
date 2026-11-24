# home-energy-rl-agent

Reinforcement learning + LLM agent energy manager for a home battery, using real household power data (UCI).

> **Real vs simulated:** demand is REAL (UCI Individual Household Electric Power Consumption, one French household, 2006-2010). The battery, tariff, and any solar are SIMULATED. Savings are simulation results, not field results.

Status: Tier 0 complete, plus Tier 1 (data pipeline, simulated environment, baselines, tabular Q-learning, 8 parameter sweeps, generalization test on 2010, perfect-foresight oracles, PPO and DQN deep RL with a peak-input ablation). The language-model agent layer is not built. Results will be added only from real runs saved in `results/`.

## Quickstart
```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/make_eda.py
uv sync --extra oracle        # optional: cvxpy for the oracle upper bound
uv run python scripts/run_oracle.py
uv run python scripts/run_oracle_discrete.py    # about 15 minutes
uv sync --extra oracle --extra deeprl   # optional: torch + stable-baselines3
uv run python scripts/run_deep_rl.py    # about 6 minutes on 10 cores
uv run pytest
```

## Results so far (validation split = 2009, 50 episodes)
From `results/baselines/validation.json` and `results/qlearning/*/summary.json`
(re-create with `scripts/run_baselines.py` and `scripts/train_qlearning.py <baseline|tuned|plan_reward>`).
Demand is REAL; battery and tariff are SIMULATED, so these are simulation results, not field results.
Q-learning rows are mean ± std over 3 seeds. These are validation numbers; the 2010 test split is not used yet.

| Controller | Energy savings | Total savings (incl. demand charge) | Peak change | Violations / episode |
|---|---|---|---|---|
| No battery | 0% | 0% | 0% | 0 |
| Rule-based | 18.5% | 0.04% | 28.3% higher | 4.96 |
| Greedy threshold | 18.2% | 9.6% | 3.7% higher | 2.34 |
| Q-learning, plan's hyperparameters (alpha 0.1, gamma 0.95) | 5.5 ± 1.5% | -1.0 ± 1.4% | 10.9% higher | 0.01 |
| Q-learning, tuned (alpha 0.02, gamma 0.9), 3 seeds | 8.0 ± 0.3% | 2.6 ± 0.7% | 5.8% higher | 0.03 |
| Same setting, 5 seeds (from the alpha x gamma grid) | n/a | 1.2 ± 1.9% | n/a | n/a |
| Q-learning, plan's literal reward (ablation) | 8.7 ± 0.8% | -9.8 ± 0.8% | 38.0% higher | 1.6 |

What this shows:
- Tabular Q-learning beats the no-battery baseline and the rule-based controller on total cost, with almost no constraint violations, but the simple greedy rule is clearly better (9.6%).
- The plan's literal reward trained a worse agent (see `docs/METHODOLOGY.md`, "Reward design").
- The tuned settings were picked from validation-split scans, and the 3-seed figure is optimistic: the same setting over 5 seeds is 1.2 ± 1.9%, which is not distinguishable from zero.

## Parameter sweeps (validation split, 5 seeds each)
Seven one-factor sweeps and an alpha x gamma grid, 235 runs in total. Tables are in `reports/tables/`,
figures (convergence curves with seed bands, final savings with error bars, alpha x gamma heatmap) are in
`reports/figures/`, and the analysis is in [`docs/SWEEP_FINDINGS.md`](docs/SWEEP_FINDINGS.md).
Re-create with `uv run python scripts/run_experiment.py all --workers 8` and `uv run python scripts/make_report.py`.

| Factor | Total savings by value |
|---|---|
| Learning rate alpha (0.01 / 0.05 / 0.1 / 0.3) | +1.5 / -1.2 / -2.2 / -2.0 % |
| Discount gamma (0.8 / 0.9 / 0.95 / 0.99) | -1.0 / -1.2 / -2.2 / -3.3 % |
| Training episodes (500 / 1500 / 3000 / 6000) | -5.6 / -2.1 / -2.2 / -0.1 % |
| State grid (coarse / medium / fine) | +2.7 / -2.2 / -3.3 % |
| Battery (5 / 10 / 13.5 / 20 kWh) | -0.7 / -2.2 / -1.9 / -1.1 % |

The plan's baseline settings do not beat doing nothing (-2.2 ± 3.3%); the best sweep settings reach about
+1 to +3%, well below the greedy rule's +9.6%. The 5-seed std is 1 to 3 points, so many gaps between
neighbouring values are within noise (see the findings doc for which effects are clear).

## Generalization test: train 2007-2008, test 2010 (100 sampled weeks)
From `results/final/records.json`; full tables in [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md). Q-learning rows are mean ± std over 5 seeds.
The 2010 split was used only after the Q-learning configs were fixed on validation.

| Controller | Total cost ($/week) | Total savings | % of oracle savings | Peak (kW) |
|---|---|---|---|---|
| No battery | 45.31 | 0% | 0% | 3.45 |
| Rule-based | 47.51 | -4.9% | -13.7% | 4.95 |
| Greedy threshold | 41.78 | +7.8% | 21.9% | 3.78 |
| Q-learning, plan settings | 47.53 ± 0.98 | -4.9 ± 2.2% | -13.8 ± 6.1% | 4.16 ± 0.15 |
| Q-learning, tuned | 46.06 ± 0.76 | -1.7 ± 1.7% | -4.7 ± 4.7% | 3.99 ± 0.16 |
| Q-learning, selected (coarse grid) | 44.61 ± 0.62 | +1.5 ± 1.4% | 4.3 ± 3.8% | 3.47 ± 0.07 |
| **PPO (deep RL)** | 42.08 ± 0.28 | **+7.1 ± 0.6%** | 20.0 ± 1.8% | 2.98 ± 0.02 |
| **DQN (deep RL)** | 40.53 ± 0.99 | **+10.5 ± 2.2%** | 29.6 ± 6.2% | 2.94 ± 0.09 |
| PPO, peak input hidden (ablation) | 42.69 ± 0.99 | +5.8 ± 2.2% | 16.2 ± 6.1% | 3.12 ± 0.07 |
| DQN, peak input hidden (ablation) | 43.75 ± 1.20 | +3.4 ± 2.7% | 9.7 ± 7.5% | 3.52 ± 0.16 |
| Oracle, 5 power levels (best plan found) | 32.86 | +27.5% | 77.1% | 2.32 |
| Oracle, continuous power (perfect foresight) | 29.17 | +35.6% | 100% | 1.55 |

**Main finding.** Tabular Q-learning (a 4-feature state table) barely beats doing nothing and loses clearly to the
simple greedy rule. Deep RL with the same environment, reward, actions and sample budget does much better:
**DQN beats the greedy rule on average** (+10.5 ± 2.2% against +7.8%; four of five seeds are above it) and PPO
roughly matches it (+7.1 ± 0.6%). Both keep the weekly peak below the no-battery level (2.9 to 3.0 kW against
3.45); no other non-oracle controller does (greedy 3.78, best tabular 3.47, rule-based 4.95). They also stay positive in the
summer week (+9.0% and +12.6%) where the rule-based controller and the finer tabular agents lose money.

**Why (ablation).** Hiding the peak-so-far input costs DQN about 7 points (+10.5% to +3.4%, weekly peak back to
3.52 kW) and PPO about 1.3 points (within noise), so seeing the peak is a large part of the advantage, mostly for
DQN. Giving the same peak information to the tabular agent as extra state bins did not help (`sweep_peak_bins`:
-1.8 ± 1.5% and -3.0 ± 1.7% against -2.2 ± 3.3%, with more violations), consistent with the table being starved
for data while a network generalizes across similar states. PPO without the peak input still beats every
tabular agent (+5.8%), so the richer inputs and function approximation matter too; this project did not
separate those two effects further.

**Fairness notes.** The deep agents see the full observation (peak so far, last 3 hours of demand, day of week), while
the tabular agent was limited to the plan's 4 discretized features. Deep RL hyperparameters are common defaults that
were fixed before the 2010 split was evaluated and not tuned; the sample budget equals Q-learning's (504,000 steps).
Training is far costlier: about 320 s per run (10 running in parallel) against about 10 s for tabular.

Every controller loses ground from 2009 to 2010, and summer weeks are hardest for controllers that charge at full power
(the summer week's whole peak is below the 3 kW charge power). The oracle is a linear program that knows the whole week of demand and prices in advance and uses continuous power, so it is an upper bound, not a deployable controller: about 59% of its saving comes from flattening the weekly peak (3.45 kW down to 1.55 kW) and 41% from shifting energy to cheap hours. A second oracle restricted to the environment's 5 power levels (a mixed-integer program) saves 27.5% on test (proven ceiling 28.7%), so the 5-level action set explains about a quarter of the distance between the best tabular learner and the continuous oracle. DQN captures 38.4 ± 8.0% of the 5-level oracle's saving, PPO 25.9 ± 2.3%, the greedy rule 28.4% and the best tabular agent 5.6 ± 5.0%. The environment ignores the energy left in the battery at the end of a week, so a terminal-adjusted version is also reported in `docs/GENERALIZATION.md`; it changes no ranking (DQN +10.8%, PPO +7.4%, greedy +8.5%, best tabular +1.3 ± 1.4%, continuous oracle +34.9%). Oracle scores are re-created with `uv sync --extra oracle` and `uv run python scripts/run_oracle.py`; deep RL with `uv sync --extra oracle --extra deeprl` and `uv run python scripts/run_deep_rl.py`.

Figures: `reports/figures/controllers_test.png`, `representative_days.png`, `policy_heatmap_*.png`. The settings used are recorded in `reports/tables/parameter_record.md`.
