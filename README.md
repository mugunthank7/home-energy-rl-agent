# home-energy-rl-agent

Reinforcement learning + LLM agent energy manager for a home battery, using real household power data (UCI).

> **Real vs simulated:** demand is REAL (UCI Individual Household Electric Power Consumption, one French household, 2006-2010). The battery, tariff, and any solar are SIMULATED. Savings are simulation results, not field results.

Status: Tier 0 complete, plus the Tier 1 optimal-control oracle (data pipeline, simulated environment, baselines, Q-learning, 7 parameter sweeps, generalization test on 2010, perfect-foresight oracle). Deep RL and the agent layer are not built. Results will be added only from real runs saved in `results/`.

## Quickstart
```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/make_eda.py
uv sync --extra oracle        # optional: cvxpy for the oracle upper bound
uv run python scripts/run_oracle.py
uv run python scripts/run_oracle_discrete.py    # about 15 minutes
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
| Oracle, 5 power levels (best plan found) | 32.86 | +27.5% | 77.1% | 2.32 |
| Oracle, continuous power (perfect foresight) | 29.17 | +35.6% | 100% | 1.55 |

The best learned controller is only slightly better than doing nothing and clearly worse than the simple greedy rule.
Every controller loses ground from 2009 to 2010, and summer weeks are hardest for controllers that charge at full power
(the summer week's whole peak is below the 3 kW charge power). The oracle is a linear program that knows the whole week of demand and prices in advance and uses continuous power, so it is an upper bound, not a deployable controller: about 59% of its saving comes from flattening the weekly peak (3.45 kW down to 1.55 kW) and 41% from shifting energy to cheap hours. The greedy rule captures about a fifth of it and the best Q-learner about 4%. A second oracle restricted to the environment's 5 power levels (a mixed-integer program) saves 27.5% on test (proven ceiling 28.7%), so the 5-level action set explains about a quarter of the distance between the best learner and the continuous oracle, and perfect foresight with 5 levels would still save far more than any controller built here. The greedy rule captures 28.4% of the 5-level oracle's saving and the best Q-learner 5.6 ± 5.0%. The environment ignores the energy left in the battery at the end of a week, so a terminal-adjusted version is also reported in `docs/GENERALIZATION.md`; it changes no ranking (best Q-learner +1.3 ± 1.4%, greedy +8.5%, oracle +34.9%). Oracle scores are re-created with `uv sync --extra oracle` and `uv run python scripts/run_oracle.py`.

Figures: `reports/figures/controllers_test.png`, `representative_days.png`, `policy_heatmap_*.png`. The settings used are recorded in `reports/tables/parameter_record.md`.
