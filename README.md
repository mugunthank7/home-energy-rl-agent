# home-energy-rl-agent

Reinforcement learning + LLM agent energy manager for a home battery, using real household power data (UCI).

> **Real vs simulated:** demand is REAL (UCI Individual Household Electric Power Consumption, one French household, 2006-2010). The battery, tariff, and any solar are SIMULATED. Savings are simulation results, not field results.

Status: Phase 4 done (data pipeline, EDA, simulated environment, baseline controllers, Q-learning). Results will be added only from real runs saved in `results/`.

## Quickstart
```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/make_eda.py
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
| Q-learning, tuned (alpha 0.02, gamma 0.9) | 8.0 ± 0.3% | 2.6 ± 0.7% | 5.8% higher | 0.03 |
| Q-learning, plan's literal reward (ablation) | 8.7 ± 0.8% | -9.8 ± 0.8% | 38.0% higher | 1.6 |

What this shows:
- Tabular Q-learning beats the no-battery baseline and the rule-based controller on total cost, with almost no constraint violations, but the simple greedy rule is clearly better (9.6%).
- The plan's literal reward trained a worse agent (see `docs/METHODOLOGY.md`, "Reward design").
- The tuned settings were picked from validation-split scans, so the 2.6% is slightly optimistic.
