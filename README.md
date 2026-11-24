# home-energy-rl-agent

Reinforcement learning + LLM agent energy manager for a home battery, using real household power data (UCI).

> **Real vs simulated:** demand is REAL (UCI Individual Household Electric Power Consumption, one French household, 2006-2010). The battery, tariff, and any solar are SIMULATED. Savings are simulation results, not field results.

Status: Phase 3 done (data pipeline, EDA, simulated environment, baseline controllers). Results will be added only from real runs saved in `results/`.

## Quickstart
```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_hourly.py
uv run python scripts/make_eda.py
uv run pytest
```

## Baseline results (validation split = 2009, 50 episodes, seed 42)
From `results/baselines/validation.json` (re-create with `uv run python scripts/run_baselines.py`).
Demand is REAL; battery and tariff are SIMULATED, so these are simulation results. Preliminary: Q-learning is not built yet.

| Controller | Energy savings | Total savings (incl. demand charge) | Peak change | Violations / episode |
|---|---|---|---|---|
| No battery | 0% | 0% | 0% | 0 |
| Rule-based | 18.5% | 0.04% | +28.3% higher | 4.96 |
| Greedy threshold | 18.2% | 9.6% | +3.7% higher | 2.34 |

Rule-based saves energy cost but its full-rate charging creates higher peaks, which the demand charge penalizes.
