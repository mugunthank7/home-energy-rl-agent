"""Evaluate the non-learning baselines on a REAL demand split and save results to results/.

Usage: uv run python scripts/run_baselines.py [--split validation] [--episodes 50] [--seed 42]
The test split is meant for the final comparison only, so the default is validation.
"""
import argparse

import pandas as pd

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.env import EnergyEnv
from home_energy_rl_agent.evaluate import evaluate, metrics_table, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Build the env on the chosen split; greedy thresholds use the TRAIN split only ---
    splits = data.get_splits(data.load_hourly())
    env = EnergyEnv(splits[args.split], seed=args.seed)
    controllers = build_baselines(env, splits["train"])

    # --- Evaluate every controller on identical episodes ---
    results = [evaluate(c, env, episodes=args.episodes, seed=args.seed) for c in controllers]
    table = metrics_table(results)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))

    out = data.ROOT / "results" / "baselines" / f"{args.split}.json"
    save_results(results, out, extra={"split": args.split, "episodes": args.episodes,
                                      "seed": args.seed})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
