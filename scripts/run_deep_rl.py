"""Train PPO and DQN (5 seeds each) and score them on validation, the 2010 test split, and the seasons.

Writes results/final/deep_rl.json (merged into the tables by make_report.py) and model files under
results/deep_rl/. Needs: uv sync --extra deeprl
Usage: uv run python scripts/run_deep_rl.py [--algos ppo dqn] [--workers 10] [--steps N] [--seeds 42,43]
Ablation: --mask peak_kw --tag nopeak --out results/final/deep_rl_nopeak.json (hides the peak-so-far input)
"""
import argparse

from home_energy_rl_agent.deep_rl import run_deep_rl
from home_energy_rl_agent.experiments import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="+", default=["ppo", "dqn"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--steps", type=int, help="override total_timesteps (for quick checks)")
    parser.add_argument("--seeds", help="comma-separated; default is the config's seeds")
    parser.add_argument("--out", help="write records here instead of results/final/deep_rl.json")
    parser.add_argument("--mask", nargs="*", default=[], help="observation names hidden from the network (ablation)")
    parser.add_argument("--tag", help="suffix for the controller name, e.g. nopeak")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else load_experiment(args.algos[0])["seeds"]
    overrides = {"total_timesteps": args.steps} if args.steps else {}
    if args.mask:
        overrides["mask_obs"] = args.mask
    if args.tag:
        overrides["tag"] = args.tag
    kwargs = {"out_path": __import__("pathlib").Path(args.out)} if args.out else {}
    run_deep_rl(args.algos, seeds, args.workers, overrides, **kwargs)
    print("done")


if __name__ == "__main__":
    main()
