"""Run one or more sweeps defined in configs/experiments/sweep_*.yaml.

Usage: uv run python scripts/run_experiment.py all --workers 8
       uv run python scripts/run_experiment.py sweep_alpha sweep_gamma --seeds 42,43
Finished runs are skipped (resume), unless --force is given.
"""
import argparse

from home_energy_rl_agent.experiments import EXP_DIR, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", nargs="+", help="sweep names, or 'all'")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", help="comma-separated, overrides the config")
    args = parser.parse_args()

    names = sorted(p.stem for p in EXP_DIR.glob("sweep_*.yaml")) if args.sweeps == ["all"] else args.sweeps
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    for name in names:
        run_sweep(name, args.workers, args.force, seeds)


if __name__ == "__main__":
    main()
