"""Train the final Q-learning configs (5 seeds) and score every controller on validation and the
2010 test split, plus the held-out winter/summer weeks and seasons. Saves results/final/records.json.

Usage: uv run python scripts/run_final.py
This is the only script that touches the 2010 test split.
"""
from home_energy_rl_agent.final_eval import run_final


def main() -> None:
    run_final()
    print("Saved results/final/records.json")


if __name__ == "__main__":
    main()
