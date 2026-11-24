"""Rebuild all report tables and figures from the saved results in results/.

Usage: uv run python scripts/make_report.py
(Phase 5: sweep tables and figures. Later phases add the remaining report assets.)
"""
from home_energy_rl_agent.experiments import EXP_DIR
from home_energy_rl_agent.reporting import report_sweep


def main() -> None:
    for path in sorted(EXP_DIR.glob("sweep_*.yaml")):
        agg = report_sweep(path.stem)
        print(f"{path.stem}: " + ("no results yet" if agg is None else f"{len(agg)} variants"))


if __name__ == "__main__":
    main()
