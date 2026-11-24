"""Rebuild all report tables and figures from the saved results in results/.

Usage: uv run python scripts/make_report.py
Builds the sweep tables/figures, then (if results/final exists) the final comparison tables,
generalization table, controller bar charts, representative days, policy heatmaps and the parameter record.
"""
from home_energy_rl_agent.experiments import EXP_DIR
from home_energy_rl_agent.final_eval import FINAL_DIR
from home_energy_rl_agent.final_report import parameter_record, report_final
from home_energy_rl_agent.reporting import TABLE_DIR, report_sweep


def main() -> None:
    for path in sorted(EXP_DIR.glob("sweep_*.yaml")):
        agg = report_sweep(path.stem)
        print(f"{path.stem}: " + ("no results yet" if agg is None else f"{len(agg)} variants"))
    if (FINAL_DIR / "records.json").exists():
        report_final()
        print("final comparison tables and figures written")
    else:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        parameter_record(TABLE_DIR / "parameter_record.md")
        print("results/final not found: run scripts/run_final.py for the final comparison")


if __name__ == "__main__":
    main()
