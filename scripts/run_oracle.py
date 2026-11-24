"""Score the perfect-foresight oracle on the same weeks as every other controller.

Writes results/final/oracle.json, which make_report.py merges into the tables ("% of oracle savings").
Two variants: "oracle" minimizes the plain bill; "oracle_adj" also charges for stored energy used up
(the bound for the terminal-adjusted metric).
Needs cvxpy: uv sync --extra oracle. Takes a few seconds. Run after scripts/run_final.py is not
required: the oracle never trains, and it uses the same episode starts as evaluate().
"""
import json

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.oracle import evaluate_oracle
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.final_eval import FINAL_DIR, _record, eval_specs, spec_starts
from home_energy_rl_agent.tariff import load_tariff_config


def main() -> None:
    splits = data.get_splits(data.load_hourly())
    env_cfg, tariff = load_env_config(), load_tariff_config()
    records = []
    for spec in eval_specs(splits):
        env = EnergyEnv(spec["series"], env_cfg, tariff, seed=42)
        starts = spec_starts(env, spec)
        for value_terminal in (False, True):
            result = evaluate_oracle(env, starts, seed=42, value_terminal=value_terminal)
            records.append(_record(spec["name"], result.name, None, result))
            s = result.summary
            key = "savings_total_adj_pct" if value_terminal else "savings_total_pct"
            print(f"{spec['name']} [{result.name}]: total savings {s[key]:.1f}%, "
                  f"peak reduction {s['peak_reduction_pct']:.1f}% ({s['episodes']} weeks)", flush=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "oracle.json").write_text(json.dumps({"records": records}))
    print(f"Saved {FINAL_DIR / 'oracle.json'}")


if __name__ == "__main__":
    main()
