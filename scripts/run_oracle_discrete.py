"""Score the perfect-foresight oracle restricted to the env's 5 power levels (a mixed-integer program).

Writes results/final/oracle_discrete.json, which make_report.py merges into the tables. Takes a few
minutes (about 530 small MILPs, run in parallel). Needs scipy only (installed with cvxpy or pandas stack).
"""
import argparse
import json
import time

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.discrete_oracle import evaluate_discrete_oracle
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.final_eval import FINAL_DIR, _record, eval_specs, spec_starts
from home_energy_rl_agent.tariff import load_tariff_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=30.0, help="seconds per week")
    args = parser.parse_args()
    splits = data.get_splits(data.load_hourly())
    env_cfg, tariff = load_env_config(), load_tariff_config()
    records = []
    for spec in eval_specs(splits):
        env = EnergyEnv(spec["series"], env_cfg, tariff, seed=42)
        starts = spec_starts(env, spec)
        for value_terminal in (False, True):
            t0 = time.time()
            result = evaluate_discrete_oracle(env, starts, 42, args.workers, value_terminal, args.time_limit)
            records.append(_record(spec["name"], result.name, None, result))
            s = result.summary
            key = "savings_total_adj_pct" if value_terminal else "savings_total_pct"
            print(f"{spec['name']} [{result.name}]: savings {s[key]:.1f}% (proven bound {s['dual_bound_savings_pct']:.1f}%), "
                  f"mean gap {100 * s['bound_gap_mean']:.2f}%, {time.time() - t0:.0f}s", flush=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "oracle_discrete.json").write_text(json.dumps({"records": records}))
    print(f"Saved {FINAL_DIR / 'oracle_discrete.json'}")


if __name__ == "__main__":
    main()
