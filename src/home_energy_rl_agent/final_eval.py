"""Final comparison: train the chosen Q-learning configs on 2007-08, then score every controller on
validation (2009), test (2010), and the held-out winter/summer weeks and seasons of 2010.

The test split is only used here, after the configs were fixed from the validation sweeps.
"""
import json
from pathlib import Path

import numpy as np

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import choose_starts, evaluate
from home_energy_rl_agent.experiments import deep_merge, load_experiment, train_and_evaluate
from home_energy_rl_agent.tariff import load_tariff_config

FINAL_DIR = data.ROOT / "results" / "final"
Q_CONFIGS = ["baseline", "tuned", "selected"]
SEEDS = [42, 43, 44, 45, 46]
WINTER_MONTHS, SUMMER_MONTHS = [1, 2], [6, 7]     # July is used; August had the household away
PER_EPISODE_COLS = ["reward", "energy_cost", "total_cost", "peak_kw", "violations",
                    "equivalent_full_cycles", "base_energy_cost", "base_total_cost", "base_peak_kw",
                    "final_soc", "total_cost_adj"]


def season_starts(env: EnergyEnv, months: list[int]) -> np.ndarray:
    """Valid episode starts at midnight on days in the given months (one 7-day window per day)."""
    idx = env.series.index[env.valid_starts]
    return env.valid_starts[(idx.month.isin(months)) & (idx.hour == 0)]


def spec_starts(env: EnergyEnv, spec: dict, seed: int = 42) -> np.ndarray:
    """The exact episode starts evaluate() uses for a spec (so the oracle scores identical weeks)."""
    if "months" in spec:
        return season_starts(env, spec["months"])
    return choose_starts(env, spec["episodes"], seed)


def eval_specs(splits: dict) -> list[dict]:
    """The splits every controller is scored on, and how episodes are chosen on each."""
    return [
        {"name": "validation", "series": splits["validation"], "episodes": 50},
        {"name": "test", "series": splits["test"], "episodes": 100},
        {"name": "test_winter_week", "series": splits["test_winter"], "episodes": 1},
        {"name": "test_summer_week", "series": splits["test_summer"], "episodes": 1},
        {"name": "test_winter_season", "series": splits["test"], "months": WINTER_MONTHS},
        {"name": "test_summer_season", "series": splits["test"], "months": SUMMER_MONTHS},
    ]


def split_stats(splits: dict) -> dict:
    """Plain descriptive statistics of REAL demand per split (used to discuss distribution shift)."""
    out = {}
    for name, s in splits.items():
        weekly_peak = s.groupby(s.index.to_period("W")).max()
        out[name] = {"hours": len(s), "mean_kw": float(s.mean()), "p95_kw": float(s.quantile(0.95)),
                     "max_kw": float(s.max()), "mean_weekly_peak_kw": float(weekly_peak.mean())}
    return out


def _record(split, controller, seed, result, train_time=None) -> dict:
    df = result.per_episode
    return {"split": split, "controller": controller, "seed": seed, "train_time_s": train_time,
            "summary": result.summary,
            "per_episode": {c: [round(float(v), 4) for v in df[c]] for c in PER_EPISODE_COLS}}


def run_final(seeds: list[int] = SEEDS, out_dir: Path = FINAL_DIR) -> dict:
    """Train every config/seed on the train split, then evaluate everything on every split."""
    splits = data.get_splits(data.load_hourly())
    env_cfg, tariff = load_env_config(), load_tariff_config()

    # --- Train (sequentially, so the recorded training times are honest) ---
    agents = {}
    for name in Q_CONFIGS:
        exp = load_experiment(name)
        cfg = deep_merge(env_cfg, exp.get("env_overrides", {}))
        for seed in seeds:
            _, history, agent = train_and_evaluate(exp, seed, splits, out_dir / name / f"seed_{seed}")
            agents[(name, seed)] = (agent, history["train_time_s"], cfg)
            print(f"trained {name} seed {seed} ({history['train_time_s']:.1f}s)", flush=True)

    # --- Evaluate every controller on every split with the same episodes ---
    records = []
    for spec in eval_specs(splits):
        env = EnergyEnv(spec["series"], env_cfg, tariff, seed=42)
        starts = season_starts(env, spec["months"]) if "months" in spec else None
        kw = {"starts": starts} if starts is not None else {"episodes": spec["episodes"]}
        for ctrl in build_baselines(env, splits["train"]):
            records.append(_record(spec["name"], ctrl.name, None, evaluate(ctrl, env, seed=42, **kw)))
        for (name, seed), (agent, t, _) in agents.items():
            records.append(_record(spec["name"], f"q_{name}", seed, evaluate(agent, env, seed=42, **kw), t))
        print(f"evaluated {spec['name']}", flush=True)

    out = {"seeds": seeds, "q_configs": Q_CONFIGS, "split_stats": split_stats(splits), "records": records}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(json.dumps(out))
    return out
