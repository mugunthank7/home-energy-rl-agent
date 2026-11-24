"""Train PPO / DQN on the train split and score them on every split, exactly like the other controllers.

Hyperparameters live in configs/experiments/ppo.yaml and dqn.yaml and were fixed before the 2010 test
split was evaluated. The sample budget equals Q-learning's (3000 episodes x 168 steps).
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.sb3_agents import (
    SB3Controller,
    ScaledEnv,
    build_model,
    obs_scales,
)
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import evaluate
from home_energy_rl_agent.experiments import load_experiment
from home_energy_rl_agent.final_eval import FINAL_DIR, _record, eval_specs, spec_starts
from home_energy_rl_agent.tariff import load_tariff_config

DEEP_DIR = data.ROOT / "results" / "deep_rl"


def train_and_score(args: tuple) -> dict:
    """Train one (algo, seed), then evaluate on every split. Returns records plus the learning curve."""
    algo, seed, overrides = args
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)                                   # many workers run side by side
    cfg = load_experiment(algo)
    cfg.update(overrides)
    name = f"{algo}_{cfg['tag']}" if cfg.get("tag") else algo      # e.g. ppo_nopeak for the ablation
    splits = data.get_splits(data.load_hourly())
    env_cfg, tariff = load_env_config(), load_tariff_config()
    train_env = EnergyEnv(splits["train"], env_cfg, tariff, seed=seed)
    val_env = EnergyEnv(splits["validation"], env_cfg, tariff, seed=seed)
    scales = obs_scales(train_env.obs_names, float(splits["train"].quantile(cfg["demand_ref_quantile"])),
                        cfg.get("mask_obs", []))
    model = build_model(algo, ScaledEnv(train_env, scales, cfg["reward_scale"]), cfg["hyperparameters"], seed)
    curve = []

    class ValidationCurve(BaseCallback):
        """Greedy validation reward every `eval_every` steps, for the learning curve."""

        def __init__(self):
            super().__init__()
            self.next_at = cfg["eval_every"]

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_at:
                ctrl = SB3Controller(self.model, scales, name)
                r = evaluate(ctrl, val_env, episodes=cfg["eval_episodes"], seed=42).summary["reward_mean"]
                curve.append({"timesteps": int(self.num_timesteps), "reward": float(r)})
                self.next_at += cfg["eval_every"]
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=ValidationCurve())
    train_time = time.perf_counter() - t0
    out_dir = DEEP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"seed_{seed}")

    controller = SB3Controller(model, scales, name)
    records = []
    for spec in eval_specs(splits):
        env = EnergyEnv(spec["series"], env_cfg, tariff, seed=42)
        result = evaluate(controller, env, starts=spec_starts(env, spec), seed=42)
        records.append(_record(spec["name"], name, seed, result, train_time))
    return {"algo": name, "seed": seed, "train_time_s": train_time, "curve": curve, "records": records}


def run_deep_rl(algos: list[str], seeds: list[int], workers: int = 10, overrides: dict | None = None,
                out_path: Path = FINAL_DIR / "deep_rl.json") -> dict:
    """Train every (algo, seed) in parallel and save all records to one JSON."""
    jobs = [(a, s, overrides or {}) for a in algos for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = []
        for r in pool.map(train_and_score, jobs):
            print(f"trained {r['algo']} seed {r['seed']} ({r['train_time_s']:.0f}s)", flush=True)
            results.append(r)
    out = {"records": [rec for r in results for rec in r["records"]],
           "curves": [{"algo": r["algo"], "seed": r["seed"], "curve": r["curve"]} for r in results]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    return out


__all__ = ["np", "run_deep_rl", "train_and_score"]
