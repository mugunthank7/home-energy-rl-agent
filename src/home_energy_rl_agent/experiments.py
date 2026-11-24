"""Config-driven experiment runner: one training run, sweep expansion, parallel execution.

A sweep config (configs/experiments/sweep_*.yaml) names a base experiment and either
`factor` + `values` (one-factor-at-a-time), `grid` (every combination), or explicit `variants`.
Overrides use dotted paths into the experiment config, e.g. `qlearning.alpha` or
`env_overrides.battery.capacity_kwh`. Each run is saved to results/<sweep>/<run_id>.json.
"""
import copy
import itertools
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.controllers.qlearning import Discretizer, QLearningAgent
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import evaluate
from home_energy_rl_agent.tariff import load_tariff_config

EXP_DIR = data.ROOT / "configs" / "experiments"
RESULTS_DIR = data.ROOT / "results"


# --- Config helpers ---
def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment(name: str) -> dict:
    return yaml.safe_load((EXP_DIR / f"{name}.yaml").read_text())


def set_path(cfg: dict, dotted: str, value) -> None:
    """Set cfg['a']['b']['c'] = value for 'a.b.c', creating dicts on the way."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def short(value) -> str:
    """Compact text for a value inside a label."""
    return str(value).replace(" ", "")


# --- Sweep expansion: a sweep config becomes an ordered list of labeled variants ---
def expand_variants(sweep: dict) -> list[dict]:
    """Each variant is {'label': str, 'set': {dotted.path: value}}."""
    if "variants" in sweep:
        return [{"label": v["label"], "set": v["set"]} for v in sweep["variants"]]
    if "grid" in sweep:
        keys = list(sweep["grid"])
        return [
            {"label": ",".join(f"{k.split('.')[-1]}={short(v)}" for k, v in zip(keys, combo, strict=True)),
             "set": dict(zip(keys, combo, strict=True))}
            for combo in itertools.product(*sweep["grid"].values())
        ]
    name = sweep["factor"].split(".")[-1]
    return [{"label": f"{name}={short(v)}", "set": {sweep["factor"]: v}} for v in sweep["values"]]


def resolve_variant(base: dict, variant: dict) -> dict:
    """Base experiment config with the variant's overrides applied."""
    exp = copy.deepcopy(base)
    for path, value in variant["set"].items():
        set_path(exp, path, value)
    return exp


# --- One training run: train on the train split, score the greedy policy on validation ---
def train_and_evaluate(exp: dict, seed: int, splits: dict, save_dir: Path | None = None):
    """Returns (summary, history, agent). The summary includes training time and coverage."""
    env_cfg = deep_merge(load_env_config(), exp.get("env_overrides", {}))
    q, tariff = exp["qlearning"], load_tariff_config()
    train_env = EnergyEnv(splits["train"], env_cfg, tariff, seed=seed)
    val_env = EnergyEnv(splits["validation"], env_cfg, tariff, seed=seed)
    disc = Discretizer.fit(train_env.obs_names, splits["train"], train_env.battery.p,
                           exp["discretization"])
    idle = ActionMap.from_fractions(train_env.fractions).idle
    agent = QLearningAgent(disc, train_env.action_space.n, idle, q)

    def eval_fn(a):
        return evaluate(a, val_env, episodes=q["eval_episodes"], seed=42).summary["reward_mean"]

    history = agent.train(train_env, q["episodes"], seed, q["eval_every"], eval_fn)
    if save_dir is not None:
        agent.save(save_dir)
    summary = evaluate(agent, val_env, episodes=50, seed=42).summary   # same 50 weeks every run
    summary["train_time_s"] = history["train_time_s"]
    summary["states_visited"] = int((agent.N.sum(axis=1) > 0).sum())
    summary["n_states"] = disc.n_states
    return summary, history, agent


def run_id(label: str, seed: int) -> str:
    return f"{slug(label)}_seed{seed}"


def run_job(job: dict) -> dict:
    """Run and save one (sweep, variant, seed). Top-level so worker processes can import it."""
    out = RESULTS_DIR / job["sweep"] / f"{run_id(job['label'], job['seed'])}.json"
    splits = data.get_splits(data.load_hourly())
    summary, history, _ = train_and_evaluate(job["exp"], job["seed"], splits)
    history["episode_reward"] = [round(x, 3) for x in history["episode_reward"]]
    history["epsilon"] = [round(x, 4) for x in history["epsilon"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sweep": job["sweep"], "label": job["label"], "order": job["order"], "seed": job["seed"],
        "set": job["set"], "summary": summary, "history": history,
    }))
    return {"label": job["label"], "seed": job["seed"], "path": str(out)}


# --- Sweep driver: build all jobs, skip finished ones, run the rest in parallel ---
def build_jobs(sweep_name: str, seeds: list[int] | None = None) -> list[dict]:
    sweep = load_experiment(sweep_name)
    base = load_experiment(sweep["base"])
    seeds = seeds or sweep.get("seeds") or base["seeds"]
    jobs = []
    for order, variant in enumerate(expand_variants(sweep)):
        exp = resolve_variant(base, variant)
        for seed in seeds:
            jobs.append({"sweep": sweep["name"], "label": variant["label"], "order": order,
                         "set": variant["set"], "seed": seed, "exp": exp})
    return jobs


def run_sweep(sweep_name: str, workers: int = 6, force: bool = False,
              seeds: list[int] | None = None) -> int:
    """Run every missing job of a sweep. Returns the number of runs executed."""
    jobs = build_jobs(sweep_name, seeds)
    todo = [j for j in jobs
            if force or not (RESULTS_DIR / j["sweep"] / f"{run_id(j['label'], j['seed'])}.json").exists()]
    print(f"{sweep_name}: {len(jobs)} runs, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, j) for j in todo]
        for done, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  [{done}/{len(todo)}] {r['label']} seed {r['seed']}", flush=True)
    return len(todo)
