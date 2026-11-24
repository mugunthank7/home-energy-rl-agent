"""One shared evaluation function so every controller is compared on identical episodes."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from home_energy_rl_agent.controllers.base import ActionMap, Controller
from home_energy_rl_agent.controllers.no_battery import NoBattery
from home_energy_rl_agent.env import EnergyEnv

EPISODE_METRICS = [
    "reward", "energy_cost", "demand_charge_cost", "total_cost",
    "peak_kw", "violations", "equivalent_full_cycles",
]


@dataclass
class EvalResult:
    name: str
    per_episode: pd.DataFrame     # one row per episode, with matching no-battery columns
    summary: dict                 # means, stds and savings vs no-battery


# --- Episode selection: same starts for every controller, chosen from the env's valid windows ---
def choose_starts(env: EnergyEnv, episodes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = env.valid_starts
    return rng.choice(pool, size=episodes, replace=episodes > len(pool))


# --- Run one episode from a fixed start; optionally keep the per-step logs for plotting ---
def rollout(controller: Controller, env: EnergyEnv, start: int, keep_log: bool = False):
    obs, _ = env.reset(options={"start": int(start)})
    controller.reset()
    total, done, log = 0.0, False, []
    while not done:
        obs, reward, _, done, info = env.step(controller.act(obs))
        total += reward
        if keep_log:
            log.append(info)
    summary = {k: float(v) for k, v in info["episode"].items()}
    summary["reward"] = float(total)
    return summary, log


def adjusted_total_cost(total_cost, final_soc, env: EnergyEnv):
    """Bill plus the replacement cost of energy used from the starting charge (credit if it ended fuller).

    The plain bill never values the energy left in the battery, so a controller that drains its
    starting charge gets it for free and one that ends full pays for energy it does not use.
    """
    p = env.battery.p
    return total_cost + (p.soc_start - final_soc) * p.capacity_kwh * env.terminal_value_per_kwh


def _run_all(controller, env, starts) -> pd.DataFrame:
    rows = []
    for s in starts:
        summary, _ = rollout(controller, env, s)
        summary["start_time"] = str(env.series.index[int(s)])
        rows.append(summary)
    return pd.DataFrame(rows)


# --- evaluate(): run the controller and a no-battery reference on the same episodes ---
def evaluate(
    controller: Controller,
    env: EnergyEnv,
    episodes: int = 50,
    seed: int = 42,
    starts: np.ndarray | None = None,
) -> EvalResult:
    starts = choose_starts(env, episodes, seed) if starts is None else np.asarray(starts)
    actions = ActionMap.from_fractions(env.fractions)
    df = _run_all(controller, env, starts)
    base = _run_all(NoBattery(actions), env, starts)
    for col in ("energy_cost", "total_cost", "peak_kw"):
        df[f"base_{col}"] = base[col]
    df["total_cost_adj"] = adjusted_total_cost(df["total_cost"], df["final_soc"], env)
    return EvalResult(controller.name, df, summarize(df, controller.name, len(starts), seed))


def summarize(df: pd.DataFrame, name: str, episodes: int, seed: int) -> dict:
    """Means, stds and savings vs the no-battery columns (base_*) of a per-episode frame."""
    summary = {"controller": name, "episodes": episodes, "seed": seed}
    for m in EPISODE_METRICS:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"] = float(df[m].std(ddof=0))
    # Savings are the ratio of mean costs, so one cheap week cannot dominate the percentage.
    summary["savings_energy_pct"] = _saving(df["energy_cost"], df["base_energy_cost"])
    summary["savings_total_pct"] = _saving(df["total_cost"], df["base_total_cost"])
    summary["peak_reduction_pct"] = _saving(df["peak_kw"], df["base_peak_kw"])
    if "total_cost_adj" in df:          # the no-battery run never changes its charge, so adj == plain
        summary["savings_total_adj_pct"] = _saving(df["total_cost_adj"], df["base_total_cost"])
        summary["final_soc_mean"] = float(df["final_soc"].mean())
    return summary


def _saving(value: pd.Series, base: pd.Series) -> float:
    denom = float(base.mean())
    return 0.0 if denom == 0 else float(100 * (1 - value.mean() / denom))


# --- Tables and saving ---
def metrics_table(results: list[EvalResult]) -> pd.DataFrame:
    """One row per controller with the headline metrics."""
    cols = {
        "controller": "controller",
        "reward_mean": "avg_episode_reward",
        "energy_cost_mean": "energy_cost",
        "total_cost_mean": "total_cost_incl_demand_charge",
        "savings_energy_pct": "energy_savings_pct",
        "savings_total_pct": "total_savings_pct",
        "peak_kw_mean": "peak_kw",
        "peak_reduction_pct": "peak_reduction_pct",
        "violations_mean": "violations",
        "equivalent_full_cycles_mean": "equiv_full_cycles",
    }
    return pd.DataFrame([r.summary for r in results])[list(cols)].rename(columns=cols)


def save_results(results: list[EvalResult], path: str | Path, extra: dict | None = None) -> None:
    """Write summaries (and per-episode rows) to JSON under results/."""
    out = {
        "extra": extra or {},
        "results": [
            {"summary": r.summary, "per_episode": r.per_episode.to_dict(orient="records")}
            for r in results
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
