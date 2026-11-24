"""Train tabular Q-learning for one experiment config, over several seeds.

Trains on the REAL train split, scores the greedy policy on the validation split against the
baselines (same episodes, same reward settings), and saves everything under results/.

Usage: uv run python scripts/train_qlearning.py baseline   (or: tuned, plan_reward)
The 2010 test split is not used here.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.controllers.qlearning import Discretizer, QLearningAgent
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import evaluate
from home_energy_rl_agent.tariff import load_tariff_config

METRICS = ["reward_mean", "energy_cost_mean", "total_cost_mean", "savings_energy_pct",
           "savings_total_pct", "peak_kw_mean", "peak_reduction_pct", "violations_mean",
           "equivalent_full_cycles_mean"]


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# --- One seed: train, then evaluate the greedy policy on validation ---
def run_seed(exp: dict, seed: int, splits: dict, env_cfg: dict, out_dir: Path) -> dict:
    q = exp["qlearning"]
    train_env = EnergyEnv(splits["train"], env_cfg, load_tariff_config(), seed=seed)
    val_env = EnergyEnv(splits["validation"], env_cfg, load_tariff_config(), seed=seed)
    disc = Discretizer.fit(train_env.obs_names, splits["train"], train_env.battery.p,
                           exp["discretization"])
    idle = ActionMap.from_fractions(train_env.fractions).idle
    agent = QLearningAgent(disc, train_env.action_space.n, idle, q)

    def eval_fn(a):
        return evaluate(a, val_env, episodes=q["eval_episodes"], seed=42).summary["reward_mean"]

    history = agent.train(train_env, q["episodes"], seed, q["eval_every"], eval_fn)
    agent.save(out_dir / f"seed_{seed}")
    summary = evaluate(agent, val_env, episodes=50, seed=42).summary   # same 50 weeks as baselines
    summary["train_time_s"] = history["train_time_s"]
    summary["states_visited"] = int((agent.N.sum(axis=1) > 0).sum())
    summary["n_states"] = disc.n_states
    return {"summary": summary, "history": history}


# --- Learning-curve figure: smoothed training reward and greedy validation reward ---
def plot_curves(name: str, runs: dict, baselines: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train = np.array([r["history"]["episode_reward"] for r in runs.values()])
    smooth = np.array([pd.Series(t).rolling(100, min_periods=1).mean() for t in train])
    x = np.arange(train.shape[1])
    axes[0].plot(x, smooth.mean(0))
    axes[0].fill_between(x, smooth.mean(0) - smooth.std(0), smooth.mean(0) + smooth.std(0), alpha=0.25)
    axes[0].set(xlabel="Training episode", ylabel="Reward (100-episode mean)",
                title=f"{name}: training reward (mean ± std over {len(runs)} seeds)")

    ev = np.array([[e["reward"] for e in r["history"]["eval"]] for r in runs.values()])
    eps = [e["episode"] for e in next(iter(runs.values()))["history"]["eval"]]
    axes[1].errorbar(eps, ev.mean(0), ev.std(0), marker="o", capsize=3, label="Q-learning (greedy)")
    for b, s in baselines.items():
        axes[1].axhline(s["reward_mean"], ls="--", lw=1, label=b.replace("_", " "),
                        color={"no_battery": "gray", "rule_based": "tab:red"}.get(b, "tab:green"))
    axes[1].set(xlabel="Training episode", ylabel="Validation reward (20 weeks)",
                title="Greedy policy on validation split")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    name = parser.parse_args().experiment
    exp = yaml.safe_load((data.ROOT / "configs" / "experiments" / f"{name}.yaml").read_text())
    env_cfg = deep_merge(load_env_config(), exp.get("env_overrides", {}))
    splits = data.get_splits(data.load_hourly())
    out_dir = data.ROOT / "results" / "qlearning" / name

    # --- Baselines on the same env settings so reward numbers are comparable ---
    ref_env = EnergyEnv(splits["validation"], env_cfg, load_tariff_config(), seed=42)
    baselines = {c.name: evaluate(c, ref_env, episodes=50, seed=42).summary
                 for c in build_baselines(ref_env, splits["train"])}

    # --- Train every seed ---
    runs = {}
    for seed in exp["seeds"]:
        runs[seed] = run_seed(exp, seed, splits, env_cfg, out_dir)
        s = runs[seed]["summary"]
        print(f"seed {seed}: reward {s['reward_mean']:.2f}, total savings {s['savings_total_pct']:.1f}%, "
              f"violations {s['violations_mean']:.2f}, {s['train_time_s']:.1f}s", flush=True)

    # --- Aggregate over seeds (mean and std) and print next to the baselines ---
    agg = {}
    for m in METRICS + ["train_time_s"]:
        vals = [r["summary"][m] for r in runs.values()]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    rows = [{"controller": f"q_learning ({name}, {len(runs)} seeds)",
             **{m: f"{agg[m]['mean']:.2f} ± {agg[m]['std']:.2f}" for m in METRICS}}]
    rows += [{"controller": b, **{m: f"{s[m]:.2f}" for m in METRICS}} for b, s in baselines.items()]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(rows).to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "experiment": exp, "env_config": env_cfg, "aggregate": agg, "baselines": baselines,
        "seeds": {str(k): v for k, v in runs.items()},
    }, indent=2))
    fig_path = data.ROOT / "reports" / "figures" / f"qlearning_{name}_curve.png"
    plot_curves(name, runs, baselines, fig_path)
    print(f"Saved {out_dir / 'summary.json'} and {fig_path}")


if __name__ == "__main__":
    main()
