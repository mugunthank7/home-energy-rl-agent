"""Tables and figures for the final comparison (results/final/records.json) and the parameter record."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.qlearning import QLearningAgent
from home_energy_rl_agent.final_eval import FINAL_DIR
from home_energy_rl_agent.reporting import FIG_DIR, TABLE_DIR

ORDER = ["no_battery", "rule_based", "greedy_threshold", "q_baseline", "q_tuned", "q_selected",
         "oracle_discrete", "oracle"]
LABEL = {
    "no_battery": "No battery", "rule_based": "Rule-based", "greedy_threshold": "Greedy threshold",
    "q_baseline": "Q-learning (plan settings)", "q_tuned": "Q-learning (tuned)",
    "q_selected": "Q-learning (selected: coarse grid)",
    "oracle_discrete": "Oracle, 5 power levels (best plan found)",
    "oracle": "Oracle (perfect foresight, upper bound)",
}
SHORT = {"no_battery": "No battery", "rule_based": "Rule-based", "greedy_threshold": "Greedy threshold",
         "q_baseline": "Q: plan settings", "q_tuned": "Q: tuned", "q_selected": "Q: coarse grid",
         "oracle_discrete": "Oracle 5-level", "oracle": "Oracle (continuous)"}
COLORS = dict(zip(ORDER, ["#888888", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#6a6a6a", "#000000"], strict=True))
SPLIT_TITLE = {
    "validation": "Validation (2009)", "test": "Test (2010)",
    "test_winter_week": "Winter week (Jan 18-24, 2010)", "test_summer_week": "Summer week (Jul 5-11, 2010)",
    "test_winter_season": "Winter weeks (Jan-Feb 2010)", "test_summer_season": "Summer weeks (Jun-Jul 2010)",
}
METRICS = [
    ("reward_mean", "avg episode reward"), ("energy_cost_mean", "energy cost ($/wk)"),
    ("total_cost_mean", "total cost incl. demand charge ($/wk)"), ("savings_energy_pct", "energy savings %"),
    ("savings_total_pct", "total savings %"), ("peak_kw_mean", "peak (kW)"),
    ("peak_reduction_pct", "peak reduction %"), ("violations_mean", "violations / wk"),
    ("equivalent_full_cycles_mean", "full cycles / wk"),
]


def load_records(path: Path = FINAL_DIR / "records.json") -> dict:
    """The final comparison records, plus the oracle's if scripts/run_oracle.py has been run."""
    rec = json.loads(path.read_text())
    for extra in ("oracle.json", "oracle_discrete.json"):
        if (path.parent / extra).exists():
            rec["records"] += json.loads((path.parent / extra).read_text())["records"]
    return rec


def controllers_in(records: list[dict]) -> list[str]:
    present = {r["controller"] for r in records}
    return [c for c in ORDER if c in present]


def seed_stats(records: list[dict], split: str, controller: str, metric: str) -> tuple[float, float]:
    """Mean and std of a summary metric. Q-learning: over seeds. Baselines: single deterministic value."""
    vals = [r["summary"][metric] for r in records if r["split"] == split and r["controller"] == controller]
    return float(np.mean(vals)), float(np.std(vals))


def train_time(records: list[dict], controller: str) -> float | None:
    t = [r["train_time_s"] for r in records if r["controller"] == controller and r["train_time_s"]]
    return float(np.mean(t)) if t else None


# --- Savings, with and without valuing the energy left in the battery, and the share of the oracle ---
ADJ_NAME = {"oracle": "oracle_adj", "oracle_discrete": "oracle_discrete_adj"}   # variants for the adjusted metric


def savings_values(records: list[dict], split: str, controller: str, adjusted: bool) -> list[float]:
    """Total-savings % per seed (one value for deterministic controllers)."""
    key = "savings_total_adj_pct" if adjusted else "savings_total_pct"
    name = ADJ_NAME.get(controller, controller) if adjusted else controller
    return [r["summary"][key] for r in records if r["split"] == split and r["controller"] == name]


def pct_of_oracle(records: list[dict], split: str, controller: str, adjusted: bool, reference: str = "oracle"):
    """(mean, std) of savings as a percentage of a reference oracle's savings, or None if unavailable."""
    ref = savings_values(records, split, reference, adjusted)
    if not ref or abs(ref[0]) < 1e-9:
        return None
    vals = [100 * v / ref[0] for v in savings_values(records, split, controller, adjusted)]
    return float(np.mean(vals)), float(np.std(vals))


def _fmt(controller: str, m: float, s: float, digits: int = 2) -> str:
    return f"{m:.{digits}f} ± {s:.{digits}f}" if controller.startswith("q_") else f"{m:.{digits}f}"


# --- Metrics table for every controller on one split ---
def metrics_table(rec: dict, split: str) -> pd.DataFrame:
    rows = []
    for c in controllers_in(rec["records"]):
        row = {"controller": LABEL[c]}
        for key, name in METRICS:
            m, s = seed_stats(rec["records"], split, c, key)
            row[name] = _fmt(c, m, s)
        adj = savings_values(rec["records"], split, c, adjusted=True)
        row["total savings, terminal-adjusted %"] = _fmt(c, float(np.mean(adj)), float(np.std(adj)))
        t = train_time(rec["records"], c)
        row["train time (s)"] = f"{t:.1f}" if t else "0 (no training)"
        for label, adjusted, ref, hint in [
            ("% of oracle savings", False, "oracle", "scripts/run_oracle.py"),
            ("% of oracle (terminal-adjusted)", True, "oracle", "scripts/run_oracle.py"),
            ("% of 5-level oracle savings", False, "oracle_discrete", "scripts/run_oracle_discrete.py"),
            ("% of 5-level oracle (terminal-adjusted)", True, "oracle_discrete", "scripts/run_oracle_discrete.py"),
        ]:
            share = pct_of_oracle(rec["records"], split, c, adjusted, ref)
            row[label] = f"n/a (run {hint})" if share is None else _fmt(c, *share, digits=1)
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, title: str, note: str = "") -> str:
    lines = [f"### {title}", "", note, "", "| " + " | ".join(df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# --- Generalization: how savings change from validation to test and to held-out seasons ---
def generalization_table(rec: dict) -> pd.DataFrame:
    splits = ["validation", "test", "test_winter_week", "test_summer_week",
              "test_winter_season", "test_summer_season"]
    rows = []
    for c in controllers_in(rec["records"]):
        row = {"controller": LABEL[c]}
        for sp in splits:
            m, s = seed_stats(rec["records"], sp, c, "savings_total_pct")
            row[SPLIT_TITLE[sp]] = f"{m:.1f} ± {s:.1f}" if c.startswith("q_") else f"{m:.1f}"
        if c != "no_battery":
            v, _ = seed_stats(rec["records"], "validation", c, "savings_total_pct")
            t, _ = seed_stats(rec["records"], "test", c, "savings_total_pct")
            row["change val to test (pts)"] = f"{t - v:+.1f}"
        else:
            row["change val to test (pts)"] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_share_table(rec: dict, adjusted: bool = False, reference: str = "oracle") -> pd.DataFrame:
    """Savings as a % of a reference oracle's savings, for every split."""
    splits = ["validation", "test", "test_winter_week", "test_summer_week",
              "test_winter_season", "test_summer_season"]
    rows = []
    for c in controllers_in(rec["records"]):
        row = {"controller": LABEL[c]}
        for sp in splits:
            share = pct_of_oracle(rec["records"], sp, c, adjusted, reference)
            row[SPLIT_TITLE[sp]] = "n/a" if share is None else _fmt(c, *share, digits=1)
        rows.append(row)
    return pd.DataFrame(rows)


def split_stats_table(rec: dict) -> pd.DataFrame:
    rows = [{"split": k, "hours": v["hours"], "mean demand (kW)": f"{v['mean_kw']:.2f}",
             "95th pct (kW)": f"{v['p95_kw']:.2f}", "max (kW)": f"{v['max_kw']:.2f}",
             "mean weekly peak (kW)": f"{v['mean_weekly_peak_kw']:.2f}"} for k, v in rec["split_stats"].items()]
    return pd.DataFrame(rows)


# --- Bar charts: cost, peak and savings by controller, with error bars ---
def _episode_mean_se(rec_row: dict, col: str) -> tuple[float, float]:
    v = np.array(rec_row["per_episode"][col])
    return float(v.mean()), float(v.std(ddof=0) / np.sqrt(max(len(v), 1)))


def _bootstrap_saving_se(rec_row: dict, n: int = 1000) -> float:
    cost, base = np.array(rec_row["per_episode"]["total_cost"]), np.array(rec_row["per_episode"]["base_total_cost"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(cost), (n, len(cost)))
    return float((100 * (1 - cost[idx].mean(1) / base[idx].mean(1))).std())


def plot_controller_bars(rec: dict, split: str, path: Path) -> None:
    ctrls = controllers_in(rec["records"])
    panels = [("total_cost", "total_cost_mean", "Total cost incl. demand charge ($/week)"),
              ("peak_kw", "peak_kw_mean", "Weekly peak grid import (kW)"),
              (None, "savings_total_pct", "Total savings vs no battery (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (col, key, title) in zip(axes, panels, strict=True):
        vals, errs = [], []
        for c in ctrls:
            rows = [r for r in rec["records"] if r["split"] == split and r["controller"] == c]
            if c.startswith("q_"):                       # randomness = training seed
                v = [r["summary"][key] for r in rows]
                vals.append(np.mean(v)); errs.append(np.std(v))
            elif col:                                    # deterministic: SE over episodes
                m, se = _episode_mean_se(rows[0], col)
                vals.append(m); errs.append(se)
            else:
                vals.append(rows[0]["summary"][key]); errs.append(_bootstrap_saving_se(rows[0]))
        ax.bar(range(len(ctrls)), vals, yerr=errs, capsize=3, color=[COLORS[c] for c in ctrls])
        ax.set_xticks(range(len(ctrls)), [SHORT[c] for c in ctrls], fontsize=8, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        if col is None:
            ax.axhline(0, color="k", lw=0.8)
    fig.suptitle(f"{SPLIT_TITLE[split]}: error bars are std over seeds (Q-learning) or standard error "
                 "over episodes (deterministic controllers)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Representative days: demand, grid import, SOC and Q-learning actions ---
def plot_days(days: dict, path: Path) -> None:
    """days[season][controller] = list of 24 info dicts (one day). Seasons are the columns."""
    seasons = list(days)
    fig, axes = plt.subplots(3, len(seasons), figsize=(6.5 * len(seasons), 8), sharex=True, squeeze=False)
    for j, season in enumerate(seasons):
        logs = days[season]
        any_log = next(iter(logs.values()))
        hours = np.arange(len(any_log))
        axes[0, j].plot(hours, [i["demand"] for i in any_log], "k-", lw=2, label="Demand (REAL)")
        for c, log in logs.items():
            style = "--" if c == "oracle" else "-"
            axes[0, j].plot(hours, [i["grid_import"] for i in log], style, color=COLORS[c], label=LABEL[c], lw=1.4)
            axes[1, j].plot(hours, [i["soc"] for i in log], style, color=COLORS[c], lw=1.4)
        axes[0, j].set(title=f"{season}", ylabel="kW")
        axes[1, j].set(ylabel="Battery SOC (SIMULATED)", ylim=(0, 1.02))
        q = next((c for c in logs if c.startswith("q_")), None)
        if q:
            acts = np.array([i["action"] for i in logs[q]])
            axes[2, j].bar(hours, acts - 2, color=["#2166ac" if a < 2 else "#b2182b" if a > 2 else "#999999"
                                                  for a in acts])
            axes[2, j].set(ylabel=f"{LABEL[q]}\naction (- discharge, + charge)", xlabel="Hour of day",
                           yticks=[-2, -1, 0, 1, 2])
        ax2 = axes[2, j].twinx()
        ax2.step(hours, [i["price"] for i in any_log], where="mid", color="tab:red", alpha=0.6)
        ax2.set_ylabel("Price $/kWh (SIMULATED)", color="tab:red")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Q-table policy heatmap: best action by battery level and hour ---
ACTION_COLORS = ["#08519c", "#6baed6", "#f0f0f0", "#fb9a99", "#cb181d"]
ACTION_NAMES = ["discharge high", "discharge low", "idle", "charge low", "charge high"]


def policy_grid(agent: QLearningAgent, demand_bin: int) -> np.ndarray:
    """Greedy action for every (SOC bin, hour bucket) at one demand level; -1 where never visited.

    When an hour bucket spans several tariff periods, the period with the most training visits is shown.
    """
    d = agent.disc
    grid = np.full((d.soc_bins, d.hour_buckets), -1)
    for s in range(d.soc_bins):
        for h in range(d.hour_buckets):
            states = [(((s * d.hour_buckets + h) * d.demand_bins + demand_bin) * d.peak_bins) * d.tariff_periods + p
                      for p in range(d.tariff_periods)]
            visits = [agent.N[st].sum() for st in states]
            if max(visits) > 0:
                grid[s, h] = agent.greedy(states[int(np.argmax(visits))])
    return grid


def plot_policy_heatmap(agent: QLearningAgent, title: str, path: Path) -> None:
    d = agent.disc
    levels = sorted({0, d.demand_bins // 2, d.demand_bins - 1})
    names = {0: "lowest demand bin", d.demand_bins // 2: "middle demand bin", d.demand_bins - 1: "highest demand bin"}
    fig, axes = plt.subplots(1, len(levels), figsize=(4.8 * len(levels), 4), squeeze=False)
    cmap = ListedColormap(["#cccccc", *ACTION_COLORS])
    width = 24 // d.hour_buckets
    for ax, lv in zip(axes[0], levels, strict=True):
        ax.imshow(policy_grid(agent, lv) + 1, cmap=cmap, vmin=0, vmax=5, origin="lower", aspect="auto")
        ax.set_xticks(range(d.hour_buckets), [f"{h * width}" for h in range(d.hour_buckets)],
                      fontsize=7 if d.hour_buckets > 12 else 9)
        ax.set_yticks(range(d.soc_bins), [f"{(d.soc_min + (i + 0.5) * (d.soc_max - d.soc_min) / d.soc_bins):.2f}"
                                          for i in range(d.soc_bins)], fontsize=7)
        ax.set(xlabel="Hour of day (start of bucket)", ylabel="Battery SOC (bin centre)",
               title=names[lv] + " (weekday-like period shown)")
    fig.legend(handles=[Patch(color="#cccccc", label="never visited")]
               + [Patch(color=c, label=n) for c, n in zip(ACTION_COLORS, ACTION_NAMES, strict=True)],
               loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_agent(name: str, seed: int, obs_names: list[str]) -> QLearningAgent:
    return QLearningAgent.load(FINAL_DIR / name / f"seed_{seed}", obs_names)


# --- Parameter record: every setting used, auto-generated from configs and results ---
def _flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    out = []
    for k, v in d.items():
        out += _flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else [(f"{prefix}{k}", v)]
    return out


def parameter_record(path: Path) -> None:
    root = data.ROOT / "configs"
    parts = ["# Parameter record", "",
             ("Generated by `scripts/make_report.py` from `configs/` and `results/`. "
              "Demand is REAL; everything under environment and tariff is SIMULATED."), ""]
    for title, f in [("Environment (simulated battery, reward)", "env.yaml"), ("Tariff (simulated)", "tariff.yaml"),
                     ("Data pipeline (real data)", "data.yaml"), ("Baseline controllers", "controllers.yaml")]:
        rows = _flatten(yaml.safe_load((root / f).read_text()))
        parts += [f"## {title} (`configs/{f}`)", "", "| parameter | value |", "|---|---|"]
        parts += [f"| {k} | {v} |" for k, v in rows] + [""]
    exp = root / "experiments"
    parts += ["## Q-learning experiments (`configs/experiments/`)", "",
              "| experiment | alpha | gamma | epsilon schedule | episodes | grid (SOC x hour x demand x tariff) | env overrides |",
              "|---|---|---|---|---|---|---|"]
    for name in ["baseline", "tuned", "selected", "plan_reward"]:
        e = yaml.safe_load((exp / f"{name}.yaml").read_text())
        q, g = e["qlearning"], e["discretization"]
        parts.append(f"| {name} | {q['alpha']} | {q['gamma']} | {q['epsilon_schedule']} | {q['episodes']} | "
                     f"{g['soc_bins']}x{g['hour_buckets']}x{g['demand_bins']}x{g['tariff_periods']} | "
                     f"{e.get('env_overrides') or 'none'} |")
    parts += ["", "## Sweeps", "", "| sweep | base | changed | seeds | runs on disk |", "|---|---|---|---|---|"]
    for p in sorted(exp.glob("sweep_*.yaml")):
        s = yaml.safe_load(p.read_text())
        changed = (s.get("factor") and f"{s['factor']} in {s['values']}") or \
                  (s.get("grid") and "grid " + "; ".join(f"{k} in {v}" for k, v in s["grid"].items())) or \
                  "; ".join(v["label"] for v in s["variants"])
        n = len(list((data.ROOT / "results" / s["name"]).glob("*.json")))
        parts.append(f"| {s['name']} | {s['base']} | {changed} | {s['seeds']} | {n} |")
    path.write_text("\n".join(parts) + "\n")


def oracle_week(env) -> list[dict] | None:
    """The oracle's plan for the week starting at index 0, as 24-hour records for the Wednesday plot."""
    try:
        from home_energy_rl_agent.controllers.oracle import OracleLP
    except ImportError:
        return None
    try:
        lp = OracleLP(env.battery.p, env.steps_per_episode, env.tariff.demand_charge_per_kw)
    except ImportError:                      # cvxpy not installed
        return None
    d, p = env._demand_all[: env.steps_per_episode], env._price_all[: env.steps_per_episode]
    plan = lp.solve(d, p)
    return [{"demand": d[t], "grid_import": plan["grid_import"][t], "soc": plan["soc"][t], "action": 2,
             "price": p[t]} for t in range(48, 72)]


# --- Driver ---
def report_final() -> None:
    from home_energy_rl_agent.controllers import build_baselines
    from home_energy_rl_agent.env import EnergyEnv, load_env_config
    from home_energy_rl_agent.evaluate import rollout
    from home_energy_rl_agent.tariff import load_tariff_config

    rec = load_records()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    note = ("Q-learning rows: mean ± std over 5 training seeds. Baselines and the oracle are deterministic. "
            "'Terminal-adjusted' values the energy left in the battery at the off-peak replacement price. "
            "'% of oracle' is savings divided by the oracle's savings under the same accounting. "
            "All numbers are simulation results (REAL demand, SIMULATED battery and tariff).")
    for split in ["validation", "test", "test_winter_season", "test_summer_season",
                  "test_winter_week", "test_summer_week"]:
        df = metrics_table(rec, split)
        df.to_csv(TABLE_DIR / f"final_metrics_{split}.csv", index=False)
        (TABLE_DIR / f"final_metrics_{split}.md").write_text(df_to_markdown(df, SPLIT_TITLE[split], note))
    gen = generalization_table(rec)
    gen.to_csv(TABLE_DIR / "generalization.csv", index=False)
    stats = split_stats_table(rec)
    share, share_adj = oracle_share_table(rec), oracle_share_table(rec, adjusted=True)
    share5 = oracle_share_table(rec, reference="oracle_discrete")
    share.to_csv(TABLE_DIR / "oracle_share.csv", index=False)
    share5.to_csv(TABLE_DIR / "oracle_share_5level.csv", index=False)
    (TABLE_DIR / "generalization.md").write_text(
        df_to_markdown(gen, "Total savings vs no battery (%) by split", note)
        + "\n" + df_to_markdown(share, "Savings as a % of the oracle's savings, by split", note)
        + "\n" + df_to_markdown(share_adj, "Same, with leftover battery energy valued (terminal-adjusted)", note)
        + "\n" + df_to_markdown(share5, "Savings as a % of the 5-power-level oracle's savings, by split", note)
        + "\n" + df_to_markdown(stats, "Demand statistics by split (REAL data)"))
    for split in ["validation", "test"]:
        plot_controller_bars(rec, split, FIG_DIR / f"controllers_{split}.png")

    # Representative winter and summer days from the held-out weeks (Wednesday of each week).
    splits = data.get_splits(data.load_hourly())
    env_cfg, tariff = load_env_config(), load_tariff_config()
    days = {}
    for label, key in [("Winter weekday (Wed Jan 20, 2010)", "test_winter"), ("Summer weekday (Wed Jul 7, 2010)", "test_summer")]:
        env = EnergyEnv(splits[key], env_cfg, tariff, seed=42)
        ctrls = {c.name: c for c in build_baselines(env, splits["train"]) if c.name != "no_battery"}
        ctrls["q_selected"] = load_agent("selected", rec["seeds"][0], env.obs_names)
        days[label] = {n: rollout(c, env, 0, keep_log=True)[1][48:72] for n, c in ctrls.items()}
        plan = oracle_week(env)
        if plan is not None:
            days[label]["oracle"] = plan
    plot_days(days, FIG_DIR / "representative_days.png")

    # Q-table policy heatmaps for the plan's baseline grid and the selected coarse grid.
    env = EnergyEnv(splits["validation"], env_cfg, tariff, seed=42)
    for name in ["baseline", "selected"]:
        plot_policy_heatmap(load_agent(name, rec["seeds"][0], env.obs_names),
                            f"Learned Q-learning policy ({name} config, seed {rec['seeds'][0]})",
                            FIG_DIR / f"policy_heatmap_{name}.png")
    parameter_record(TABLE_DIR / "parameter_record.md")

