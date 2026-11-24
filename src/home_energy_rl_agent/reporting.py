"""Aggregate saved sweep runs into tables (mean ± std over seeds) and figures."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from home_energy_rl_agent import data

RESULTS_DIR = data.ROOT / "results"
FIG_DIR = data.ROOT / "reports" / "figures"
TABLE_DIR = data.ROOT / "reports" / "tables"

TABLE_METRICS = {
    "savings_total_pct": "total savings %",
    "savings_energy_pct": "energy savings %",
    "peak_reduction_pct": "peak reduction %",
    "violations_mean": "violations / week",
    "equivalent_full_cycles_mean": "cycles / week",
    "train_time_s": "train time (s)",
}


# --- Loading ---
def load_runs(sweep: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """All saved runs of a sweep, sorted by variant order then seed."""
    runs = [json.loads(p.read_text()) for p in (results_dir / sweep).glob("*.json")
            if p.name != "summary.json"]
    return sorted(runs, key=lambda r: (r["order"], r["seed"]))


def runs_frame(runs: list[dict]) -> pd.DataFrame:
    """One row per run with the summary metrics."""
    rows = [{"label": r["label"], "order": r["order"], "seed": r["seed"], **r["summary"]} for r in runs]
    return pd.DataFrame(rows)


# --- Tables ---
def aggregate(runs: list[dict]) -> pd.DataFrame:
    """Per-variant mean and std over seeds, in variant order."""
    df = runs_frame(runs)
    cols = list(TABLE_METRICS)
    g = df.groupby(["order", "label"], sort=True)[cols]
    mean, std = g.mean(), g.std(ddof=0)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["seeds"] = g.size()
    return out.reset_index()


def to_markdown(agg: pd.DataFrame, title: str) -> str:
    lines = [f"### {title}", "", "| variant | " + " | ".join(TABLE_METRICS.values()) + " | seeds |",
             "|---|" + "---|" * (len(TABLE_METRICS) + 1)]
    for _, r in agg.iterrows():
        cells = [f"{r[m + '_mean']:.2f} ± {r[m + '_std']:.2f}" for m in TABLE_METRICS]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + f" | {int(r['seeds'])} |")
    return "\n".join(lines) + "\n"


# --- Figures ---
def plot_sweep(sweep: str, runs: list[dict], agg: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    """Two figures: convergence curves with seed bands, and final metrics with error bars."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = list(agg["label"])
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(labels)))

    # Convergence: smoothed training reward per variant, mean ± std over seeds.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for label, color in zip(labels, colors, strict=True):
        rs = [r for r in runs if r["label"] == label]
        curves = np.array([pd.Series(r["history"]["episode_reward"]).rolling(100, min_periods=1).mean()
                           for r in rs])
        x = np.arange(curves.shape[1])
        axes[0].plot(x, curves.mean(0), color=color, label=label)
        axes[0].fill_between(x, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0),
                             color=color, alpha=0.15)
        ev = np.array([[e["reward"] for e in r["history"]["eval"]] for r in rs])
        eps = [e["episode"] for e in rs[0]["history"]["eval"]]
        axes[1].errorbar(eps, ev.mean(0), ev.std(0), color=color, marker="o", ms=3, capsize=2, label=label)
    axes[0].set(xlabel="Training episode", ylabel="Training reward (100-episode mean)",
                title=f"{sweep}: convergence (mean ± std over seeds)")
    axes[1].set(xlabel="Training episode", ylabel="Validation reward (greedy, 20 weeks)",
                title="Greedy policy on validation split")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{sweep}_convergence.png", dpi=150)
    plt.close(fig)

    # Final metrics: total savings and peak change, with error bars.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(labels))
    for ax, metric, title in zip(
        axes,
        ["savings_total_pct", "peak_reduction_pct", "violations_mean"],
        ["Total savings vs no battery (%)", "Peak reduction vs no battery (%)", "Violations per week"],
        strict=True,
    ):
        ax.bar(x, agg[f"{metric}_mean"], yerr=agg[f"{metric}_std"], capsize=3, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
    fig.suptitle(f"{sweep} (validation split, mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{sweep}_final.png", dpi=150)
    plt.close(fig)


def plot_heatmap(runs: list[dict], metric: str, path: Path, title: str) -> None:
    """alpha x gamma sensitivity heatmap of a metric's mean over seeds."""
    rows = [{"alpha": r["set"]["qlearning.alpha"], "gamma": r["set"]["qlearning.gamma"],
             "value": r["summary"][metric]} for r in runs]
    grid = pd.DataFrame(rows).groupby(["alpha", "gamma"])["value"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lim = np.nanmax(np.abs(grid.to_numpy()))
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set(xlabel="gamma", ylabel="alpha", title=title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Driver: one sweep -> table (markdown + CSV) and figures ---
def report_sweep(sweep: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame | None:
    runs = load_runs(sweep, results_dir)
    if not runs:
        return None
    agg = aggregate(runs)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLE_DIR / f"{sweep}.csv", index=False)
    (TABLE_DIR / f"{sweep}.md").write_text(to_markdown(agg, sweep))
    if sweep.startswith("sweep_grid"):
        plot_heatmap(runs, "savings_total_pct", FIG_DIR / f"{sweep}_heatmap.png",
                     "Total savings % (mean over seeds)")
        plot_heatmap(runs, "violations_mean", FIG_DIR / f"{sweep}_violations_heatmap.png",
                     "Violations per week (mean over seeds)")
    else:
        plot_sweep(sweep, runs, agg)
    return agg
