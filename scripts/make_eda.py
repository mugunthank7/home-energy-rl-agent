"""EDA figures for the REAL hourly demand data, saved to reports/figures/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from home_energy_rl_agent import data

FIG_DIR = data.ROOT / "reports" / "figures"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


# --- Daily profile: mean demand by hour of day, weekday vs weekend ---
def daily_profile(h: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    weekend = h.index.day_of_week >= 5
    for label, mask in [("Weekday", ~weekend), ("Weekend", weekend)]:
        ax.plot(h[mask].groupby(h[mask].index.hour).mean(), label=label)
    ax.set(xlabel="Hour of day", ylabel="Mean demand (kW)", title="Daily profile (REAL data)")
    ax.legend()
    save(fig, "eda_daily_profile.png")


# --- Weekly profile: mean demand by hour of week ---
def weekly_profile(h: pd.Series) -> None:
    how = h.index.day_of_week * 24 + h.index.hour
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(h.groupby(how).mean())
    ax.set_xticks(np.arange(0, 168, 24) + 12, DAY_NAMES)
    ax.set(ylabel="Mean demand (kW)", title="Weekly profile (REAL data)")
    save(fig, "eda_weekly_profile.png")


# --- Monthly mean: seasonality ---
def monthly_mean(h: pd.Series) -> None:
    m = h.groupby(h.index.month).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(m.index, m.values)
    ax.set(xlabel="Month", ylabel="Mean demand (kW)", title="Monthly mean demand (REAL data)")
    save(fig, "eda_monthly_mean.png")


# --- Load duration curve: demand sorted high to low ---
def load_duration(h: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.linspace(0, 100, len(h)), np.sort(h.to_numpy())[::-1])
    ax.set(xlabel="% of hours", ylabel="Demand (kW)", title="Load duration curve (REAL data)")
    save(fig, "eda_load_duration.png")


# --- Missing-data heatmap: fraction of hours present per day, by month and year ---
def missing_heatmap(h: pd.Series) -> None:
    full = pd.date_range(h.index.min().normalize(), h.index.max().normalize(), freq="D")
    present = h.groupby(h.index.normalize()).size().reindex(full, fill_value=0) / 24
    grid = present.groupby([present.index.year, present.index.month]).mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(grid.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set(xlabel="Month", ylabel="Year", title="Share of days kept after cleaning")
    fig.colorbar(im, ax=ax)
    save(fig, "eda_missing_heatmap.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    h = data.load_hourly()
    for fn in (daily_profile, weekly_profile, monthly_mean, load_duration, missing_heatmap):
        fn(h)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
