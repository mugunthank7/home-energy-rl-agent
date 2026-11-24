"""Load, clean and split the REAL UCI household power data.

Output is an hourly demand series in kW (mean kW over an hour equals kWh for that hour).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


# --- Config ---
def load_config(path: str | Path = ROOT / "configs" / "data.yaml") -> dict:
    """Read the data pipeline settings from YAML."""
    return yaml.safe_load(Path(path).read_text())


# --- Raw loading ---
def load_raw(path: str | Path) -> pd.Series:
    """Parse the ';'-separated file into a minute-level kW series; '?' becomes NaN."""
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"],
        dtype={"Global_active_power": "float64"},
    )
    index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return pd.Series(df["Global_active_power"].to_numpy(), index=index, name="demand_kw")


# --- Cleaning ---
def fill_short_gaps(series: pd.Series, max_gap_steps: int) -> pd.Series:
    """Forward-fill only NaN runs of at most max_gap_steps; longer runs stay NaN."""
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_len = is_nan.groupby(run_id).transform("size")
    fillable = is_nan & (run_len <= max_gap_steps)
    return series.where(~fillable, series.ffill())


def to_hourly(minute: pd.Series, max_fill_hours: int) -> tuple[pd.Series, dict]:
    """Fill short gaps, resample to hourly means, drop days that still have gaps."""
    n_missing_raw = int(minute.isna().sum())
    filled = fill_short_gaps(minute, max_fill_hours * 60)
    n_missing_after_fill = int(filled.isna().sum())

    # Hours with any missing minute stay NaN so a partial hour never looks complete.
    hourly = filled.resample("h").apply(lambda x: x.mean() if x.notna().all() else np.nan)

    # Keep only calendar days with all 24 hours present.
    complete_day = hourly.notna().groupby(hourly.index.normalize()).transform("sum") == 24
    kept = hourly[complete_day]

    total_days = hourly.index.normalize().nunique()
    kept_days = kept.index.normalize().nunique()
    log = {
        "raw_minute_rows": len(minute),
        "missing_minutes_raw": n_missing_raw,
        "missing_pct_raw": round(100 * n_missing_raw / len(minute), 3),
        "missing_minutes_after_short_fill": n_missing_after_fill,
        "max_fill_gap_hours": max_fill_hours,
        "total_days": int(total_days),
        "days_kept": int(kept_days),
        "days_dropped": int(total_days - kept_days),
        "hourly_rows_kept": len(kept),
    }
    return kept.rename("demand_kw"), log


def build_hourly(cfg: dict | None = None) -> tuple[pd.Series, dict]:
    """Full pipeline: raw file -> clean hourly series plus a cleaning log."""
    cfg = cfg or load_config()
    minute = load_raw(ROOT / cfg["raw_file"])
    return to_hourly(minute, cfg["max_fill_gap_hours"])


def save_hourly(hourly: pd.Series, log: dict, cfg: dict | None = None) -> None:
    """Write the hourly parquet and the cleaning log JSON."""
    cfg = cfg or load_config()
    out = ROOT / cfg["processed_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_frame().to_parquet(out)
    (ROOT / cfg["cleaning_log_file"]).write_text(json.dumps(log, indent=2))


def load_hourly(cfg: dict | None = None) -> pd.Series:
    """Read the processed hourly series."""
    cfg = cfg or load_config()
    return pd.read_parquet(ROOT / cfg["processed_file"])["demand_kw"]


# --- Splits (by time, never shuffled) ---
def split_by_year(hourly: pd.Series, years: list[int]) -> pd.Series:
    """Rows whose calendar year is in `years`."""
    return hourly[hourly.index.year.isin(years)]


def get_splits(hourly: pd.Series, cfg: dict | None = None) -> dict[str, pd.Series]:
    """Train / validation / test series plus the held-out winter and summer weeks."""
    cfg = cfg or load_config()
    out = {name: split_by_year(hourly, yrs) for name, yrs in cfg["splits"].items()}
    for name, (start, end) in cfg["test_windows"].items():
        window = hourly[start : f"{end} 23:00"]
        out[f"test_{name}"] = window
    return out


def valid_episode_starts(series: pd.Series, days: int = 7) -> np.ndarray:
    """Integer positions where a full contiguous `days`-day window of hourly rows begins.

    Dropped days leave holes in the index, so a window is valid only if its
    first and last timestamps are exactly days*24 - 1 hours apart.
    """
    steps = days * 24
    if len(series) < steps:
        return np.array([], dtype=int)
    idx = series.index.to_numpy()
    span = idx[steps - 1 :] - idx[: len(idx) - steps + 1]
    ok = span == np.timedelta64(steps - 1, "h")
    return np.flatnonzero(ok)
