"""Tests for the data pipeline using small synthetic series (no raw file needed)."""
import numpy as np
import pandas as pd

from home_energy_rl_agent import data


def _minutes(days: int, value: float = 1.0) -> pd.Series:
    idx = pd.date_range("2008-01-01", periods=days * 24 * 60, freq="min")
    return pd.Series(value, index=idx, name="demand_kw")


def test_hourly_mean_is_kw():
    s = _minutes(2, 2.0)
    hourly, log = data.to_hourly(s, max_fill_hours=3)
    assert len(hourly) == 48
    assert np.allclose(hourly.to_numpy(), 2.0)
    assert log["days_dropped"] == 0


def test_short_gap_filled_long_gap_drops_day():
    s = _minutes(3, 1.0)
    s.iloc[100:160] = np.nan                  # 1 h gap on day 1: filled
    s.iloc[1440 + 60 : 1440 + 60 + 400] = np.nan  # ~6.7 h gap on day 2: day dropped
    hourly, log = data.to_hourly(s, max_fill_hours=3)
    days = set(hourly.index.normalize())
    assert pd.Timestamp("2008-01-01") in days
    assert pd.Timestamp("2008-01-02") not in days
    assert pd.Timestamp("2008-01-03") in days
    assert log["days_dropped"] == 1


def test_fill_short_gaps_leaves_long_runs():
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 3.0])
    out = data.fill_short_gaps(s, max_gap_steps=2)
    assert out.iloc[1] == 1.0 and out.iloc[2] == 1.0
    assert out.iloc[4:8].isna().all()


def test_split_by_year_is_chronological_and_disjoint():
    idx = pd.date_range("2007-01-01", "2010-12-31 23:00", freq="h")
    s = pd.Series(1.0, index=idx)
    cfg = {"splits": {"train": [2007, 2008], "validation": [2009], "test": [2010]},
           "test_windows": {}}
    sp = data.get_splits(s, cfg)
    assert sp["train"].index.max() < sp["validation"].index.min()
    assert sp["validation"].index.max() < sp["test"].index.min()


def test_valid_episode_starts_skip_holes():
    idx = pd.date_range("2008-01-01", periods=10 * 24, freq="h")
    s = pd.Series(1.0, index=idx)
    s = s.drop(s.index[5 * 24 : 6 * 24])      # drop day 6, leaving a hole
    starts = data.valid_episode_starts(s, days=3)
    for i in starts:
        window = s.index[i : i + 72]
        assert window[-1] - window[0] == pd.Timedelta(hours=71)
    assert len(starts) > 0
