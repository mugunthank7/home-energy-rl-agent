"""SIMULATED time-of-use tariff plus a demand charge. Not a real utility's prices."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Tariff period codes (used in observations and by controllers).
OFF_PEAK, MID_PEAK, ON_PEAK = 0, 1, 2


def load_tariff_config(path: str | Path = ROOT / "configs" / "tariff.yaml") -> dict:
    """Read tariff settings from YAML."""
    return yaml.safe_load(Path(path).read_text())


class Tariff:
    """Maps timestamps to a period code and a price per kWh."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.demand_charge_per_kw = float(cfg["demand_charge_per_kw"])
        p = cfg["prices_per_kwh"]
        # Index by period code so prices[period] works on arrays.
        self.price_by_period = np.array([p["off_peak"], p["mid_peak"], p["on_peak"]], dtype=float)

    def periods(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Period code (0 off, 1 mid, 2 on) for each timestamp."""
        hour = index.hour.to_numpy()
        weekday = index.day_of_week.to_numpy() < 5
        on = np.isin(hour, self.cfg["on_peak_hours"])
        if self.cfg["on_peak_weekdays_only"]:
            on &= weekday
        off = np.isin(hour, self.cfg["off_peak_hours"])
        return np.where(on, ON_PEAK, np.where(off, OFF_PEAK, MID_PEAK)).astype(np.int64)

    def prices(self, periods: np.ndarray) -> np.ndarray:
        """Price per kWh for an array of period codes."""
        return self.price_by_period[periods]
