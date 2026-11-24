"""SIMULATED battery model.

Power numbers are on the grid (AC) side. The round-trip efficiency is split evenly:
charging stores `charge_kw * sqrt(eta)` kWh; delivering `discharge_kw` removes
`discharge_kw / sqrt(eta)` kWh from storage.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class BatteryParams:
    capacity_kwh: float = 10.0
    max_power_kw: float = 3.0
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 1.00
    soc_start: float = 0.50

    @classmethod
    def from_config(cls, cfg: dict) -> "BatteryParams":
        return cls(**cfg)

    @property
    def eta_one_way(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


class BatteryStep(NamedTuple):
    charge_kw: float      # AC power actually drawn to charge
    discharge_kw: float   # AC power actually delivered to the home
    violation: bool       # request was beyond SOC bounds or the power limit
    curtailed_kw: float   # discharge cut because it would exceed demand (no export); not a violation


class Battery:
    def __init__(self, params: BatteryParams):
        self.p = params
        self.soc_kwh = params.soc_start * params.capacity_kwh

    def reset(self, soc_start: float | None = None) -> None:
        frac = self.p.soc_start if soc_start is None else soc_start
        self.soc_kwh = frac * self.p.capacity_kwh

    @property
    def soc(self) -> float:
        """State of charge as a fraction of capacity."""
        return self.soc_kwh / self.p.capacity_kwh

    def step(self, request_kw: float, demand_kw: float) -> BatteryStep:
        """Apply one hour. request_kw > 0 charges, < 0 discharges. Returns what really happened."""
        p, eta = self.p, self.p.eta_one_way
        lo, hi = p.soc_min * p.capacity_kwh, p.soc_max * p.capacity_kwh
        charge = discharge = curtailed = 0.0
        violation = abs(request_kw) > p.max_power_kw + EPS
        request_kw = float(np.clip(request_kw, -p.max_power_kw, p.max_power_kw))

        if request_kw > 0:
            headroom_kw = (hi - self.soc_kwh) / eta          # most AC power the pack can absorb
            charge = min(request_kw, max(headroom_kw, 0.0))
            violation |= charge < request_kw - EPS
            self.soc_kwh += charge * eta
        elif request_kw < 0:
            want = -request_kw
            available_kw = max((self.soc_kwh - lo) * eta, 0.0)  # most AC power the pack can give
            feasible = min(want, available_kw)
            violation |= feasible < want - EPS
            discharge = min(feasible, max(demand_kw, 0.0))      # no export to the grid
            curtailed = feasible - discharge
            self.soc_kwh -= discharge / eta

        self.soc_kwh = float(np.clip(self.soc_kwh, lo - EPS, hi + EPS))
        return BatteryStep(charge, discharge, bool(violation), curtailed)
