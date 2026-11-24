"""Gymnasium environment: REAL household demand + SIMULATED battery and tariff.

One step = 1 hour. One episode = `episode_days` days (default 7 = 168 steps).
"""
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces

from home_energy_rl_agent.battery import Battery, BatteryParams
from home_energy_rl_agent.data import valid_episode_starts
from home_energy_rl_agent.tariff import OFF_PEAK, Tariff, load_tariff_config

ROOT = Path(__file__).resolve().parents[2]


def load_env_config(path: str | Path = ROOT / "configs" / "env.yaml") -> dict:
    """Read environment settings from YAML."""
    return yaml.safe_load(Path(path).read_text())


def obs_names(k: int, forecast_h: int) -> list[str]:
    """Observation layout, in order."""
    names = ["hour_sin", "hour_cos", "day_of_week", "soc", "demand_kw", "tariff_period", "peak_kw"]
    names += [f"demand_lag{i}" for i in range(1, k + 1)]
    names += [f"demand_next{i}" for i in range(1, forecast_h + 1)]
    return names


class EnergyEnv(gym.Env):
    """Battery control env. Actions: 0 discharge high ... 2 idle ... 4 charge high."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        demand: pd.Series,
        env_cfg: dict | None = None,
        tariff_cfg: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.cfg = env_cfg or load_env_config()
        self.tariff = Tariff(tariff_cfg or load_tariff_config())
        self.battery = Battery(BatteryParams.from_config(self.cfg["battery"]))
        self.fractions = np.array(self.cfg["actions_fraction_of_max"], dtype=float)
        self.steps_per_episode = int(self.cfg["episode_days"]) * 24
        self.k = int(self.cfg["last_k_demand"])
        self.forecast_h = int(self.cfg["forecast_horizon_h"]) if self.cfg["forecast"] == "perfect" else 0
        r = self.cfg["reward"]
        self.w_peak, self.w_viol, self.w_cycle = r["w_peak"], r["w_viol"], r["w_cycle"]
        # "increase": penalize each rise in the episode peak (the plan's literal formula).
        # "excess": penalize each rise in (peak import - peak demand), i.e. only the peak the
        # battery adds. Peak demand ignores actions, so this changes the reward by an
        # action-independent constant: same best policy, far less noise.
        self.peak_term = r.get("peak_term", "increase")
        if self.peak_term not in ("increase", "excess"):
            raise ValueError(f"unknown peak_term {self.peak_term!r}")

        # --- Precompute per-hour arrays for the whole series so step() stays fast ---
        self.series = demand
        self._demand_all = demand.to_numpy(dtype=float)
        self._period_all = self.tariff.periods(demand.index)
        self._price_all = self.tariff.prices(self._period_all)
        self._hour_all = demand.index.hour.to_numpy()
        self._dow_all = demand.index.day_of_week.to_numpy()
        self.valid_starts = valid_episode_starts(demand, self.cfg["episode_days"])
        if len(self.valid_starts) == 0:
            raise ValueError("Demand series has no contiguous episode-length window.")

        self.action_space = spaces.Discrete(len(self.fractions))
        self.obs_names = obs_names(self.k, self.forecast_h)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.obs_names),), np.float32)
        self.reset(seed=seed)

    # --- Observation ---
    def _obs(self) -> np.ndarray:
        t = min(self.t, self.steps_per_episode - 1)
        i = self.start + t
        hour = self._hour_all[i]
        lags = [self._demand_all[max(i - j, self.start)] for j in range(1, self.k + 1)]
        ahead = [
            self._demand_all[min(i + j, self.end - 1)] for j in range(1, self.forecast_h + 1)
        ]
        vec = [
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            self._dow_all[i], self.battery.soc, self._demand_all[i], self._period_all[i],
            self.peak, *lags, *ahead,
        ]
        return np.array(vec, dtype=np.float32)

    # --- Reset: pick an episode window (random from valid starts, or fixed via options) ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "start" in options:
            self.start = int(options["start"])
        else:
            self.start = int(self.np_random.choice(self.valid_starts))
        self.end = self.start + self.steps_per_episode
        self.t = 0
        self.battery.reset(options.get("soc_start"))
        self.peak = 0.0
        self.demand_peak = 0.0
        self.totals = {
            "energy_cost": 0.0, "throughput_kwh": 0.0, "violations": 0, "demand_kwh": 0.0,
        }
        start_time = str(self.series.index[self.start])
        return self._obs(), {"start": self.start, "start_time": start_time}

    # --- Step: apply the battery action, then compute grid import, cost and reward ---
    def step(self, action: int):
        i = self.start + self.t
        demand, price = self._demand_all[i], self._price_all[i]
        res = self.battery.step(self.fractions[action] * self.battery.p.max_power_kw, demand)

        grid_import = demand + res.charge_kw - res.discharge_kw
        cost = price * grid_import
        prev_peak, prev_demand_peak = self.peak, self.demand_peak
        self.peak = max(prev_peak, grid_import)
        self.demand_peak = max(prev_demand_peak, demand)
        if self.peak_term == "excess":
            peak_increase = (self.peak - self.demand_peak) - (prev_peak - prev_demand_peak)
        else:
            peak_increase = self.peak - prev_peak
        throughput = res.charge_kw + res.discharge_kw
        reward = (
            -cost
            - self.w_peak * peak_increase
            - self.w_viol * float(res.violation)
            - self.w_cycle * throughput
        )

        self.totals["energy_cost"] += cost
        self.totals["throughput_kwh"] += throughput
        self.totals["violations"] += int(res.violation)
        self.totals["demand_kwh"] += demand
        info = {
            "timestamp": str(self.series.index[i]), "step": self.t, "demand": demand,
            "action": int(action), "charge_kw": res.charge_kw, "discharge_kw": res.discharge_kw,
            "soc": self.battery.soc, "grid_import": grid_import, "price": price, "cost": cost,
            "peak": self.peak, "violation": res.violation, "curtailed_kw": res.curtailed_kw,
        }
        self.t += 1
        truncated = self.t >= self.steps_per_episode
        if truncated:
            info["episode"] = self.episode_summary()
        return self._obs(), reward, False, truncated, info

    @property
    def terminal_value_per_kwh(self) -> float:
        """Replacement value of one stored kWh: the off-peak price of buying it back, over efficiency.

        The base reward ignores energy left in the battery at the end of an episode. This value is
        used only for the terminal-adjusted comparison (see docs/METHODOLOGY.md).
        """
        return float(self.tariff.price_by_period[OFF_PEAK] / self.battery.p.eta_one_way)

    # --- Episode totals (energy cost and the demand charge on the episode peak) ---
    def episode_summary(self) -> dict:
        demand_cost = self.tariff.demand_charge_per_kw * self.peak
        cap = self.battery.p.capacity_kwh
        return {
            "energy_cost": self.totals["energy_cost"],
            "demand_charge_cost": demand_cost,
            "total_cost": self.totals["energy_cost"] + demand_cost,
            "peak_kw": self.peak,
            "violations": self.totals["violations"],
            "equivalent_full_cycles": self.totals["throughput_kwh"] / (2 * cap),
            "final_soc": self.battery.soc,
            "demand_kwh": self.totals["demand_kwh"],
        }
