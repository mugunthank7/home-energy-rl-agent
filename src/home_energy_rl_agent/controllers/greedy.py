"""Greedy threshold controller: act on the current price relative to train-price percentiles.

Differences from the rule-based controller: thresholds come from price percentiles (not named
tariff periods), charging is gentle (low rate) to avoid creating new demand peaks, and the
discharge rate follows current demand.
"""
import numpy as np

from home_energy_rl_agent.controllers.base import ActionMap, ObsReader


class GreedyThreshold:
    name = "greedy_threshold"

    def __init__(
        self,
        actions: ActionMap,
        obs_names: list[str],
        cfg: dict,
        period_prices: np.ndarray,
        train_prices: np.ndarray,
    ):
        self.actions = actions
        self.obs = ObsReader(obs_names)
        self.cfg = cfg
        self.period_prices = np.asarray(period_prices, dtype=float)
        # Thresholds are computed from TRAIN prices only, never from the evaluation data.
        self.low = float(np.percentile(train_prices, cfg["price_low_percentile"]))
        self.high = float(np.percentile(train_prices, cfg["price_high_percentile"]))

    def reset(self) -> None:
        pass

    def act(self, obs) -> int:
        price = self.period_prices[self.obs.period(obs)]
        soc, demand = self.obs.soc(obs), self.obs.demand(obs)
        if price > self.high and soc > self.cfg["discharge_soc_floor"]:
            if demand >= self.cfg["discharge_high_demand_kw"]:
                return self.actions.discharge_high
            return self.actions.discharge_low
        if price < self.low and soc < self.cfg["charge_soc_max"]:
            return self.actions.charge_low
        return self.actions.idle
