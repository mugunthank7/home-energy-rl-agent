"""Rule-based controller: charge off-peak, discharge on-peak, otherwise idle."""
from home_energy_rl_agent.controllers.base import ActionMap, ObsReader
from home_energy_rl_agent.tariff import OFF_PEAK, ON_PEAK


class RuleBased:
    name = "rule_based"

    def __init__(self, actions: ActionMap, obs_names: list[str], cfg: dict):
        self.actions = actions
        self.obs = ObsReader(obs_names)
        self.charge_soc_max = cfg["charge_soc_max"]
        self.discharge_soc_floor = cfg["discharge_soc_floor"]

    def reset(self) -> None:
        pass

    def act(self, obs) -> int:
        period, soc = self.obs.period(obs), self.obs.soc(obs)
        # Cheap hours: fill the battery as fast as allowed, up to the SOC ceiling.
        if period == OFF_PEAK and soc < self.charge_soc_max:
            return self.actions.charge_high
        # Expensive hours: empty the battery as fast as allowed, down to the SOC floor.
        if period == ON_PEAK and soc > self.discharge_soc_floor:
            return self.actions.discharge_high
        return self.actions.idle
