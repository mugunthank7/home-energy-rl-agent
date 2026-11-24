"""Controllers share one interface: reset() and act(obs) -> action index."""
import pandas as pd

from home_energy_rl_agent.controllers.base import ActionMap, load_controller_config
from home_energy_rl_agent.controllers.greedy import GreedyThreshold
from home_energy_rl_agent.controllers.no_battery import NoBattery
from home_energy_rl_agent.controllers.rule_based import RuleBased


def build_baselines(env, train_demand: pd.Series, cfg: dict | None = None) -> list:
    """No-battery, rule-based and greedy-threshold controllers for this env.

    Greedy price thresholds come from the TRAIN demand index only.
    """
    cfg = cfg or load_controller_config()
    actions = ActionMap.from_fractions(env.fractions)
    train_prices = env.tariff.prices(env.tariff.periods(train_demand.index))
    return [
        NoBattery(actions),
        RuleBased(actions, env.obs_names, cfg["rule_based"]),
        GreedyThreshold(
            actions, env.obs_names, cfg["greedy_threshold"],
            env.tariff.price_by_period, train_prices,
        ),
    ]
