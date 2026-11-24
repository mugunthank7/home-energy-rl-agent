"""Tests for the baseline controllers and the shared evaluate() function."""
import numpy as np
import pandas as pd
import pytest

from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import choose_starts, evaluate, metrics_table
from home_energy_rl_agent.tariff import MID_PEAK, OFF_PEAK, ON_PEAK, load_tariff_config


def make_demand(days: int = 40, seed: int = 0) -> pd.Series:
    """Synthetic demand with a morning and an evening bump (starts on a Monday)."""
    idx = pd.date_range("2008-03-03", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    base = 0.5 + 1.2 * np.exp(-((idx.hour - 19) ** 2) / 8) + 0.6 * np.exp(-((idx.hour - 7) ** 2) / 4)
    return pd.Series(base + rng.uniform(0, 0.3, len(idx)), index=idx, name="demand_kw")


@pytest.fixture
def env():
    return EnergyEnv(make_demand(), load_env_config(), load_tariff_config(), seed=42)


@pytest.fixture
def controllers(env):
    return {c.name: c for c in build_baselines(env, env.series)}


def obs_for(env, **values):
    """Build an observation vector with the given named values (others zero)."""
    obs = np.zeros(len(env.obs_names), dtype=np.float32)
    for k, v in values.items():
        obs[env.obs_names.index(k)] = v
    return obs


def test_action_map_from_fractions():
    a = ActionMap.from_fractions([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert (a.discharge_high, a.discharge_low, a.idle, a.charge_low, a.charge_high) == (0, 1, 2, 3, 4)


def test_rule_based_decisions(env, controllers):
    rb, a = controllers["rule_based"], ActionMap.from_fractions(env.fractions)
    assert rb.act(obs_for(env, tariff_period=OFF_PEAK, soc=0.5)) == a.charge_high
    assert rb.act(obs_for(env, tariff_period=OFF_PEAK, soc=0.95)) == a.idle
    assert rb.act(obs_for(env, tariff_period=ON_PEAK, soc=0.5)) == a.discharge_high
    assert rb.act(obs_for(env, tariff_period=ON_PEAK, soc=0.10)) == a.idle
    assert rb.act(obs_for(env, tariff_period=MID_PEAK, soc=0.5)) == a.idle


def test_greedy_decisions(env, controllers):
    g, a = controllers["greedy_threshold"], ActionMap.from_fractions(env.fractions)
    assert g.act(obs_for(env, tariff_period=OFF_PEAK, soc=0.5)) == a.charge_low
    assert g.act(obs_for(env, tariff_period=ON_PEAK, soc=0.5, demand_kw=3.0)) == a.discharge_high
    assert g.act(obs_for(env, tariff_period=ON_PEAK, soc=0.5, demand_kw=0.5)) == a.discharge_low
    assert g.act(obs_for(env, tariff_period=ON_PEAK, soc=0.10, demand_kw=3.0)) == a.idle
    assert g.act(obs_for(env, tariff_period=MID_PEAK, soc=0.5)) == a.idle


def test_no_battery_has_zero_savings_and_no_activity(env, controllers):
    res = evaluate(controllers["no_battery"], env, episodes=5)
    assert res.summary["savings_total_pct"] == pytest.approx(0.0)
    assert res.summary["violations_mean"] == 0
    assert res.summary["equivalent_full_cycles_mean"] == 0


def test_rule_based_cuts_energy_cost(env, controllers):
    res = evaluate(controllers["rule_based"], env, episodes=10)
    assert res.summary["savings_energy_pct"] > 0


def test_all_controllers_see_identical_episodes(env, controllers):
    starts = [evaluate(c, env, episodes=6, seed=7).per_episode["start_time"].tolist()
              for c in controllers.values()]
    assert starts[0] == starts[1] == starts[2]


def test_evaluate_is_deterministic(env, controllers):
    a = evaluate(controllers["rule_based"], env, episodes=5, seed=3).summary
    b = evaluate(controllers["rule_based"], env, episodes=5, seed=3).summary
    assert a == b


def test_choose_starts_depends_on_seed(env):
    assert list(choose_starts(env, 8, seed=1)) == list(choose_starts(env, 8, seed=1))
    assert list(choose_starts(env, 8, seed=1)) != list(choose_starts(env, 8, seed=2))


def test_metrics_table_has_one_row_per_controller(env, controllers):
    results = [evaluate(c, env, episodes=3) for c in controllers.values()]
    table = metrics_table(results)
    assert len(table) == 3
    assert "total_savings_pct" in table.columns
