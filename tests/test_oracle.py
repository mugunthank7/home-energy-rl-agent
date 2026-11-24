"""Tests for the perfect-foresight LP oracle: constraints, agreement with the simulator, bound property."""
import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from home_energy_rl_agent.battery import Battery
from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.controllers.oracle import OracleLP, evaluate_oracle
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import choose_starts, evaluate
from home_energy_rl_agent.final_eval import spec_starts
from home_energy_rl_agent.tariff import load_tariff_config

TOL = 1e-5


def make_demand(days: int = 40, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2008-03-03", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    hours = idx.hour
    base = 0.5 + 1.2 * np.exp(-((hours - 19) ** 2) / 8) + 0.6 * np.exp(-((hours - 7) ** 2) / 4)
    return pd.Series(base + rng.uniform(0, 0.3, len(idx)), index=idx, name="demand_kw")


@pytest.fixture
def env():
    return EnergyEnv(make_demand(), load_env_config(), load_tariff_config(), seed=0)


def episode_arrays(env, start):
    sl = slice(start, start + env.steps_per_episode)
    return env._demand_all[sl], env._price_all[sl]


def test_plan_satisfies_all_constraints(env):
    b = env.battery.p
    lp = OracleLP(b, env.steps_per_episode, env.tariff.demand_charge_per_kw)
    d, p = episode_arrays(env, int(env.valid_starts[10]))
    plan = lp.solve(d, p)
    e, c, g = plan["energy_kwh"], plan["charge_kw"], plan["discharge_kw"]
    eta = b.eta_one_way
    assert e[0] == pytest.approx(b.soc_start * b.capacity_kwh, abs=TOL)
    assert np.allclose(e[1:], e[:-1] + eta * c - g / eta, atol=TOL)           # energy balance
    assert e.min() >= b.soc_min * b.capacity_kwh - TOL and e.max() <= b.soc_max * b.capacity_kwh + TOL
    assert c.max() <= b.max_power_kw + TOL and g.max() <= b.max_power_kw + TOL
    assert (g <= d + TOL).all()                                                # no export
    assert np.allclose(plan["grid_import"], d + c - g, atol=TOL)
    assert plan["peak_kw"] == pytest.approx(plan["grid_import"].max())


def test_plan_replays_in_the_simulator_battery_without_surprises(env):
    """Feeding the plan to the env's own Battery gives the same charge, discharge, SOC and grid import."""
    lp = OracleLP(env.battery.p, env.steps_per_episode, env.tariff.demand_charge_per_kw)
    for start in choose_starts(env, 5, 3):
        d, p = episode_arrays(env, int(start))
        plan = lp.solve(d, p)
        bat = Battery(env.battery.p)
        for t in range(env.steps_per_episode):
            r = bat.step(plan["charge_kw"][t] - plan["discharge_kw"][t], d[t])
            assert r.charge_kw == pytest.approx(plan["charge_kw"][t], abs=TOL)
            assert r.discharge_kw == pytest.approx(plan["discharge_kw"][t], abs=TOL)
            assert bat.soc == pytest.approx(plan["soc"][t], abs=TOL)


def test_oracle_is_a_lower_bound_on_every_baseline(env):
    starts = choose_starts(env, 12, 7)
    oracle = evaluate_oracle(env, starts).per_episode
    for ctrl in build_baselines(env, env.series):
        other = evaluate(ctrl, env, starts=starts).per_episode
        assert (oracle["total_cost"].to_numpy() <= other["total_cost"].to_numpy() + 1e-6).all(), ctrl.name


def test_oracle_summary_matches_evaluate_format(env):
    starts = choose_starts(env, 6, 7)
    res = evaluate_oracle(env, starts)
    ref = evaluate(build_baselines(env, env.series)[0], env, starts=starts)
    assert set(ref.summary) <= set(res.summary)
    assert res.summary["savings_total_pct"] > 0
    assert res.summary["violations_mean"] == 0
    # The no-battery columns are the same weeks, so the references agree with evaluate().
    assert np.allclose(res.per_episode["base_total_cost"], ref.per_episode["base_total_cost"])
    assert list(res.per_episode["start_time"]) == list(ref.per_episode["start_time"])


def flat_env():
    tariff = copy.deepcopy(load_tariff_config())
    tariff["prices_per_kwh"] = {"off_peak": 0.15, "mid_peak": 0.15, "on_peak": 0.15}
    tariff["demand_charge_per_kw"] = 0.0
    return EnergyEnv(make_demand(), load_env_config(), tariff, seed=0)


def test_flat_prices_leave_nothing_to_save_once_leftover_energy_is_valued():
    """With flat prices and no demand charge the only 'saving' is the free starting charge."""
    env = flat_env()
    starts = choose_starts(env, 5, 1)
    plain = evaluate_oracle(env, starts).summary
    assert plain["savings_total_pct"] > 1.0                      # drains the starting 5 kWh for free
    adjusted = evaluate_oracle(env, starts, value_terminal=True).summary
    assert abs(adjusted["savings_total_adj_pct"]) < 1e-3         # nothing left once it is paid for


def test_peak_shaving_alone_is_found_when_prices_are_flat(env):
    tariff = copy.deepcopy(load_tariff_config())
    tariff["prices_per_kwh"] = {"off_peak": 0.15, "mid_peak": 0.15, "on_peak": 0.15}
    cfg = copy.deepcopy(load_env_config())
    cfg["battery"]["round_trip_efficiency"] = 1.0
    flat = EnergyEnv(make_demand(), cfg, tariff, seed=0)
    res = evaluate_oracle(flat, choose_starts(flat, 5, 1)).per_episode
    assert (res["peak_kw"] < res["base_peak_kw"] - 0.05).all()          # the demand charge pays for shaving


def test_price_spread_is_exploited(env):
    res = evaluate_oracle(env, choose_starts(env, 5, 1))
    assert res.summary["savings_energy_pct"] > 10


def test_spec_starts_match_what_evaluate_uses(env):
    spec = {"name": "x", "episodes": 6}
    ctrl = build_baselines(env, env.series)[0]
    via_evaluate = evaluate(ctrl, env, episodes=6, seed=42).per_episode["start_time"].tolist()
    oracle_starts = evaluate_oracle(env, spec_starts(env, spec)).per_episode["start_time"].tolist()
    assert via_evaluate == oracle_starts


def test_adjusted_oracle_is_a_lower_bound_under_the_adjusted_metric(env):
    starts = choose_starts(env, 12, 7)
    oracle = evaluate_oracle(env, starts, value_terminal=True).per_episode
    for ctrl in build_baselines(env, env.series):
        other = evaluate(ctrl, env, starts=starts).per_episode
        assert (oracle["total_cost_adj"].to_numpy() <= other["total_cost_adj"].to_numpy() + 1e-6).all()


def test_terminal_value_is_off_peak_price_over_efficiency(env):
    expected = 0.08 / env.battery.p.eta_one_way
    assert env.terminal_value_per_kwh == pytest.approx(expected)


def test_adjusted_cost_credits_a_battery_that_ends_fuller_and_charges_one_that_ends_emptier(env):
    from home_energy_rl_agent.evaluate import adjusted_total_cost

    cap, v = env.battery.p.capacity_kwh, env.terminal_value_per_kwh
    assert adjusted_total_cost(10.0, 0.5, env) == pytest.approx(10.0)                  # unchanged
    assert adjusted_total_cost(10.0, 0.9, env) == pytest.approx(10.0 - 0.4 * cap * v)   # ended fuller
    assert adjusted_total_cost(10.0, 0.1, env) == pytest.approx(10.0 + 0.4 * cap * v)   # ended emptier


def test_no_battery_has_equal_plain_and_adjusted_savings(env):
    res = evaluate(build_baselines(env, env.series)[0], env, episodes=4)
    assert res.summary["savings_total_adj_pct"] == pytest.approx(res.summary["savings_total_pct"])
    assert res.summary["final_soc_mean"] == pytest.approx(env.battery.p.soc_start)
