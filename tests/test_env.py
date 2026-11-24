"""Tests for the battery, tariff and EnergyEnv using a small synthetic demand series."""
import copy

import numpy as np
import pandas as pd
import pytest

from home_energy_rl_agent.battery import Battery, BatteryParams
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.tariff import MID_PEAK, OFF_PEAK, ON_PEAK, Tariff, load_tariff_config

IDLE, DISCHARGE_HIGH, CHARGE_HIGH = 2, 0, 4


def make_demand(days: int = 30, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2008-03-03", periods=days * 24, freq="h")   # starts on a Monday
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0.3, 3.0, len(idx)), index=idx, name="demand_kw")


def make_env(seed: int = 42, demand: pd.Series | None = None) -> EnergyEnv:
    demand = make_demand() if demand is None else demand
    return EnergyEnv(demand, load_env_config(), load_tariff_config(), seed=seed)


def run_episode(env: EnergyEnv, policy, seed: int = 1):
    """Run one episode, returning the list of info dicts and the SOC (kWh) before each step."""
    env.reset(seed=seed)
    infos, soc_before, done = [], [], False
    obs = env._obs()
    while not done:
        soc_before.append(env.battery.soc_kwh)
        obs, _, _, done, info = env.step(policy(obs))
        infos.append(info)
    return infos, soc_before


def test_energy_conservation():
    env = make_env()
    eta = env.battery.p.eta_one_way
    rng = np.random.default_rng(5)
    infos, soc_before = run_episode(env, lambda o: int(rng.integers(5)))
    for info, soc0 in zip(infos, soc_before, strict=True):
        # Grid supplies demand plus charging, minus what the battery delivers.
        assert info["grid_import"] == pytest.approx(
            info["demand"] + info["charge_kw"] - info["discharge_kw"]
        )
        # Stored energy changes by charge*eta in and discharge/eta out.
        soc1 = info["soc"] * env.battery.p.capacity_kwh
        assert soc1 - soc0 == pytest.approx(info["charge_kw"] * eta - info["discharge_kw"] / eta)


def test_soc_always_in_bounds():
    env = make_env()
    b = env.battery.p
    rng = np.random.default_rng(6)
    for seed in range(5):
        infos, _ = run_episode(env, lambda o: int(rng.integers(5)), seed=seed)
        socs = np.array([i["soc"] for i in infos])
        assert socs.min() >= b.soc_min - 1e-6
        assert socs.max() <= b.soc_max + 1e-6


def test_no_export_and_nonnegative_import():
    env = make_env()
    infos, _ = run_episode(env, lambda o: DISCHARGE_HIGH)
    assert all(i["grid_import"] >= -1e-9 for i in infos)
    assert all(i["discharge_kw"] <= i["demand"] + 1e-9 for i in infos)


def test_no_battery_baseline_equals_demand_times_price():
    demand = make_demand()
    env = make_env(demand=demand)
    infos, _ = run_episode(env, lambda o: IDLE)
    start = env.start
    window = demand.iloc[start : start + 168]
    prices = env.tariff.prices(env.tariff.periods(window.index))
    summary = infos[-1]["episode"]
    assert summary["energy_cost"] == pytest.approx(float((window.to_numpy() * prices).sum()))
    assert summary["peak_kw"] == pytest.approx(window.max())
    assert summary["violations"] == 0
    assert summary["equivalent_full_cycles"] == 0.0


def test_determinism_under_seed():
    def trajectory(seed):
        env = make_env(seed=seed)
        rng = np.random.default_rng(0)
        infos, _ = run_episode(env, lambda o: int(rng.integers(5)), seed=seed)
        return [(i["timestamp"], i["grid_import"], i["soc"]) for i in infos]

    assert trajectory(42) == trajectory(42)
    assert trajectory(42) != trajectory(43)


def test_charging_a_full_battery_is_a_violation():
    b = Battery(BatteryParams(soc_start=1.0))
    res = b.step(request_kw=3.0, demand_kw=1.0)
    assert res.violation and res.charge_kw == 0.0


def test_discharging_an_empty_battery_is_a_violation():
    b = Battery(BatteryParams(soc_start=0.10))     # already at the floor
    res = b.step(request_kw=-3.0, demand_kw=2.0)
    assert res.violation and res.discharge_kw == 0.0


def test_demand_cap_is_curtailed_not_a_violation():
    b = Battery(BatteryParams(soc_start=0.9))
    res = b.step(request_kw=-3.0, demand_kw=1.0)
    assert not res.violation
    assert res.discharge_kw == pytest.approx(1.0)
    assert res.curtailed_kw == pytest.approx(2.0)


def test_round_trip_efficiency():
    b = Battery(BatteryParams(soc_start=0.5, round_trip_efficiency=0.81))
    e0 = b.soc_kwh
    charged = b.step(2.0, 5.0).charge_kw
    delivered = b.step(-2.0, 5.0).discharge_kw
    # Net: put in `charged` kWh at AC side, got back less; storage change is the loss.
    assert delivered == pytest.approx(charged)         # same AC power both ways...
    assert e0 - b.soc_kwh == pytest.approx(2.0 * (1 / 0.9 - 0.9))  # ...so storage ends lower


def test_tariff_periods():
    t = Tariff(load_tariff_config())
    idx = pd.DatetimeIndex(
        ["2008-03-03 02:00", "2008-03-03 12:00", "2008-03-03 18:00", "2008-03-08 18:00"]
    )  # Mon night, Mon noon, Mon evening, Sat evening
    assert list(t.periods(idx)) == [OFF_PEAK, MID_PEAK, ON_PEAK, MID_PEAK]
    prices = t.prices(t.periods(idx))
    assert prices[2] > prices[1] > prices[0]


def test_episodes_never_span_a_data_hole():
    demand = make_demand(days=20)
    demand = demand.drop(demand.index[10 * 24 : 11 * 24])        # remove day 11
    env = make_env(demand=demand)
    for seed in range(20):
        env.reset(seed=seed)
        window = demand.index[env.start : env.start + 168]
        assert window[-1] - window[0] == pd.Timedelta(hours=167)


def test_forecast_option_changes_observation_size():
    cfg = copy.deepcopy(load_env_config())
    cfg["forecast"] = "perfect"
    env = EnergyEnv(make_demand(), cfg, load_tariff_config(), seed=0)
    base = make_env()
    assert env.observation_space.shape[0] == base.observation_space.shape[0] + 3
