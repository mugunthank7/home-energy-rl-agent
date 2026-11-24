"""Tests for the 5-power-level oracle (MILP): exact agreement with the env, bound ordering, local optimality."""
import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cvxpy")

from home_energy_rl_agent.controllers.discrete_oracle import (
    PlanController,
    compute_plans,
    evaluate_discrete_oracle,
    solve_discrete_episode,
)
from home_energy_rl_agent.controllers.oracle import OracleLP
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import choose_starts, evaluate
from home_energy_rl_agent.tariff import load_tariff_config


def make_demand(days: int = 30, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2008-03-03", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    hours = idx.hour
    base = 0.5 + 1.8 * np.exp(-((hours - 19) ** 2) / 8) + 0.6 * np.exp(-((hours - 7) ** 2) / 4)
    return pd.Series(base + rng.uniform(0, 0.4, len(idx)), index=idx, name="demand_kw")


@pytest.fixture
def env():
    cfg = copy.deepcopy(load_env_config())
    cfg["episode_days"] = 1                                   # 24-hour episodes keep the MILP tiny
    return EnergyEnv(make_demand(), cfg, load_tariff_config(), seed=0)


def solve(env, start, levels=None, **kw):
    b = env.battery.p
    sl = slice(int(start), int(start) + env.steps_per_episode)
    levels = env.fractions * b.max_power_kw if levels is None else levels
    return solve_discrete_episode(
        env._demand_all[sl], env._price_all[sl], levels, b.capacity_kwh, b.eta_one_way, b.soc_min,
        b.soc_max, b.soc_start, env.tariff.demand_charge_per_kw, **kw)


def replay(env, start, actions):
    env.reset(options={"start": int(start)})
    for a in actions:
        _, _, _, done, info = env.step(int(a))
    assert done
    return info["episode"]


def test_plan_replays_in_the_env_with_the_predicted_cost_and_no_violations(env):
    for start in choose_starts(env, 4, 3):
        plan = solve(env, start)
        ep = replay(env, start, plan["actions"])
        assert ep["total_cost"] == pytest.approx(plan["objective"], abs=1e-4)
        assert ep["violations"] == 0


def test_bounds_are_ordered(env):
    lp = OracleLP(env.battery.p, env.steps_per_episode, env.tariff.demand_charge_per_kw)
    for start in choose_starts(env, 4, 5):
        sl = slice(int(start), int(start) + env.steps_per_episode)
        d, p = env._demand_all[sl], env._price_all[sl]
        cont = lp.solve(d, p)
        continuous_cost = cont["energy_cost"] + env.tariff.demand_charge_per_kw * cont["peak_kw"]
        plan = solve(env, start)
        no_battery = float(p @ d + env.tariff.demand_charge_per_kw * d.max())
        assert plan["dual_bound"] <= plan["objective"] + 1e-6
        assert continuous_cost <= plan["dual_bound"] + 1e-3          # continuous power can only do better
        assert plan["objective"] <= no_battery + 1e-6                # idle is one of the 5-level plans


def test_no_small_change_to_the_plan_beats_it(env):
    """Local optimality: perturb a few hours of the optimal plan; no violation-free result is cheaper."""
    rng = np.random.default_rng(0)
    start = int(choose_starts(env, 1, 7)[0])
    plan = solve(env, start, rel_gap=1e-6)
    best = plan["objective"]
    checked = 0
    for _ in range(300):
        actions = plan["actions"].copy()
        for h in rng.choice(len(actions), size=int(rng.integers(1, 4)), replace=False):
            actions[h] = int(rng.integers(5))
        ep = replay(env, start, actions)
        if ep["violations"] == 0:
            checked += 1
            assert ep["total_cost"] >= best - 1e-4
    assert checked > 20


def test_fewer_power_levels_can_only_cost_more(env):
    start = int(choose_starts(env, 1, 9)[0])
    five = solve(env, start, rel_gap=1e-6)["objective"]
    three = solve(env, start, levels=np.array([-3.0, 0.0, 3.0]), rel_gap=1e-6)["objective"]
    two = solve(env, start, levels=np.array([0.0, 3.0]), rel_gap=1e-6)["objective"]
    assert five <= three + 1e-4 <= two + 2e-4


def test_plan_controller_in_evaluate_matches_the_milp(env):
    starts = choose_starts(env, 4, 11)
    plans = compute_plans(env, starts, workers=1)
    res = evaluate(PlanController(env, plans), env, starts=starts)
    assert res.summary["total_cost_mean"] == pytest.approx(
        np.mean([plans[int(s)]["objective"] for s in starts]), abs=1e-4)
    assert res.summary["violations_mean"] == 0


def test_terminal_adjusted_variant_optimizes_the_adjusted_cost(env):
    starts = choose_starts(env, 3, 13)
    plans = compute_plans(env, starts, workers=1, value_terminal=True)
    res = evaluate(PlanController(env, plans, "oracle_discrete_adj"), env, starts=starts)
    assert res.per_episode["total_cost_adj"].mean() == pytest.approx(
        np.mean([plans[int(s)]["objective"] for s in starts]), abs=1e-4)


def test_evaluate_discrete_oracle_reports_the_proven_bound(env):
    res = evaluate_discrete_oracle(env, choose_starts(env, 3, 15), workers=1)
    s = res.summary
    assert res.name == "oracle_discrete"
    assert s["dual_bound_savings_pct"] >= s["savings_total_pct"] - 1e-6   # bound can't be below the plan
    assert 0 <= s["bound_gap_mean"] < 0.05
    assert s["violations_mean"] == 0
