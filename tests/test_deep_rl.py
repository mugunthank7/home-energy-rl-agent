"""Tests for the deep RL wrappers, controller and config wiring (tiny runs only)."""
import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from home_energy_rl_agent.controllers.sb3_agents import (
    SB3Controller,
    ScaledEnv,
    build_model,
    obs_scales,
)
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import choose_starts, evaluate
from home_energy_rl_agent.experiments import load_experiment
from home_energy_rl_agent.tariff import load_tariff_config


def make_env(days: int = 1) -> EnergyEnv:
    idx = pd.date_range("2008-03-03", periods=30 * 24, freq="h")
    rng = np.random.default_rng(0)
    demand = pd.Series(0.6 + 1.2 * np.exp(-((idx.hour - 19) ** 2) / 8) + rng.uniform(0, 0.3, len(idx)), index=idx)
    cfg = copy.deepcopy(load_env_config())
    cfg["episode_days"] = days
    return EnergyEnv(demand, cfg, load_tariff_config(), seed=0)


def test_obs_scales_bring_every_input_to_a_similar_range():
    env = make_env()
    scales = obs_scales(env.obs_names, demand_ref_kw=3.0)
    by_name = dict(zip(env.obs_names, scales, strict=True))
    assert by_name["day_of_week"] == 6.0 and by_name["tariff_period"] == 2.0
    assert by_name["demand_kw"] == by_name["peak_kw"] == by_name["demand_lag1"] == 3.0
    assert by_name["soc"] == 1.0 and by_name["hour_sin"] == 1.0
    obs, _ = env.reset(seed=0)
    assert np.abs(obs / scales).max() < 3.0


def test_masked_inputs_are_always_zero_for_the_network():
    env = make_env()
    scales = obs_scales(env.obs_names, 3.0, mask=["peak_kw"])
    i = env.obs_names.index("peak_kw")
    assert np.isinf(scales[i]) and np.isfinite(np.delete(scales, i)).all()
    wrapped = ScaledEnv(make_env(), scales, 0.1)
    obs, _ = wrapped.reset(seed=0)
    for _ in range(10):
        obs, *_ = wrapped.step(4)                  # charging raises the peak, but the network cannot see it
        assert obs[i] == 0.0


def test_scaled_env_scales_obs_and_reward_but_not_the_task():
    env = make_env()
    scales = obs_scales(env.obs_names, 3.0)
    wrapped = ScaledEnv(make_env(), scales, reward_scale=0.1)
    raw_obs, _ = env.reset(seed=1)
    obs, _ = wrapped.reset(seed=1)
    assert np.allclose(obs, raw_obs / scales)
    raw = env.step(2)
    scaled = wrapped.step(2)
    assert scaled[1] == pytest.approx(raw[1] * 0.1)
    assert np.allclose(scaled[0], raw[0] / scales)


@pytest.mark.parametrize("algo", ["ppo", "dqn"])
def test_real_configs_build_models(algo):
    cfg = load_experiment(algo)
    env = make_env()
    model = build_model(algo, ScaledEnv(env, obs_scales(env.obs_names, 3.0), cfg["reward_scale"]),
                        cfg["hyperparameters"], seed=0)
    assert model.observation_space.shape[0] == len(env.obs_names)
    assert cfg["total_timesteps"] == 3000 * 168            # same sample budget as Q-learning


@pytest.mark.parametrize("algo", ["ppo", "dqn"])
def test_tiny_training_run_gives_a_working_controller(algo):
    env = make_env()
    scales = obs_scales(env.obs_names, 3.0)
    hp = copy.deepcopy(load_experiment(algo)["hyperparameters"])
    if algo == "ppo":
        hp.update(n_steps=48, batch_size=24, n_epochs=2)
    else:
        hp.update(learning_starts=50, buffer_size=2000)
    model = build_model(algo, ScaledEnv(make_env(), scales, 0.1), hp, seed=0)
    model.learn(total_timesteps=300)
    ctrl = SB3Controller(model, scales, algo)
    obs, _ = env.reset(seed=2)
    assert ctrl.act(obs) in range(5)
    assert ctrl.act(obs) == ctrl.act(obs)                   # greedy: no exploration noise
    res = evaluate(ctrl, env, starts=choose_starts(env, 3, 5))
    assert res.summary["controller"] == algo
    assert "savings_total_adj_pct" in res.summary


def test_training_is_reproducible_for_a_seed():
    def run():
        env = make_env()
        scales = obs_scales(env.obs_names, 3.0)
        hp = copy.deepcopy(load_experiment("dqn")["hyperparameters"])
        hp.update(learning_starts=50, buffer_size=2000)
        model = build_model("dqn", ScaledEnv(make_env(), scales, 0.1), hp, seed=3)
        model.learn(total_timesteps=300)
        obs, _ = env.reset(seed=4)
        return model.predict(obs / scales, deterministic=True)[0]

    assert run() == run()
