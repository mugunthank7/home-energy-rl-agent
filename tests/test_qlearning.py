"""Tests for the discretizer, epsilon schedules, Q-update, saving, and a small learning smoke test."""
import itertools

import numpy as np
import pandas as pd
import pytest
import yaml

from home_energy_rl_agent.battery import BatteryParams
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.controllers.qlearning import Discretizer, QLearningAgent, epsilon_at
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import evaluate
from home_energy_rl_agent.tariff import load_tariff_config

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "data" / "processed" / "hourly.parquet"
CFG = yaml.safe_load((ROOT / "configs" / "experiments" / "baseline.yaml").read_text())
DISC_CFG, QCFG = CFG["discretization"], CFG["qlearning"]


def make_demand(days: int = 60, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2008-03-03", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    base = 0.5 + 1.2 * np.exp(-((idx.hour - 19) ** 2) / 8) + 0.6 * np.exp(-((idx.hour - 7) ** 2) / 4)
    return pd.Series(base + rng.uniform(0, 0.3, len(idx)), index=idx, name="demand_kw")


@pytest.fixture
def env():
    return EnergyEnv(make_demand(), load_env_config(), load_tariff_config(), seed=0)


def make_agent(env, **overrides):
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, DISC_CFG)
    params = {**QCFG, **overrides}
    actions = ActionMap.from_fractions(env.fractions)
    return QLearningAgent(disc, env.action_space.n, actions.idle, params)


def test_state_space_size_matches_spec(env):
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, DISC_CFG)
    assert disc.n_states == 10 * 24 * 5 * 3


def test_every_observation_maps_to_valid_state_and_roundtrips(env):
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, DISC_CFG)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        obs, _, _, done, _ = env.step(int(rng.integers(5)))
        s = disc(obs)
        assert 0 <= s < disc.n_states
        _, hour_bin, _, _, period = disc.decode(s)
        assert hour_bin == disc.hour(obs)
        assert period == int(obs[env.obs_names.index("tariff_period")])
        if done:
            obs, _ = env.reset()


def test_soc_bins_cover_full_range():
    p = BatteryParams()
    disc = Discretizer(["hour_sin", "hour_cos", "soc", "demand_kw", "tariff_period", "peak_kw"],
                       [1.0], p.soc_min, p.soc_max, 10, 24, 3)

    def bin_of(soc):
        obs = np.array([0.0, 1.0, soc, 0.5, 0, 0.0], dtype=np.float32)
        return disc.decode(disc(obs))[0]

    assert bin_of(p.soc_min) == 0
    assert bin_of(p.soc_max) == 9
    assert bin_of(0.55) == 5


def test_hour_buckets_coarse(env):
    cfg = {**DISC_CFG, "hour_buckets": 6}
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, cfg)
    assert disc.n_states == 10 * 6 * 5 * 3
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        obs, *_ = env.step(2)
        assert 0 <= disc(obs) < disc.n_states


def test_demand_edges_come_from_train_quantiles(env):
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, DISC_CFG)
    expected = np.quantile(env.series.to_numpy(), [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(disc.demand_edges, expected)


@pytest.mark.parametrize("schedule", ["linear", "exponential_fast", "exponential", "exponential_slow"])
def test_epsilon_schedule_endpoints_and_monotone(schedule):
    f = QCFG["schedule_fractions"]
    eps = [epsilon_at(e, 1000, 1.0, 0.05, schedule, f) for e in range(1000)]
    assert eps[0] == pytest.approx(1.0)
    assert eps[-1] == pytest.approx(0.05)
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(eps))


def test_fast_schedule_decays_before_slow():
    f = QCFG["schedule_fractions"]
    fast = epsilon_at(300, 1000, 1.0, 0.05, "exponential_fast", f)
    slow = epsilon_at(300, 1000, 1.0, 0.05, "exponential_slow", f)
    assert fast < slow


def test_unvisited_state_defaults_to_idle(env):
    agent = make_agent(env)
    assert agent.greedy(0) == agent.idle


def test_first_visit_sets_q_to_target_not_a_fraction_of_it():
    """First visit sets Q to the TD target (no zero-init bias). A 1-day episode means the first
    (hour, ...) state cannot recur, so its value is exactly that one first-visit target."""
    cfg = load_env_config()
    cfg["episode_days"] = 1
    env = EnergyEnv(make_demand(), cfg, load_tariff_config(), seed=0)
    agent = make_agent(env, epsilon_start=0.0, epsilon_end=0.0, alpha=0.05, gamma=0.9)
    obs, _ = env.reset(seed=1)
    s = agent.disc(obs)
    a = agent.greedy(s)                         # idle, since nothing has been tried yet
    _, r, *_ = env.step(a)
    agent.train(env, episodes=1, seed=1)        # replays the same first step (same seed, eps=0)
    assert agent.N[s, a] >= 1
    assert agent.Q[s, a] == pytest.approx(r)    # next state was unvisited, so its value is 0


def test_untried_actions_are_never_chosen_over_tried_ones(env):
    agent = make_agent(env)
    agent.Q[7, 0] = -5.0                        # only action 0 has been tried in state 7
    agent.N[7, 0] = 3
    assert agent.greedy(7) == 0                 # untried actions sit at 0 but must be ignored
    assert agent.value(7) == -5.0
    assert agent.value(8) == 0.0                # a never-visited state has value 0


def test_peak_bins_multiply_the_state_space(env):
    cfg = {**DISC_CFG, "peak_quantiles": [0.75, 0.95]}
    disc = Discretizer.fit(env.obs_names, env.series, env.battery.p, cfg)
    assert disc.n_states == 10 * 24 * 5 * 3 * 3
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, *_ = env.step(4)
        s = disc(obs)
        assert 0 <= s < disc.n_states
        assert disc.decode(s)[3] < 3


def test_training_is_deterministic_for_a_seed(env):
    def run():
        a = make_agent(env)
        a.train(env, episodes=20, seed=5)
        return a.Q

    assert np.array_equal(run(), run())


def test_save_and_load_roundtrip(env, tmp_path):
    agent = make_agent(env)
    agent.train(env, episodes=10, seed=2)
    agent.save(tmp_path)
    loaded = QLearningAgent.load(tmp_path, env.obs_names)
    assert np.array_equal(agent.Q, loaded.Q)
    obs, _ = env.reset(seed=3)
    assert agent.act(obs) == loaded.act(obs)


def test_training_improves_reward_and_policy_is_feasible_on_synthetic_data(env):
    """Smoke test on a tiny synthetic series: reward rises a lot and the policy rarely violates.

    Beating no-battery on REAL validation data is checked by the test below (needs the data).
    """
    agent = make_agent(env, alpha=0.02, gamma=0.9)       # the tuned settings
    hist = agent.train(env, episodes=1500, seed=0)
    rewards = np.array(hist["episode_reward"])
    assert rewards[-100:].mean() > rewards[:100].mean() + 100
    assert evaluate(agent, env, episodes=15, seed=11).summary["violations_mean"] < 1.0


@pytest.mark.skipif(not REAL_DATA.exists(), reason="needs data/processed/hourly.parquet")
def test_tuned_q_learning_beats_no_battery_on_real_validation_split():
    """The plan's smoke test: on REAL 2009 demand the tuned agent beats the no-battery cost."""
    from home_energy_rl_agent import data

    splits = data.get_splits(data.load_hourly())
    tuned = yaml.safe_load((ROOT / "configs" / "experiments" / "tuned.yaml").read_text())
    train_env = EnergyEnv(splits["train"], load_env_config(), load_tariff_config(), seed=42)
    val_env = EnergyEnv(splits["validation"], load_env_config(), load_tariff_config(), seed=42)
    disc = Discretizer.fit(train_env.obs_names, splits["train"], train_env.battery.p,
                           tuned["discretization"])
    idle = ActionMap.from_fractions(train_env.fractions).idle
    agent = QLearningAgent(disc, 5, idle, tuned["qlearning"])
    agent.train(train_env, tuned["qlearning"]["episodes"], seed=42)
    res = evaluate(agent, val_env, episodes=50, seed=42).summary
    assert res["savings_total_pct"] > 0
    assert res["violations_mean"] < 1.0
