"""Tests for the demo GIF data and writer (synthetic demand, no trained model needed)."""
import numpy as np
import pandas as pd
import pytest

from home_energy_rl_agent.controllers import build_baselines
from home_energy_rl_agent.demo import make_gif, week_traces
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.evaluate import rollout
from home_energy_rl_agent.tariff import load_tariff_config


@pytest.fixture
def env():
    idx = pd.date_range("2008-03-03", periods=40 * 24, freq="h")
    rng = np.random.default_rng(0)
    demand = pd.Series(0.5 + 1.2 * np.exp(-((idx.hour - 19) ** 2) / 8) + rng.uniform(0, 0.3, len(idx)), index=idx)
    return EnergyEnv(demand, load_env_config(), load_tariff_config(), seed=0)


def test_final_running_bill_equals_the_episode_total_cost(env):
    start = int(env.valid_starts[5])
    for ctrl in build_baselines(env, env.series):
        traces = week_traces(ctrl, env, start)
        summary, _ = rollout(ctrl, env, start)
        assert traces["bill"][-1] == pytest.approx(summary["total_cost"])
        assert traces["peak"][-1] == pytest.approx(summary["peak_kw"])


def test_running_peak_and_bill_never_decrease(env):
    traces = week_traces(build_baselines(env, env.series)[1], env, int(env.valid_starts[2]))
    assert (np.diff(traces["peak"]) >= 0).all() and (np.diff(traces["bill"]) >= 0).all()
    assert len(traces["demand"]) == 168 and traces["soc"].min() >= 0.0


def test_gif_is_written_and_has_several_frames(env, tmp_path):
    from PIL import Image

    start = int(env.valid_starts[2])
    base_ctrl, rule = build_baselines(env, env.series)[0], build_baselines(env, env.series)[1]
    path = make_gif(week_traces(rule, env, start), week_traces(base_ctrl, env, start), "Rule-based",
                    "test title", tmp_path / "demo.gif", step=24, hold_frames=2, dpi=40)
    assert path.exists() and path.stat().st_size > 1000
    with Image.open(path) as im:
        assert im.n_frames >= 5
