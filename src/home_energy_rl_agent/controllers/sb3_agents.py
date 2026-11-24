"""Deep RL controllers (PPO, DQN from stable-baselines3) on the same env, reward and actions.

The network sees the env's observation divided by fixed scales so every input is roughly in [-1, 1];
the reward is multiplied by a constant so values are not large. Neither changes the task. The
controllers are scored on the raw env with the same evaluate() as every other controller.

Needs the optional dependencies: uv sync --extra deeprl
"""
import gymnasium as gym
import numpy as np


def obs_scales(obs_names: list[str], demand_ref_kw: float, mask: tuple | list = ()) -> np.ndarray:
    """Divisor per observation entry. demand_ref_kw is a high percentile of TRAIN demand.

    Names in `mask` get an infinite divisor, so the network always sees 0 for them (used for ablations).
    """
    scales = []
    for name in obs_names:
        if name in mask:
            scales.append(np.inf)
        elif name in ("hour_sin", "hour_cos", "soc"):
            scales.append(1.0)
        elif name == "day_of_week":
            scales.append(6.0)
        elif name == "tariff_period":
            scales.append(2.0)
        else:                              # demand, demand lags/leads, peak so far (all kW)
            scales.append(float(demand_ref_kw))
    return np.array(scales, dtype=np.float32)


class ScaledEnv(gym.Wrapper):
    """Scales observations and rewards for training only."""

    def __init__(self, env, scales: np.ndarray, reward_scale: float):
        super().__init__(env)
        self.scales, self.reward_scale = scales, reward_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs / self.scales, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs / self.scales, reward * self.reward_scale, terminated, truncated, info


class SB3Controller:
    """Wraps a trained SB3 model as a controller: act(obs) -> action, greedy (no exploration)."""

    def __init__(self, model, scales: np.ndarray, name: str):
        self.model, self.scales, self.name = model, scales, name

    def reset(self) -> None:
        pass

    def act(self, obs) -> int:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32) / self.scales, deterministic=True)
        return int(action)


def build_model(algo: str, env, hp: dict, seed: int):
    """PPO or DQN with an MLP policy and the hyperparameters from the experiment config."""
    from stable_baselines3 import DQN, PPO

    net = {"net_arch": list(hp["net_arch"])}
    common = {"seed": seed, "verbose": 0, "device": "cpu", "policy_kwargs": net}
    if algo == "ppo":
        return PPO("MlpPolicy", env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
                   batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
                   gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], **common)
    if algo == "dqn":
        return DQN("MlpPolicy", env, learning_rate=hp["learning_rate"], buffer_size=hp["buffer_size"],
                   learning_starts=hp["learning_starts"], batch_size=hp["batch_size"], gamma=hp["gamma"],
                   train_freq=hp["train_freq"], gradient_steps=hp["gradient_steps"],
                   target_update_interval=hp["target_update_interval"],
                   exploration_fraction=hp["exploration_fraction"],
                   exploration_final_eps=hp["exploration_final_eps"], **common)
    raise ValueError(f"unknown algo {algo!r}")
