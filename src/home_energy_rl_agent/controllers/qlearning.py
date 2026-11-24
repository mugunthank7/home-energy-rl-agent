"""Tabular Q-learning controller.

State = (SOC bin, hour bucket, demand-level bin, tariff period), discretized from the env's
observation. Demand bins are quantiles of TRAIN demand only.
"""
import json
import math
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from home_energy_rl_agent.battery import BatteryParams
from home_energy_rl_agent.controllers.base import ObsReader


# --- Discretizer: turn a continuous observation into one integer state index ---
class Discretizer:
    def __init__(
        self,
        obs_names: list[str],
        demand_edges: list[float],
        soc_min: float,
        soc_max: float,
        soc_bins: int = 10,
        hour_buckets: int = 24,
        tariff_periods: int = 3,
        peak_edges: list[float] | None = None,
    ):
        self.i = ObsReader(obs_names).i
        self.demand_edges = [float(e) for e in demand_edges]
        self.peak_edges = [float(e) for e in (peak_edges or [])]   # empty = peak feature off
        self.peak_bins = len(self.peak_edges) + 1
        self.soc_min, self.soc_max = soc_min, soc_max
        self.soc_bins, self.hour_buckets = soc_bins, hour_buckets
        self.demand_bins = len(self.demand_edges) + 1
        self.tariff_periods = tariff_periods
        self.n_states = (
            soc_bins * hour_buckets * self.demand_bins * self.peak_bins * tariff_periods
        )
        self._soc_width = (soc_max - soc_min) / soc_bins

    @classmethod
    def fit(cls, obs_names, train_demand: pd.Series, battery: BatteryParams, cfg: dict):
        """Learn demand quantile edges from TRAIN demand and build the discretizer."""
        n = cfg["demand_bins"]
        edges = np.quantile(train_demand.to_numpy(), np.linspace(0, 1, n + 1)[1:-1])
        # Optional: bins for "highest import so far this episode" (default off = plan's grid).
        peak_q = cfg.get("peak_quantiles", [])
        peak_edges = np.quantile(train_demand.to_numpy(), peak_q).tolist() if peak_q else []
        return cls(
            obs_names, edges.tolist(), battery.soc_min, battery.soc_max,
            cfg["soc_bins"], cfg["hour_buckets"], cfg["tariff_periods"], peak_edges,
        )

    def hour(self, obs) -> int:
        """Recover the hour of day from its sin/cos encoding."""
        angle = math.atan2(float(obs[self.i["hour_sin"]]), float(obs[self.i["hour_cos"]]))
        return round(angle * 12 / math.pi) % 24

    def __call__(self, obs) -> int:
        soc = float(obs[self.i["soc"]])
        soc_bin = min(max(int((soc - self.soc_min) / self._soc_width), 0), self.soc_bins - 1)
        hour_bin = self.hour(obs) * self.hour_buckets // 24
        demand_bin = bisect_right(self.demand_edges, float(obs[self.i["demand_kw"]]))
        period = int(obs[self.i["tariff_period"]])
        peak_bin = bisect_right(self.peak_edges, float(obs[self.i["peak_kw"]]))
        state = (soc_bin * self.hour_buckets + hour_bin) * self.demand_bins + demand_bin
        return (state * self.peak_bins + peak_bin) * self.tariff_periods + period

    def decode(self, state: int) -> tuple[int, int, int, int, int]:
        """Inverse of __call__: (soc_bin, hour_bin, demand_bin, peak_bin, period)."""
        state, period = divmod(state, self.tariff_periods)
        state, peak_bin = divmod(state, self.peak_bins)
        state, demand_bin = divmod(state, self.demand_bins)
        soc_bin, hour_bin = divmod(state, self.hour_buckets)
        return soc_bin, hour_bin, demand_bin, peak_bin, period


# --- Epsilon schedules: fraction of training at which epsilon reaches its floor ---
def epsilon_at(ep: int, episodes: int, start: float, end: float, schedule: str, fractions: dict):
    progress = min(ep / max(fractions[schedule] * episodes, 1), 1.0)
    if start <= 0 or end <= 0:       # geometric decay is undefined at zero; fall back to a line
        return start + (end - start) * progress
    if schedule == "linear":
        return start + (end - start) * progress
    # Every exponential variant is geometric interpolation; they differ in how fast it gets there.
    return start * (end / start) ** progress


# --- The agent ---
class QLearningAgent:
    name = "q_learning"

    def __init__(self, disc: Discretizer, n_actions: int, idle_action: int, params: dict):
        self.disc, self.n_actions, self.idle = disc, int(n_actions), int(idle_action)
        self.params = params
        self.Q = np.zeros((disc.n_states, n_actions))
        # Visit counts. Rewards here are mostly negative, so an untried action (Q = 0) would look
        # better than every tried one. Counts let us ignore untried actions when choosing.
        self.N = np.zeros((disc.n_states, n_actions), dtype=np.int64)

    def reset(self) -> None:
        pass

    def greedy(self, state: int) -> int:
        tried = self.N[state] > 0
        if not tried.any():              # never-visited state: do nothing rather than guess
            return self.idle
        return int(np.argmax(np.where(tried, self.Q[state], -np.inf)))

    def value(self, state: int) -> float:
        """Best Q among tried actions; 0 for a never-visited state."""
        tried = self.N[state] > 0
        return float(self.Q[state][tried].max()) if tried.any() else 0.0

    def act(self, obs) -> int:
        return self.greedy(self.disc(obs))

    def train(self, env, episodes: int, seed: int, eval_every: int = 0, eval_fn=None) -> dict:
        """Epsilon-greedy Q-learning on `env`. Returns reward, epsilon and eval histories."""
        p = self.params
        rng = np.random.default_rng(seed)
        history = {"episode_reward": [], "epsilon": [], "eval": []}
        t0 = time.perf_counter()
        for ep in range(episodes):
            eps = epsilon_at(ep, episodes, p["epsilon_start"], p["epsilon_end"],
                             p["epsilon_schedule"], p["schedule_fractions"])
            obs, _ = env.reset(seed=seed if ep == 0 else None)
            s, total, done = self.disc(obs), 0.0, False
            while not done:
                a = int(rng.integers(self.n_actions)) if rng.random() < eps else self.greedy(s)
                obs, r, _, done, _ = env.step(a)
                s2 = self.disc(obs)
                # Bootstrap even at the 7-day cutoff: the cutoff is a time limit, not a real end.
                target = r + p["gamma"] * self.value(s2)
                if self.N[s, a] == 0:
                    self.Q[s, a] = target         # first visit: no zero-initialization bias
                else:
                    # Sample-average early (1/visits), settling to the configured alpha.
                    rate = max(p["alpha"], 1.0 / (self.N[s, a] + 1))
                    self.Q[s, a] += rate * (target - self.Q[s, a])
                self.N[s, a] += 1
                s, total = s2, total + r
            history["episode_reward"].append(total)
            history["epsilon"].append(eps)
            if eval_fn and eval_every and (ep + 1) % eval_every == 0:
                history["eval"].append({"episode": ep + 1, "reward": float(eval_fn(self))})
        history["train_time_s"] = time.perf_counter() - t0
        return history

    # --- Save / load the Q-table together with the discretizer that defines its states ---
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "qtable.npy", self.Q)
        np.save(d / "visits.npy", self.N)
        meta = {
            "params": self.params, "n_actions": self.n_actions, "idle": self.idle,
            "demand_edges": self.disc.demand_edges, "soc_min": self.disc.soc_min,
            "soc_max": self.disc.soc_max, "soc_bins": self.disc.soc_bins,
            "hour_buckets": self.disc.hour_buckets, "tariff_periods": self.disc.tariff_periods,
            "peak_edges": self.disc.peak_edges,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, obs_names: list[str]) -> "QLearningAgent":
        d = Path(directory)
        m = json.loads((d / "meta.json").read_text())
        disc = Discretizer(obs_names, m["demand_edges"], m["soc_min"], m["soc_max"],
                           m["soc_bins"], m["hour_buckets"], m["tariff_periods"],
                           m.get("peak_edges", []))
        agent = cls(disc, m["n_actions"], m["idle"], m["params"])
        agent.Q = np.load(d / "qtable.npy")
        agent.N = np.load(d / "visits.npy")
        return agent
