"""Visual sanity check: run one episode on REAL train demand with a simple inline policy.

The policy here is only for eyeballing the env (charge off-peak, discharge on-peak).
The real controllers arrive in Phase 3.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from home_energy_rl_agent import data
from home_energy_rl_agent.env import EnergyEnv
from home_energy_rl_agent.tariff import OFF_PEAK, ON_PEAK

CHARGE_HIGH, IDLE, DISCHARGE_HIGH = 4, 2, 0
OUT = data.ROOT / "reports" / "figures" / "env_check_episode.png"


def simple_policy(obs, names):
    """Charge when off-peak and SOC < 0.9, discharge when on-peak and SOC > 0.15."""
    period, soc = obs[names.index("tariff_period")], obs[names.index("soc")]
    if period == OFF_PEAK and soc < 0.9:
        return CHARGE_HIGH
    if period == ON_PEAK and soc > 0.15:
        return DISCHARGE_HIGH
    return IDLE


def main() -> None:
    # --- Build the env on the train split and run one fixed-seed episode ---
    train = data.get_splits(data.load_hourly())["train"]
    env = EnergyEnv(train, seed=42)
    obs, _ = env.reset(seed=42)
    infos, done = [], False
    while not done:
        obs, _, _, done, info = env.step(simple_policy(obs, env.obs_names))
        infos.append(info)
    df = pd.DataFrame(infos)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Plot demand vs grid import, SOC, and price ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(df.timestamp, df.demand, label="Demand (REAL)")
    axes[0].plot(df.timestamp, df.grid_import, label="Grid import (battery on)")
    axes[0].set_ylabel("kW")
    axes[0].legend()
    axes[1].plot(df.timestamp, df.soc, color="tab:green")
    axes[1].set_ylabel("Battery SOC (SIMULATED)")
    axes[2].step(df.timestamp, df.price, where="post", color="tab:red")
    axes[2].set_ylabel("Price $/kWh (SIMULATED)")
    fig.suptitle("Env check: one 7-day episode, simple charge-off-peak / discharge-on-peak policy")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")
    print({k: round(v, 3) for k, v in info["episode"].items()})


if __name__ == "__main__":
    main()
