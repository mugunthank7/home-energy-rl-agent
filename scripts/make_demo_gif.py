"""Make the demo GIF: a trained deep RL agent controlling the simulated battery for one held-out 2010 week.

Needs the deep RL extra and a trained model: uv sync --extra deeprl, then scripts/run_deep_rl.py.
Usage: uv run python scripts/make_demo_gif.py [--algo dqn|ppo] [--week winter|summer] [--seed 42]
Writes reports/figures/demo_<week>_<algo>.gif (so different runs never overwrite each other)
"""
import argparse
from pathlib import Path

from home_energy_rl_agent import data
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.controllers.no_battery import NoBattery
from home_energy_rl_agent.controllers.sb3_agents import SB3Controller, obs_scales
from home_energy_rl_agent.demo import make_gif, week_traces
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.tariff import load_tariff_config

WEEKS = {"winter": ("test_winter", "Jan 18-24, 2010"), "summer": ("test_summer", "Jul 5-11, 2010")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--week", default="winter", choices=list(WEEKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="default: reports/figures/demo_<week>_<algo>.gif")
    args = parser.parse_args()

    from stable_baselines3 import DQN, PPO

    splits = data.get_splits(data.load_hourly())
    key, dates = WEEKS[args.week]
    env = EnergyEnv(splits[key], load_env_config(), load_tariff_config(), seed=42)
    scales = obs_scales(env.obs_names, float(splits["train"].quantile(0.95)))   # same scales as training
    model = {"dqn": DQN, "ppo": PPO}[args.algo].load(
        data.ROOT / "results" / "deep_rl" / args.algo / f"seed_{args.seed}", device="cpu")
    label = args.algo.upper()
    ctrl = week_traces(SB3Controller(model, scales, args.algo), env, 0)
    base = week_traces(NoBattery(ActionMap.from_fractions(env.fractions)), env, 0)
    subtitle = (f"{label} controlling a SIMULATED 10 kWh battery on REAL household demand\n"
                f"held-out week {dates} (not seen in training)")
    out = Path(args.out) if args.out else data.ROOT / "reports" / "figures" / f"demo_{args.week}_{args.algo}.gif"
    path = make_gif(ctrl, base, label, subtitle, out)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB); final bill ${base['bill'][-1]:.2f} -> "
          f"${ctrl['bill'][-1]:.2f}, peak {base['peak'][-1]:.2f} -> {ctrl['peak'][-1]:.2f} kW")


if __name__ == "__main__":
    main()
