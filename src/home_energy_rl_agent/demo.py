"""Animated demo of one week: a controller on REAL demand with the SIMULATED battery and tariff.

The GIF replays a rollout hour by hour next to the no-battery case and shows the running bill and weekly
peak, so the effect of the controller is visible without any app.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from home_energy_rl_agent.evaluate import rollout


def week_traces(controller, env, start: int = 0) -> dict:
    """Per-hour arrays for one episode, including the running bill (energy + demand charge on the peak so far)."""
    _, log = rollout(controller, env, start, keep_log=True)
    grid = np.array([r["grid_import"] for r in log])
    cost = np.array([r["cost"] for r in log])
    peak = np.maximum.accumulate(grid)
    return {
        "time": np.array([r["timestamp"] for r in log]),
        "demand": np.array([r["demand"] for r in log]),
        "grid": grid,
        "soc": np.array([r["soc"] for r in log]),
        "price": np.array([r["price"] for r in log]),
        "peak": peak,
        "bill": np.cumsum(cost) + env.tariff.demand_charge_per_kw * peak,
    }


def make_gif(ctrl: dict, base: dict, label: str, subtitle: str, path: Path, step: int = 3, fps: int = 10,
             hold_frames: int = 12, dpi: int = 72) -> Path:
    """Animate `ctrl` against `base` (the no-battery traces of the same week) and save a GIF."""
    n = len(ctrl["demand"])
    hours = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1]})
    ax_p, ax_s, ax_c = axes
    ax_p.set_ylim(0, max(ctrl["demand"].max(), ctrl["grid"].max()) * 1.12)
    ax_p.set_xlim(0, n - 1)
    ax_p.set_ylabel("kW")
    ax_p.plot(hours, base["demand"], color="#bbbbbb", lw=1, label="Demand (REAL) = grid import with no battery")
    (line_g,) = ax_p.plot([], [], color="#1f78b4", lw=1.8, label=f"Grid import with {label}")
    ax_p.legend(loc="upper right", fontsize=8, frameon=False)
    ax_s.set_ylim(0, 1.02)
    ax_s.set_ylabel("Battery charge\n(SIMULATED)", fontsize=8)
    (line_s,) = ax_s.plot([], [], color="#33a02c", lw=1.8)
    ax_c.set_ylim(0, ctrl["price"].max() * 1.25)
    ax_c.set_ylabel("Price $/kWh\n(SIMULATED)", fontsize=8)
    ax_c.step(hours, ctrl["price"], where="post", color="#e31a1c", lw=1.2)
    ax_c.set_xticks(np.arange(0, n, 24), [f"day {d + 1}" for d in range(n // 24 + 1)][: len(np.arange(0, n, 24))])
    fig.suptitle(subtitle, fontsize=10, y=0.995)
    header = fig.text(0.5, 0.885, "", ha="center", fontsize=9, family="monospace")
    cursor = [ax.axvline(0, color="k", lw=0.6, alpha=0.5) for ax in axes]
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    frames = list(range(step - 1, n, step))
    frames += [frames[-1]] * hold_frames          # pause on the final state

    def draw(k: int):
        sl = slice(0, k + 1)
        line_g.set_data(hours[sl], ctrl["grid"][sl])
        line_s.set_data(hours[sl], ctrl["soc"][sl])
        for c in cursor:
            c.set_xdata([k, k])
        # "\\$" because matplotlib treats text between two plain dollar signs as math.
        header.set_text(f"{str(ctrl['time'][k])[:16]}   bill so far: \\${base['bill'][k]:.2f} no battery vs "
                        f"\\${ctrl['bill'][k]:.2f} {label}   peak: {base['peak'][k]:.2f} vs {ctrl['peak'][k]:.2f} kW")
        return line_g, line_s, header

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
