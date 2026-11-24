"""Optimal-control oracle: a linear program with perfect foresight over one episode.

It sees the whole week of demand and prices in advance and picks continuous charge/discharge power
to minimize the real bill (energy cost plus the demand charge on the weekly peak), under the same
battery limits as the environment. Any controller that uses the env's 5 discrete power levels is one
feasible plan of this problem, so the oracle's cost is a lower bound for all of them. It is not a
deployable controller, because it needs the future.

Needs the optional dependency cvxpy (`uv sync --extra oracle`).
"""
import numpy as np
import pandas as pd

from home_energy_rl_agent.battery import BatteryParams
from home_energy_rl_agent.evaluate import (
    EPISODE_METRICS,
    EvalResult,
    adjusted_total_cost,
    summarize,
)

TIE_BREAK = 1e-6   # $/kWh of throughput: stops pointless simultaneous charge+discharge on flat prices


class OracleLP:
    """Builds the LP once (parameterized by demand, prices, start energy) and re-solves it per episode."""

    def __init__(self, battery: BatteryParams, steps: int, demand_charge_per_kw: float,
                 terminal_value: float = 0.0):
        import cvxpy as cp

        self.cp, self.b, self.T, self.dc = cp, battery, steps, demand_charge_per_kw
        # terminal_value > 0 also charges the LP for stored energy it uses up (see evaluate.adjusted_total_cost)
        eta, cap = battery.eta_one_way, battery.capacity_kwh
        self.demand, self.price = cp.Parameter(steps, nonneg=True), cp.Parameter(steps, nonneg=True)
        self.base_energy, self.e0 = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        c, g = cp.Variable(steps, nonneg=True), cp.Variable(steps, nonneg=True)   # AC-side kW
        e, peak = cp.Variable(steps + 1), cp.Variable(nonneg=True)                 # stored kWh, weekly peak
        grid = self.demand + c - g
        constraints = [
            e[0] == self.e0,
            e[1:] == e[:-1] + eta * c - g / eta,
            e >= battery.soc_min * cap, e <= battery.soc_max * cap,
            c <= battery.max_power_kw, g <= battery.max_power_kw,
            g <= self.demand,              # no export
            grid <= peak,
        ]
        objective = self.base_energy + self.price @ (c - g) + demand_charge_per_kw * peak \
            + TIE_BREAK * cp.sum(c + g) + terminal_value * (self.e0 - e[steps])
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self.vars = (c, g, e)

    def solve(self, demand: np.ndarray, price: np.ndarray, soc_start: float | None = None) -> dict:
        """Optimal plan for one episode. Returns arrays plus the resulting costs."""
        frac = self.b.soc_start if soc_start is None else soc_start
        self.demand.value, self.price.value = np.maximum(demand, 0), price
        self.base_energy.value = float(np.dot(price, np.maximum(demand, 0)))
        self.e0.value = frac * self.b.capacity_kwh
        self.problem.solve()
        if self.problem.status != "optimal":
            raise RuntimeError(f"oracle LP not solved: {self.problem.status}")
        c, g, e = (np.clip(v.value, 0, None) if i < 2 else v.value for i, v in enumerate(self.vars))
        grid = np.maximum(demand + c - g, 0)
        return {"charge_kw": c, "discharge_kw": g, "energy_kwh": e, "grid_import": grid,
                "soc": e[1:] / self.b.capacity_kwh,
                "energy_cost": float(np.dot(price, grid)),
                "peak_kw": float(grid.max())}


def oracle_episode_metrics(plan: dict, env, start: int) -> dict:
    """Costs, cycles and reward of an oracle plan, with the same definitions as the env's summary."""
    sl = slice(start, start + env.steps_per_episode)
    demand, price = env._demand_all[sl], env._price_all[sl]
    throughput = float(plan["charge_kw"].sum() + plan["discharge_kw"].sum())
    demand_cost = env.tariff.demand_charge_per_kw * plan["peak_kw"]
    # Reward with the env's own weights; the oracle never violates a limit.
    peak_term = plan["peak_kw"] - (demand.max() if env.peak_term == "excess" else 0.0)
    reward = -plan["energy_cost"] - env.w_peak * peak_term - env.w_cycle * throughput
    return {"reward": float(reward), "energy_cost": plan["energy_cost"],
            "demand_charge_cost": float(demand_cost), "total_cost": plan["energy_cost"] + float(demand_cost),
            "peak_kw": plan["peak_kw"], "violations": 0.0,
            "equivalent_full_cycles": throughput / (2 * env.battery.p.capacity_kwh),
            "final_soc": float(plan["soc"][-1]),
            "total_cost_adj": float(adjusted_total_cost(plan["energy_cost"] + demand_cost,
                                                        float(plan["soc"][-1]), env)),
            "base_energy_cost": float(np.dot(price, demand)),
            "base_total_cost": float(np.dot(price, demand) + env.tariff.demand_charge_per_kw * demand.max()),
            "base_peak_kw": float(demand.max())}


def evaluate_oracle(env, starts: np.ndarray, seed: int = 42, value_terminal: bool = False) -> EvalResult:
    """Oracle scores on the given episode starts, in the same format as evaluate().

    value_terminal=False minimizes the plain bill (the bound for the plain metric). True also charges
    for stored energy used up, which is the bound for the terminal-adjusted metric.
    """
    name = "oracle_adj" if value_terminal else "oracle"
    lp = OracleLP(env.battery.p, env.steps_per_episode, env.tariff.demand_charge_per_kw,
                  env.terminal_value_per_kwh if value_terminal else 0.0)
    rows = []
    for s in starts:
        sl = slice(int(s), int(s) + env.steps_per_episode)
        plan = lp.solve(env._demand_all[sl], env._price_all[sl])
        row = oracle_episode_metrics(plan, env, int(s))
        row["start_time"] = str(env.series.index[int(s)])
        rows.append(row)
    df = pd.DataFrame(rows)
    assert set(EPISODE_METRICS) <= set(df.columns)
    return EvalResult(name, df, summarize(df, name, len(starts), seed))
