"""Perfect-foresight oracle restricted to the environment's 5 power levels (a mixed-integer program).

Like the continuous oracle it sees the whole week of demand and prices, but each hour it must pick
one of the env's discrete actions (discharge high/low, idle, charge low/high). It only considers
plans a controller can follow with no limit violations, so it is the bound for such controllers.
The gap between this and the continuous oracle is what the 5-level action set costs; the gap between
this and a real controller is what learning (and missing foresight) costs.

Modelled exactly like the env's battery: a discharge request is cut to the current demand (no export), a
charge or discharge request larger than what the battery can take counts as a violation, so it is excluded.

Solved with HiGHS through scipy. The solver returns the best plan found (a feasible plan, so a valid
achievable cost) and a dual bound (a proven lower bound on the cost); the gap says how tight it is.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_discrete_episode(
    demand: np.ndarray, price: np.ndarray, levels_kw: np.ndarray, capacity_kwh: float, eta: float,
    soc_min: float, soc_max: float, soc_start: float, demand_charge: float,
    terminal_value: float = 0.0, time_limit: float = 30.0, rel_gap: float = 0.002,
) -> dict:
    """Best 5-level plan for one episode. levels_kw: signed request per action (negative = discharge)."""
    T, K = len(demand), len(levels_kw)
    lo, hi = soc_min * capacity_kwh, soc_max * capacity_kwh
    chg = [k for k in range(K) if levels_kw[k] > 0]
    dis = [k for k in range(K) if levels_kw[k] < 0]
    n_u, n_e = T * K, T + 1
    iu = lambda t, k: t * K + k
    ie = lambda t: n_u + t
    ipk = n_u + n_e                        # weekly peak
    n = ipk + 1
    # Grid import at hour t = d_t + sum_chg L u - sum_dis min(L, d_t) u   (discharge is capped at demand)
    deliver = np.array([[min(-levels_kw[k], demand[t]) if levels_kw[k] < 0 else 0.0 for k in range(K)]
                        for t in range(T)])
    add = np.where(levels_kw > 0, levels_kw, 0.0)

    rows = 4 * T
    A, lb, ub = lil_matrix((rows, n)), np.zeros(rows), np.zeros(rows)
    r = 0
    for t in range(T):
        for k in range(K):                                           # exactly one action per hour
            A[r, iu(t, k)] = 1
        lb[r] = ub[r] = 1
        r += 1
        A[r, ie(t + 1)], A[r, ie(t)] = 1, -1                        # energy balance
        for k in range(K):
            A[r, iu(t, k)] = -eta * add[k] + deliver[t, k] / eta
        lb[r] = ub[r] = 0
        r += 1
        A[r, ie(t)] = 1                                              # discharge request must be coverable
        for k in dis:
            A[r, iu(t, k)] = levels_kw[k] / eta          # levels are negative for discharge
        lb[r], ub[r] = lo, np.inf
        r += 1
        A[r, ie(t)] = -1                                             # charge request must fit
        for k in chg:
            A[r, iu(t, k)] = -eta * levels_kw[k]
        lb[r], ub[r] = -hi, np.inf
        r += 1
    peak_rows = lil_matrix((T, n))
    for t in range(T):                                               # peak >= grid import
        peak_rows[t, ipk] = 1
        for k in range(K):
            peak_rows[t, iu(t, k)] = -(add[k] - deliver[t, k])
    peak_lb, peak_ub = demand.copy(), np.full(T, np.inf)

    c = np.zeros(n)
    for t in range(T):
        for k in range(K):
            c[iu(t, k)] = price[t] * (add[k] - deliver[t, k])
    c[ipk] = demand_charge
    c[ie(T)] -= terminal_value                                       # optional: pay for energy used up
    const = float(price @ demand) + terminal_value * soc_start * capacity_kwh

    integrality = np.zeros(n)
    integrality[:n_u] = 1
    lower = np.concatenate([np.zeros(n_u), np.full(n_e, lo), [0.0]])
    upper = np.concatenate([np.ones(n_u), np.full(n_e, hi), [np.inf]])
    lower[ie(0)] = upper[ie(0)] = soc_start * capacity_kwh
    res = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=[LinearConstraint(A.tocsr(), lb, ub),
                            LinearConstraint(peak_rows.tocsr(), peak_lb, peak_ub)],
               options={"time_limit": time_limit, "mip_rel_gap": rel_gap, "disp": False})
    if res.x is None:
        raise RuntimeError(f"discrete oracle found no plan: {res.message}")
    actions = res.x[:n_u].reshape(T, K).argmax(1)
    return {"actions": actions.astype(int), "objective": float(res.fun) + const,
            "dual_bound": float(res.mip_dual_bound) + const, "gap": float(res.mip_gap),
            "optimal": bool(res.status == 0)}


# --- Evaluation: solve every week in parallel, then replay the plans through the real env ---
class PlanController:
    """Replays a precomputed action sequence for each episode start (used to score the plans in the env)."""

    def __init__(self, env, plans: dict, name: str = "oracle_discrete"):
        self.env, self.plans, self.name, self.i, self.actions = env, plans, name, 0, None

    def reset(self) -> None:
        self.actions, self.i = self.plans[self.env.start]["actions"], 0

    def act(self, obs) -> int:
        a = int(self.actions[self.i])
        self.i += 1
        return a


def _solve_job(args: tuple) -> tuple[int, dict]:
    start, kwargs = args
    return start, solve_discrete_episode(**kwargs)


def compute_plans(env, starts, workers: int = 8, value_terminal: bool = False,
                  time_limit: float = 30.0) -> dict:
    """Best 5-level plan for each episode start, solved in parallel. Returns {start: result}."""
    from concurrent.futures import ProcessPoolExecutor

    b = env.battery.p
    jobs = []
    for s in starts:
        sl = slice(int(s), int(s) + env.steps_per_episode)
        jobs.append((int(s), {
            "demand": env._demand_all[sl].copy(), "price": env._price_all[sl].copy(),
            "levels_kw": env.fractions * b.max_power_kw, "capacity_kwh": b.capacity_kwh,
            "eta": b.eta_one_way, "soc_min": b.soc_min, "soc_max": b.soc_max, "soc_start": b.soc_start,
            "demand_charge": env.tariff.demand_charge_per_kw,
            "terminal_value": env.terminal_value_per_kwh if value_terminal else 0.0,
            "time_limit": time_limit}))
    if workers <= 1:
        return dict(_solve_job(j) for j in jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_solve_job, jobs))


def evaluate_discrete_oracle(env, starts, seed: int = 42, workers: int = 8,
                             value_terminal: bool = False, time_limit: float = 30.0):
    """Scores of the best 5-level plans, in evaluate()'s format, plus the solver's proven bound."""
    from home_energy_rl_agent.evaluate import evaluate

    name = "oracle_discrete_adj" if value_terminal else "oracle_discrete"
    plans = compute_plans(env, starts, workers, value_terminal, time_limit)
    result = evaluate(PlanController(env, plans, name), env, starts=starts, seed=seed)
    # Proven lower bound on cost (an upper bound on savings for any violation-free 5-level controller).
    dual = np.array([plans[int(s)]["dual_bound"] for s in starts])
    key = "total_cost_adj" if value_terminal else "total_cost"
    base = result.per_episode["base_total_cost"].to_numpy()
    sav_key = "savings_total_adj_pct" if value_terminal else "savings_total_pct"
    result.summary["dual_bound_savings_pct"] = float(100 * (1 - dual.mean() / base.mean()))
    result.summary["bound_gap_mean"] = float(np.mean([plans[int(s)]["gap"] for s in starts]))
    result.summary["solved_to_tolerance_share"] = float(np.mean([plans[int(s)]["optimal"] for s in starts]))
    assert result.per_episode[key].mean() >= dual.mean() - 1e-6 or sav_key
    return result
