"""Design optimisation: choose stage cuts and pressures for a product target.

Two tools:

stage_cut_for_spec
    One stage, one specification: the stage cut that gives a required purity
    (or recovery) of one component in the permeate or the retentate.  This is
    a scalar root-find and is the usual first step of a design.

optimise
    Any layout, several decision variables.  You write a small function that
    builds the flowsheet from named variables, for example

        def design(theta1, theta2, p_high):
            return two_stage_stripping(m, xf, F,
                                       StageSpec(theta1, p_high, 1, "cross_flow"),
                                       StageSpec(theta2, p_high, 1, "cross_flow"))

    and `optimise` finds the values (within bounds) that minimise the chosen
    objective - cost per tonne, area, power or specific energy - subject to
    minimum purity and recovery of a component in a product stream.

    Method: a quasi-random (Sobol) scan of the bounded design space, then a
    constrained local optimisation (SLSQP, then COBYLA if SLSQP fails) from the
    best feasible point of the scan.  Every evaluation is kept in `history`, so
    the scan doubles as a map of the trade-off.  The result is a local optimum;
    the scan makes a poor local optimum unlikely but cannot rule it out, so
    check the history (or widen the scan) for important designs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import qmc

from .core import InfeasibleError, Membrane, StageResult
from .energy import CostAssumptions, annual_cost
from .flowsheets import (FlowsheetResult, specific_cost_per_tonne,
                         specific_energy_kWh_per_tonne)
from .stagecut import solve_stage

OBJECTIVES = ("cost", "area", "power", "energy")


# ---------------------------------------------------------------------------
# one stage, one specification
# ---------------------------------------------------------------------------
def stage_cut_for_spec(membrane: Membrane, x_feed, component: str, target: float,
                       p_high: float, p_low: float, pattern: str = "countercurrent",
                       stream: str = "retentate", spec: str = "purity",
                       feed_flow: float = 1.0, theta_range=(1e-3, 0.98),
                       **stage_kw) -> StageResult:
    """Stage cut at which `component` reaches `target` purity or recovery.

    stream : 'retentate' or 'permeate'
    spec   : 'purity' (mole fraction) or 'recovery' (fraction of the feed's
             component leaving in `stream`)
    Extra keyword arguments go to solve_stage (e.g. sweep_flow).
    Raises InfeasibleError if the target is outside what the stage cut range
    can deliver (e.g. a permeate purity above the selectivity limit).
    """
    if spec not in ("purity", "recovery"):
        raise ValueError("spec must be 'purity' or 'recovery'")

    def value(th):
        r = solve_stage(membrane, x_feed, th, p_high, p_low, pattern, feed_flow, **stage_kw)
        return (r.purity(component, stream) if spec == "purity"
                else r.recovery(component, stream))

    lo, hi = theta_range
    # the reachable stage cut can be capped (co-current pressure-ratio limit)
    ths = np.linspace(lo, hi, 25)
    vals = []
    for th in ths:
        try:
            vals.append(value(th) - target)
        except InfeasibleError:
            vals.append(np.nan)
    vals = np.array(vals)
    ok = np.isfinite(vals)
    # refine the edge of a capped range (feasible -> infeasible) by bisection
    for k in range(len(ths) - 1):
        if ok[k] and not ok[k + 1]:
            a, b = ths[k], ths[k + 1]
            for _ in range(40):
                mid = 0.5 * (a + b)
                try:
                    value(mid)
                    a = mid
                except InfeasibleError:
                    b = mid
            ths = np.insert(ths, k + 1, a)
            vals = np.insert(vals, k + 1, value(a) - target)
            ok = np.isfinite(vals)
            break
    for k in range(len(ths) - 1):
        if ok[k] and ok[k + 1] and vals[k] * vals[k + 1] <= 0:
            th = brentq(lambda t: value(t) - target, ths[k], ths[k + 1], xtol=1e-10)
            return solve_stage(membrane, x_feed, th, p_high, p_low, pattern, feed_flow,
                               profile=False, **stage_kw)
    reach = (np.nanmin(vals + target), np.nanmax(vals + target)) if ok.any() else (np.nan,) * 2
    raise InfeasibleError(f"{component} {spec} of {target:g} in the {stream} is not reachable "
                          f"with this stage (range {reach[0]:.4g} to {reach[1]:.4g} for "
                          f"stage cuts {lo:g}-{hi:g})")


# ---------------------------------------------------------------------------
# multivariable optimisation of a flowsheet
# ---------------------------------------------------------------------------
@dataclass
class OptimisationResult:
    variables: dict                     # optimum decision variables
    result: FlowsheetResult | None      # flowsheet at the optimum
    objective: str
    value: float                        # objective at the optimum
    purity: float
    recovery: float
    feasible: bool
    message: str
    n_evaluations: int
    history: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        units = {"cost": "USD/t", "area": "m2", "power": "kW", "energy": "kWh/t"}
        lines = [f"optimum ({'feasible' if self.feasible else 'NOT feasible'}; "
                 f"{self.n_evaluations} flowsheet evaluations): {self.message}",
                 "  " + ", ".join(f"{k} = {v:.4g}" for k, v in self.variables.items()),
                 f"  {self.objective} = {self.value:.4g} {units.get(self.objective, '')}; "
                 f"purity {self.purity:.4f}, recovery {self.recovery:.4f}"]
        if self.result is not None:
            r = self.result
            lines.append(f"  total area {r.total_area:.4g} m2, total power "
                         f"{r.total_power_kW:.4g} kW, recycle/feed {r.recycle_ratio():.3f}")
        return "\n".join(lines)


def optimise(design: Callable[..., FlowsheetResult], bounds: Mapping[str, tuple],
             product: str, component: str, purity: float | None = None,
             recovery: float | None = None, objective: str | Callable = "cost",
             ca: CostAssumptions | None = None, n_scan: int | None = None,
             x0: Mapping[str, float] | None = None, seed: int = 0, local: bool = True,
             verbose: bool = False) -> OptimisationResult:
    """Minimise `objective` over the variables in `bounds`.

    design    : function(**variables) -> FlowsheetResult (single_stage,
                two_stage_*, three_stage or your own layout)
    bounds    : {'name': (low, high)} for each argument of `design`
    product   : product label ('slow-gas product', 'fast-gas product') or a
                stream name of the flowsheet
    component : component whose purity / recovery is constrained
    purity, recovery : minimum values (either or both; None = unconstrained)
    objective : 'cost' (USD per tonne of component in the product),
                'area' (m2), 'power' (kW), 'energy' (kWh per tonne), or a
                function(FlowsheetResult) -> float
    ca        : CostAssumptions for the 'cost' objective (the flowsheet is re-priced
                with these, whatever the design function used)
    n_scan    : number of scan points (default 16 per variable, at least 32);
                0 skips the scan and starts from x0
    x0        : optional starting design {'name': value}, added to the scan
                (e.g. the optimum of a similar case)
    """
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names], float)
    hi = np.array([bounds[k][1] for k in names], float)
    if np.any(hi <= lo):
        raise ValueError("each bound must be (low, high) with low < high")
    if purity is None and recovery is None:
        raise ValueError("give a purity and/or recovery target")
    if isinstance(objective, str) and objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES} or a function")
    ca = ca or CostAssumptions()
    d = len(names)
    rows, cache = [], {}

    def stream_of(res):
        return res.products.get(product, product)

    def obj_of(res):
        s = stream_of(res)
        if callable(objective):
            return float(objective(res))
        if objective == "cost":       # re-priced with this call's assumptions
            vac = sum(v for k, v in res.power_kW.items() if k.startswith("vacuum"))
            res.cost = annual_cost(res.total_area, res.total_power_kW - vac, vac, ca)
            return specific_cost_per_tonne(res, s, component, ca)
        if objective == "energy":
            return specific_energy_kWh_per_tonne(res, s, component)
        return res.total_area if objective == "area" else res.total_power_kW

    def evaluate(u):
        """u in [0, 1]^d -> (objective, constraint margins, result)."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        key = tuple(np.round(u, 12))
        if key in cache:
            return cache[key]
        v = dict(zip(names, lo + u * (hi - lo)))
        try:
            res = design(**v)
            s = stream_of(res)
            if s not in res.streams:
                raise KeyError(f"{product!r} is not a product label or stream of this layout")
            pur, rec = res.purity(s, component), res.recovery(s, component)
            f = obj_of(res)
            ok = res.converged and np.isfinite(f)
        except (InfeasibleError, RuntimeError, ValueError, FloatingPointError,
                ZeroDivisionError) as e:
            if verbose:
                print("infeasible point", v, e)
            res, pur, rec, f, ok = None, np.nan, np.nan, np.nan, False
        g = []
        if purity is not None:
            g.append((pur - purity) if ok else -1.0)
        if recovery is not None:
            g.append((rec - recovery) if ok else -1.0)
        g = np.array(g)
        out = (f if ok else np.inf, g, res)
        cache[key] = out
        rows.append({**v, "objective": f, "purity": pur, "recovery": rec,
                     "feasible": bool(ok and np.all(g >= -1e-9)), "solved": ok})
        return out

    # 1. scan
    if n_scan is None:
        n_scan = max(32, 16 * d)
    if n_scan == 0 and x0 is None:
        raise ValueError("n_scan=0 needs a starting point x0")
    U = (qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n_scan))))[:n_scan]
         if n_scan > 0 else np.empty((0, d)))
    if x0 is not None:
        u_start = (np.array([x0[k] for k in names], float) - lo) / (hi - lo)
        U = np.vstack([np.clip(u_start, 0, 1), U])
    for u in U:
        evaluate(u)
    hist = pd.DataFrame(rows)
    feas = hist[hist["feasible"]]
    if feas.empty:
        # nothing meets the targets: start from the point closest to them
        viol = []
        for u in U:
            f, g, _ = evaluate(u)
            viol.append(np.sum(np.minimum(g, 0) ** 2))
        u0 = U[int(np.argmin(viol))]
    else:
        best = feas["objective"].idxmin()
        u0 = (hist.loc[best, names].to_numpy(float) - lo) / (hi - lo)

    # 2. local refinement in scaled variables; objective normalised by the start
    f0 = evaluate(u0)[0]
    scale = abs(f0) if np.isfinite(f0) and f0 != 0 else 1.0
    message = "scan only"
    if local:
        fun = lambda u: (evaluate(u)[0] / scale if np.isfinite(evaluate(u)[0]) else 1e3)
        cons = [{"type": "ineq", "fun": (lambda u, k=k: evaluate(u)[1][k] * 100.0)}
                for k in range(len(evaluate(u0)[1]))]
        sol = minimize(fun, u0, method="SLSQP", bounds=[(0, 1)] * d, constraints=cons,
                       options={"maxiter": 100, "ftol": 1e-7, "eps": 1e-4})
        message = f"SLSQP: {sol.message}"
        f_s, g_s, _ = evaluate(sol.x)
        if not (sol.success and np.isfinite(f_s) and np.all(g_s >= -1e-6)):
            sol = minimize(fun, u0, method="COBYLA", constraints=cons
                           + [{"type": "ineq", "fun": (lambda u, k=k: u[k])} for k in range(d)]
                           + [{"type": "ineq", "fun": (lambda u, k=k: 1 - u[k])} for k in range(d)],
                           options={"maxiter": 400, "rhobeg": 0.1, "tol": 1e-6})
            message = f"COBYLA: {sol.message}"

    # 3. best feasible point seen anywhere (local solver or scan)
    hist = pd.DataFrame(rows)
    feas = hist[hist["feasible"]]
    if feas.empty:
        k = hist["objective"].replace(np.inf, np.nan).idxmin() if hist["solved"].any() else 0
        feasible = False
    else:
        k = feas["objective"].idxmin()
        feasible = True
    v = {n: float(hist.loc[k, n]) for n in names}
    f, g, res = evaluate((np.array([v[n] for n in names]) - lo) / (hi - lo))
    return OptimisationResult(v, res, objective if isinstance(objective, str) else "custom",
                              float(f), float(hist.loc[k, "purity"]),
                              float(hist.loc[k, "recovery"]), feasible, message,
                              len(hist), hist)
