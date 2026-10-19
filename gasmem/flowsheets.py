"""Model 4b - multistage membrane flowsheets with recycle, compression energy
and a rough cost.

Each stage is solved in design mode (stage cut specified) with any flow
pattern.  Recycle streams are converged by successive substitution on the
recycled component flows (tear streams).

Configurations
--------------
single_stage        feed -> compressor -> stage 1
two_stage_enriching permeate 1 is recompressed and purified in stage 2;
                    retentate 2 is recycled to the feed of stage 1.
                    Products: permeate 2 (fast gas), retentate 1 (slow gas).
two_stage_stripping retentate 1 is polished in stage 2; permeate 2 is
                    recompressed and recycled.
                    Products: permeate 1 (fast gas), retentate 2 (slow gas).
three_stage         stage 1 retentate -> stage 2 (slow-gas product);
                    stage 1 permeate -> stage 3 at the stage-1 permeate pressure
                    (fast-gas product); permeate 2 and retentate 3 are recycled
                    to the compressor.  This is the layout widely used for
                    biogas upgrading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import Membrane, StageResult
from .energy import CostAssumptions, EnergyAssumptions, annual_cost, compression_power
from .properties import ATM_BAR, gas, mixture_k
from .stagecut import solve_stage


@dataclass
class StageSpec:
    """Operating specification of one stage (pressures in bar)."""
    theta: float
    p_high: float
    p_low: float
    pattern: str = "countercurrent"


@dataclass
class FlowsheetResult:
    name: str
    components: tuple[str, ...]
    fresh_feed: np.ndarray                      # component flows, mol/s
    stages: dict[str, StageResult]
    streams: dict[str, np.ndarray]              # component flows, mol/s
    products: dict[str, str]                    # label -> stream name
    power_kW: dict[str, float]
    iterations: int
    converged: bool
    cost: dict = field(default_factory=dict)

    # -- metrics ------------------------------------------------------------
    @property
    def total_area(self) -> float:
        return float(sum(s.area for s in self.stages.values()))

    @property
    def total_power_kW(self) -> float:
        return float(sum(self.power_kW.values()))

    def purity(self, stream: str, comp: str) -> float:
        v = self.streams[stream]
        return float(v[self.components.index(comp)] / v.sum())

    def recovery(self, stream: str, comp: str) -> float:
        i = self.components.index(comp)
        return float(self.streams[stream][i] / self.fresh_feed[i])

    def recycle_ratio(self) -> float:
        rec = sum(v.sum() for k, v in self.streams.items() if k.startswith("recycle"))
        return float(rec / self.fresh_feed.sum())

    def mass_balance_error(self) -> float:
        out = sum(self.streams[s] for s in self.products.values())
        return float(np.max(np.abs(out - self.fresh_feed)) / self.fresh_feed.sum())

    def stream_table(self) -> pd.DataFrame:
        rows = []
        for k, v in self.streams.items():
            row = {"stream": k, "flow_mol_s": v.sum()}
            row.update({f"y_{c}": f for c, f in zip(self.components, v / v.sum())})
            rows.append(row)
        return pd.DataFrame(rows).set_index("stream")

    def summary(self) -> str:
        lines = [f"{self.name}: {'converged' if self.converged else 'NOT converged'} "
                 f"in {self.iterations} iterations; recycle/feed = {self.recycle_ratio():.3f}",
                 f"  total area {self.total_area:.4g} m2, total power {self.total_power_kW:.4g} kW"]
        for label, s in self.products.items():
            v = self.streams[s]
            comp = ", ".join(f"{c} {f:.4f}" for c, f in zip(self.components, v / v.sum()))
            lines.append(f"  {label} ({s}): {v.sum():.4g} mol/s; {comp}")
        for k, st in self.stages.items():
            lines.append(f"  {k}: {st.pattern}, theta {st.theta:.3f}, area {st.area:.4g} m2, "
                         f"{st.p_high:g}/{st.p_low:g} bar")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _run_stage(membrane: Membrane, flows: np.ndarray, spec: StageSpec, cache=None, key=None):
    """Solve one stage; `cache` stores the last retentate composition per stage
    so recycle iterations warm-start the counter-current shooting."""
    F = flows.sum()
    guess = cache.get(key) if cache is not None else None
    r = solve_stage(membrane, flows / F, spec.theta, spec.p_high, spec.p_low, spec.pattern, F,
                    x_ret_guess=guess)
    if cache is not None:
        cache[key] = r.x_ret
    perm = r.permeate_flow * r.y_perm
    ret = r.retentate_flow * r.x_ret
    return r, perm, ret


def _power(names, flows, p_in, p_out, ea: EnergyAssumptions, eta=None) -> float:
    if flows.sum() <= 0:
        return 0.0
    return compression_power(flows.sum(), p_in, p_out, mixture_k(names, flows / flows.sum()),
                             ea.T, eta or ea.eta_compressor, ea.max_stage_ratio) / 1e3


def _vacuum(names, flows, p_low, ea) -> float:
    return _power(names, flows, p_low, ea.p_discharge, ea, ea.eta_vacuum) if p_low < ea.p_discharge else 0.0


def _iterate(step, n, F, tol=1e-9, maxit=300, n_tear=1):
    """Converge the recycle (tear) vectors by successive substitution
    accelerated with the bounded Wegstein method."""
    x = np.zeros(n * n_tear)
    split = lambda v: [v[k * n:(k + 1) * n] for k in range(n_tear)]
    x_prev = g_prev = None
    for it in range(1, maxit + 1):
        out = step(split(x))
        g = np.concatenate(out["tears"])
        err = np.max(np.abs(g - x)) / F
        if err < tol:
            return out, it, True
        if x_prev is None:
            x_new = g
        else:
            dx = x - x_prev
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = np.where(np.abs(dx) > 1e-14 * F, (g - g_prev) / dx, 0.0)
                q = np.clip(slope / (slope - 1.0), -5.0, 0.0)
            q = np.nan_to_num(q)
            x_new = q * x + (1.0 - q) * g
        x_prev, g_prev = x, g
        x = np.clip(x_new, 0.0, None)
    return step(split(x)), maxit, False


def _finish(name, membrane, fresh, out, it, ok, products, ca):
    res = FlowsheetResult(name, membrane.components, fresh, out["stages"], out["streams"],
                          products, out["power"], it, ok)
    vac = sum(v for k, v in res.power_kW.items() if k.startswith("vacuum"))
    res.cost = annual_cost(res.total_area, res.total_power_kW - vac, vac, ca)
    return res


def single_stage(membrane: Membrane, x_feed, feed_flow: float, s1: StageSpec,
                 ea: EnergyAssumptions | None = None,
                 ca: CostAssumptions | None = None) -> FlowsheetResult:
    ea, ca = ea or EnergyAssumptions(), ca or CostAssumptions()
    names = membrane.components
    fresh = membrane.vector(x_feed) * feed_flow
    r, perm, ret = _run_stage(membrane, fresh, s1)
    out = {"stages": {"stage 1": r},
           "streams": {"feed": fresh, "permeate 1": perm, "retentate 1": ret},
           "power": {"feed compressor": _power(names, fresh, ea.p_feed_source, s1.p_high, ea),
                     "vacuum 1": _vacuum(names, perm, s1.p_low, ea)}}
    return _finish("single stage", membrane, fresh, out, 1, True,
                   {"fast-gas product": "permeate 1", "slow-gas product": "retentate 1"}, ca)


def two_stage_enriching(membrane: Membrane, x_feed, feed_flow: float, s1: StageSpec,
                        s2: StageSpec, ea: EnergyAssumptions | None = None,
                        ca: CostAssumptions | None = None, **kw) -> FlowsheetResult:
    """Permeate of stage 1 recompressed to stage 2; retentate 2 recycled."""
    ea, ca = ea or EnergyAssumptions(), ca or CostAssumptions()
    names, n = membrane.components, membrane.n
    fresh = membrane.vector(x_feed) * feed_flow

    cache = {}

    def step(tears):
        rec = tears[0]
        r1, p1, q1 = _run_stage(membrane, fresh + rec, s1, cache, "1")
        r2, p2, q2 = _run_stage(membrane, p1, s2, cache, "2")
        power = {"feed compressor": _power(names, fresh, ea.p_feed_source, s1.p_high, ea),
                 "recycle compressor": _power(names, q2, s2.p_high, s1.p_high, ea),
                 "interstage compressor": _power(names, p1, s1.p_low, s2.p_high, ea),
                 "vacuum 2": _vacuum(names, p2, s2.p_low, ea)}
        return {"tears": [q2], "stages": {"stage 1": r1, "stage 2": r2}, "power": power,
                "streams": {"feed": fresh, "stage 1 feed": fresh + rec, "permeate 1": p1,
                            "retentate 1": q1, "permeate 2": p2, "recycle (retentate 2)": q2}}

    out, it, ok = _iterate(step, n, feed_flow, **kw)
    return _finish("two-stage enriching", membrane, fresh, out, it, ok,
                   {"fast-gas product": "permeate 2", "slow-gas product": "retentate 1"}, ca)


def two_stage_stripping(membrane: Membrane, x_feed, feed_flow: float, s1: StageSpec,
                        s2: StageSpec, ea: EnergyAssumptions | None = None,
                        ca: CostAssumptions | None = None, **kw) -> FlowsheetResult:
    """Retentate of stage 1 polished in stage 2; permeate 2 recycled."""
    ea, ca = ea or EnergyAssumptions(), ca or CostAssumptions()
    names, n = membrane.components, membrane.n
    fresh = membrane.vector(x_feed) * feed_flow
    if s2.p_high > s1.p_high:
        raise ValueError("stage 2 feed pressure cannot exceed stage 1 (no booster modelled)")

    cache = {}

    def step(tears):
        rec = tears[0]
        r1, p1, q1 = _run_stage(membrane, fresh + rec, s1, cache, "1")
        r2, p2, q2 = _run_stage(membrane, q1, s2, cache, "2")
        power = {"feed compressor": _power(names, fresh, ea.p_feed_source, s1.p_high, ea),
                 "recycle compressor": _power(names, p2, s2.p_low, s1.p_high, ea),
                 "vacuum 1": _vacuum(names, p1, s1.p_low, ea)}
        return {"tears": [p2], "stages": {"stage 1": r1, "stage 2": r2}, "power": power,
                "streams": {"feed": fresh, "stage 1 feed": fresh + rec, "permeate 1": p1,
                            "retentate 1": q1, "retentate 2": q2, "recycle (permeate 2)": p2}}

    out, it, ok = _iterate(step, n, feed_flow, **kw)
    return _finish("two-stage stripping", membrane, fresh, out, it, ok,
                   {"fast-gas product": "permeate 1", "slow-gas product": "retentate 2"}, ca)


def three_stage(membrane: Membrane, x_feed, feed_flow: float, s1: StageSpec, s2: StageSpec,
                s3: StageSpec, ea: EnergyAssumptions | None = None,
                ca: CostAssumptions | None = None, **kw) -> FlowsheetResult:
    """Stage 3 is fed by permeate 1 at the stage-1 permeate pressure, so
    s3.p_high is taken equal to s1.p_low (set s1.p_low above 1 bar to give it
    a driving force)."""
    ea, ca = ea or EnergyAssumptions(), ca or CostAssumptions()
    names, n = membrane.components, membrane.n
    fresh = membrane.vector(x_feed) * feed_flow
    s3 = StageSpec(s3.theta, s1.p_low, s3.p_low, s3.pattern)
    if s3.p_low >= s3.p_high:
        raise ValueError("stage 3 needs s1.p_low > s3.p_low")
    if s2.p_high > s1.p_high:
        raise ValueError("stage 2 feed pressure cannot exceed stage 1")

    cache = {}

    def step(tears):
        rec2, rec3 = tears
        f1 = fresh + rec2 + rec3
        r1, p1, q1 = _run_stage(membrane, f1, s1, cache, "1")
        r2, p2, q2 = _run_stage(membrane, q1, s2, cache, "2")
        r3, p3, q3 = _run_stage(membrane, p1, s3, cache, "3")
        power = {"feed compressor": _power(names, fresh, ea.p_feed_source, s1.p_high, ea),
                 "recycle compressor (permeate 2)": _power(names, p2, s2.p_low, s1.p_high, ea),
                 "recycle compressor (retentate 3)": _power(names, q3, s3.p_high, s1.p_high, ea),
                 "vacuum 3": _vacuum(names, p3, s3.p_low, ea)}
        return {"tears": [p2, q3], "stages": {"stage 1": r1, "stage 2": r2, "stage 3": r3},
                "power": power,
                "streams": {"feed": fresh, "stage 1 feed": f1, "permeate 1": p1,
                            "retentate 1": q1, "retentate 2": q2, "permeate 3": p3,
                            "recycle (permeate 2)": p2, "recycle (retentate 3)": q3}}

    out, it, ok = _iterate(step, n, feed_flow, n_tear=2, **kw)
    return _finish("three-stage", membrane, fresh, out, it, ok,
                   {"fast-gas product": "permeate 3", "slow-gas product": "retentate 2"}, ca)


def specific_cost_per_tonne(res: FlowsheetResult, stream: str, comp: str,
                            ca: CostAssumptions | None = None) -> float:
    """Annualised cost (USD) per tonne of `comp` leaving in `stream`."""
    ca = ca or CostAssumptions()
    i = res.components.index(comp)
    t_per_yr = res.streams[stream][i] * gas(comp)["M"] * 3600 * ca.hours_per_year / 1000
    return res.cost["total_USD_per_yr"] / t_per_yr


def specific_energy_kWh_per_tonne(res: FlowsheetResult, stream: str, comp: str) -> float:
    i = res.components.index(comp)
    t_per_h = res.streams[stream][i] * gas(comp)["M"] * 3.6
    return res.total_power_kW / t_per_h
