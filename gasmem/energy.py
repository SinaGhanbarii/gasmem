"""Compression / vacuum energy and a rough, transparent cost estimate.

Energy
------
Ideal-gas compression in equal-ratio stages with intercooling back to the
inlet temperature (adiabatic stages, isentropic efficiency eta):

    W = n_stages * n R T k/(k-1) / eta * [r_stage^((k-1)/k) - 1]

Vacuum pumps on the permeate side use the same expression from p_perm up to
the discharge pressure (default 1 atm) with their own efficiency.

Cost
----
`CostAssumptions` collects every number used.  The defaults are *illustrative
placeholders* meant to show the shape of the trade-off with stage cut, not to
price a real plant.  Replace them with your own data before quoting anything.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .core import StageResult
from .properties import ATM_BAR, R, T_REF, mixture_k


def compression_power(n_dot: float, p_in: float, p_out: float, k: float = 1.4,
                      T: float = T_REF, eta: float = 0.75, max_ratio: float = 4.0) -> float:
    """Shaft power (W) to compress n_dot mol/s from p_in to p_out (bar)."""
    if n_dot <= 0 or p_out <= p_in:
        return 0.0
    ratio = p_out / p_in
    stages = max(1, int(np.ceil(np.log(ratio) / np.log(max_ratio) - 1e-12)))
    r = ratio ** (1.0 / stages)
    return float(stages * n_dot * R * T * k / (k - 1.0) / eta * (r ** ((k - 1.0) / k) - 1.0))


@dataclass
class EnergyAssumptions:
    p_feed_source: float = ATM_BAR    # pressure at which the feed is available, bar
    p_discharge: float = ATM_BAR      # vacuum-pump discharge, bar
    T: float = T_REF
    eta_compressor: float = 0.75
    eta_vacuum: float = 0.60
    max_stage_ratio: float = 4.0


def stage_energy(res: StageResult, ea: EnergyAssumptions | None = None) -> dict:
    """Feed compressor + permeate vacuum pump power (kW) for one stage."""
    ea = ea or EnergyAssumptions()
    names = res.components
    w_feed = compression_power(res.feed_flow, ea.p_feed_source, res.p_high,
                               mixture_k(names, res.x_feed), ea.T, ea.eta_compressor,
                               ea.max_stage_ratio)
    w_vac = compression_power(res.permeate_flow, res.p_low, ea.p_discharge,
                              mixture_k(names, res.y_perm), ea.T, ea.eta_vacuum,
                              ea.max_stage_ratio) if res.p_low < ea.p_discharge else 0.0
    return {"feed_compressor_kW": w_feed / 1e3, "vacuum_pump_kW": w_vac / 1e3,
            "total_kW": (w_feed + w_vac) / 1e3}


@dataclass
class CostAssumptions:
    """Illustrative economic inputs (USD).  Replace with your own values."""
    membrane_cost_per_m2: float = 50.0       # module replacement cost
    membrane_life_years: float = 5.0
    skid_cost_per_m2: float = 250.0          # housings, piping, frames (capital)
    compressor_cost_per_kW: float = 1500.0   # installed (capital), per kW shaft
    vacuum_cost_per_kW: float = 1500.0
    interest_rate: float = 0.08
    plant_life_years: float = 20.0
    maintenance_fraction: float = 0.03       # of capital per year
    electricity_per_kWh: float = 0.10
    hours_per_year: float = 8000.0

    def crf(self) -> float:
        i, n = self.interest_rate, self.plant_life_years
        return i * (1 + i) ** n / ((1 + i) ** n - 1) if i > 0 else 1.0 / n


def annual_cost(area_m2: float, compressor_kW: float, vacuum_kW: float,
                ca: CostAssumptions | None = None) -> dict:
    """Annualised cost breakdown (USD/year)."""
    ca = ca or CostAssumptions()
    capex = (area_m2 * ca.skid_cost_per_m2 + compressor_kW * ca.compressor_cost_per_kW
             + vacuum_kW * ca.vacuum_cost_per_kW)
    capital = capex * ca.crf()
    membrane = area_m2 * ca.membrane_cost_per_m2 / ca.membrane_life_years
    maintenance = capex * ca.maintenance_fraction
    power = (compressor_kW + vacuum_kW) * ca.hours_per_year * ca.electricity_per_kWh
    total = capital + membrane + maintenance + power
    return {"capex_USD": capex, "capital_USD_per_yr": capital,
            "membrane_replacement_USD_per_yr": membrane,
            "maintenance_USD_per_yr": maintenance, "electricity_USD_per_yr": power,
            "total_USD_per_yr": total}


def specific_cost(cost: dict, product_mol_s: float, molar_mass: float,
                  hours_per_year: float = 8000.0) -> float:
    """USD per tonne of a product component."""
    tonnes = product_mol_s * molar_mass * 3600.0 * hours_per_year / 1000.0
    return cost["total_USD_per_yr"] / tonnes if tonnes > 0 else np.inf


def stage_economics(res: StageResult, product: str, stream: str,
                    ea: EnergyAssumptions | None = None,
                    ca: CostAssumptions | None = None) -> dict:
    """Energy and cost per tonne of `product` recovered in `stream`."""
    from .properties import gas
    ca = ca or CostAssumptions()
    e = stage_energy(res, ea)
    c = annual_cost(res.area, e["feed_compressor_kW"], e["vacuum_pump_kW"], ca)
    i = res.components.index(product)
    flow = (res.permeate_flow * res.y_perm[i] if stream.startswith("p")
            else res.retentate_flow * res.x_ret[i])
    M = gas(product)["M"]
    tonnes_per_h = flow * M * 3.6
    out = {**e, **c,
           "product_t_per_h": tonnes_per_h,
           "specific_energy_kWh_per_t": e["total_kW"] / tonnes_per_h if tonnes_per_h > 0 else np.inf,
           "specific_cost_USD_per_t": specific_cost(c, flow, M, ca.hours_per_year)}
    return out


def assumptions_table(ca: CostAssumptions | None = None,
                      ea: EnergyAssumptions | None = None) -> dict:
    """All assumptions in one dictionary (for printing next to results)."""
    return {**asdict(ea or EnergyAssumptions()), **asdict(ca or CostAssumptions())}
