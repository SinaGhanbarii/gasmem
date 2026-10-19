"""Compare single-, two- and three-stage biogas upgrading at one operating point,
and sweep the stage-1 stage cut of the three-stage layout.

    python scripts/multistage_biogas.py [--pattern countercurrent] [--flow-nm3h 500]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasmem import (FEED_PRESETS, MEMBRANE_PRESETS, CostAssumptions, EnergyAssumptions, Membrane,  # noqa: E402
                    StageSpec, nm3h_to_mols, single_stage, specific_cost_per_tonne,
                    specific_energy_kWh_per_tonne, three_stage, two_stage_enriching,
                    two_stage_stripping)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="cross_flow",
                    choices=["well_mixed", "cross_flow", "cocurrent", "countercurrent"])
    ap.add_argument("--flow-nm3h", type=float, default=1000.0)
    ap.add_argument("--out", default="multistage_biogas")
    a = ap.parse_args()

    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    xf, F, p, ca = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"], nm3h_to_mols(a.flow_nm3h), a.pattern, CostAssumptions()
    ea = EnergyAssumptions(eta_compressor=0.8 * 0.9)      # Makaruk et al. (2010)
    S = lambda th, ph, pl: StageSpec(th, ph, pl, p)
    # stage cuts tuned for about 98 % CH4 with cross-flow stages (notebook 06)
    cases = {
        "single stage": single_stage(m, xf, F, S(0.533, 10, 1), ea=ea),
        "two-stage stripping": two_stage_stripping(m, xf, F, S(0.330, 10, 1), S(0.335, 10, 1), ea=ea),
        "two-stage enriching": two_stage_enriching(m, xf, F, S(0.50, 10, 1), S(0.70, 10, 1), ea=ea),
        "three-stage": three_stage(m, xf, F, S(0.40, 10, 2), S(0.30, 10, 1), S(0.70, 2, 1), ea=ea),
    }
    rows = {}
    for name, r in cases.items():
        s = r.products["slow-gas product"]
        rows[name] = {"CH4_purity": r.purity(s, "CH4"), "CH4_recovery": r.recovery(s, "CH4"),
                      "area_m2": r.total_area, "power_kW": r.total_power_kW,
                      "kWh_per_t_CH4": specific_energy_kWh_per_tonne(r, s, "CH4"),
                      "USD_per_t_CH4_indicative": specific_cost_per_tonne(r, s, "CH4", ca),
                      "converged": r.converged}
    comp = pd.DataFrame(rows).T
    print(comp.round(4).to_string())
    comp.to_csv(f"{a.out}_layouts.csv")

    sweep = []
    for th1 in np.linspace(0.3, 0.5, 11):
        r = three_stage(m, xf, F, S(th1, 10, 2), S(0.30, 10, 1), S(0.70, 2, 1), ea=ea)
        sweep.append({"theta1": th1, "CH4_purity": r.purity("retentate 2", "CH4"),
                      "CH4_recovery": r.recovery("retentate 2", "CH4"),
                      "area_m2": r.total_area, "power_kW": r.total_power_kW,
                      "USD_per_t_CH4_indicative": specific_cost_per_tonne(r, "retentate 2", "CH4", ca)})
    pd.DataFrame(sweep).to_csv(f"{a.out}_theta1_sweep.csv", index=False)
    print(f"\nwrote {a.out}_layouts.csv and {a.out}_theta1_sweep.csv")


if __name__ == "__main__":
    main()
