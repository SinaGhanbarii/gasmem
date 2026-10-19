"""Toolkit runs for the Makaruk et al. (2010) comparison (notebook 08).

Designs the two-stage cascade with recycle (98 % CH4) at several selectivities
for both methane recoveries, with cross-flow and counter-current stages, and
writes gasmem/data/makaruk2010_toolkit_runs.csv.  Takes 20-40 minutes.

    python tools/run_makaruk_cascades.py
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from gasmem import (FEED_PRESETS, EnergyAssumptions, Membrane, StageSpec,  # noqa: E402
                    mols_to_nm3h, nm3h_to_mols, two_stage_stripping)

OUT = ROOT / "gasmem" / "data" / "makaruk2010_toolkit_runs.csv"
xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
F = nm3h_to_mols(1000)
ea = EnergyAssumptions(eta_compressor=0.8 * 0.9)


def cascade(alpha, th, pattern):
    m = Membrane({"CO2": 100.0, "CH4": 100.0 / alpha})
    return two_stage_stripping(m, xf, F, StageSpec(th[0], 10, 1, pattern),
                               StageSpec(th[1], 10, 1, pattern), ea=ea)


def design(alpha, recovery, pattern, x0):
    def f(th):
        r = cascade(alpha, th, pattern)
        return [r.purity("retentate 2", "CH4") - 0.98, r.recovery("retentate 2", "CH4") - recovery]
    th = least_squares(f, x0, bounds=([0.05, 0.002], [0.8, 0.95]), xtol=1e-7, ftol=1e-7,
                       diff_step=1e-3).x
    r = cascade(alpha, th, pattern)
    return dict(pattern=pattern, alpha=alpha, recovery_target=recovery, theta1=th[0], theta2=th[1],
                purity=r.purity("retentate 2", "CH4"), recovery=r.recovery("retentate 2", "CH4"),
                area_m2=r.total_area,
                kWh_per_Nm3=r.total_power_kW / mols_to_nm3h(r.streams["retentate 2"].sum()))


def main():
    rows = []
    for rec, alphas in [(0.938, [10, 15, 22, 35, 50, 80]), (0.812, [8, 12, 22, 35, 50, 80])]:
        for a in alphas:
            t = time.time()
            cf = design(a, rec, "cross_flow", (0.3, 0.3))
            cc = design(a, rec, "countercurrent", (cf["theta1"], cf["theta2"]))
            rows += [cf, cc]
            print(f"recovery {rec}, alpha {a}: {time.time() - t:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
