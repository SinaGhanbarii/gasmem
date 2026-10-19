"""Stage-cut sweep from the command line: CSV + figure.

Examples
--------
Binary, dimensionless (p_low = 1 bar, p_high = phi):
    python scripts/stage_cut_sweep.py binary --alpha 20 --phi 8 --x-feed 0.2

Multicomponent preset:
    python scripts/stage_cut_sweep.py preset --membrane "Polyimide CO2/CH4, a=35 (Makaruk 2010)" \
        --feed "biogas, 60% CH4 (Makaruk 2010)" --p-high 15 --p-low 1 --flow-nm3h 500 --theta-max 0.6

Your own membrane (permeances in GPU) and feed:
    python scripts/stage_cut_sweep.py custom --permeances CO2=1000,N2=20,O2=45 \
        --feed-comp CO2=0.13,N2=0.82,O2=0.05 --p-high 3 --p-low 0.2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasmem import (FEED_PRESETS, MEMBRANE_PRESETS, PATTERNS, Membrane, binary_sweep,  # noqa: E402
                    nm3h_to_mols, sweep_stage_cut)
from gasmem.plotting import plot_purity_recovery, plot_stage_cut_tradeoff  # noqa: E402


def parse_pairs(text):
    out = {}
    for item in text.split(","):
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["binary", "preset", "custom"])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--x-feed", type=float, default=0.2)
    ap.add_argument("--membrane", default="Polyimide CO2/CH4, a=35 (Makaruk 2010)", choices=list(MEMBRANE_PRESETS))
    ap.add_argument("--feed", default="biogas, 60% CH4 (Makaruk 2010)", choices=list(FEED_PRESETS))
    ap.add_argument("--permeances", help="e.g. CO2=1000,N2=20 (GPU)")
    ap.add_argument("--feed-comp", help="e.g. CO2=0.13,N2=0.87")
    ap.add_argument("--p-high", type=float, default=10.0, help="feed pressure, bar")
    ap.add_argument("--p-low", type=float, default=1.0, help="permeate pressure, bar")
    ap.add_argument("--flow-nm3h", type=float, default=100.0)
    ap.add_argument("--theta-min", type=float, default=0.02)
    ap.add_argument("--theta-max", type=float, default=0.9)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--out", default="stage_cut_sweep")
    a = ap.parse_args()

    thetas = np.linspace(a.theta_min, a.theta_max, a.points)
    patterns = a.patterns.split(",")
    if a.mode == "binary":
        df = binary_sweep(a.alpha, a.phi, a.x_feed, thetas, patterns)
        comp, slow, area_col, area_label = "A", "B", "area_dimless", "A·Q_A·p_h / F"
        title = f"α = {a.alpha:g}, φ = {a.phi:g}, x_F = {a.x_feed:g}"
    else:
        if a.mode == "preset":
            m = Membrane(MEMBRANE_PRESETS[a.membrane])
            xf = {c: v for c, v in FEED_PRESETS[a.feed].items() if c in m.components}
            title = f"{a.membrane} · {a.feed}"
        else:
            if not (a.permeances and a.feed_comp):
                ap.error("custom mode needs --permeances and --feed-comp")
            m = Membrane(parse_pairs(a.permeances))
            xf = parse_pairs(a.feed_comp)
            title = "custom membrane"
        df = sweep_stage_cut(m, xf, a.p_high, a.p_low, thetas, patterns,
                             nm3h_to_mols(a.flow_nm3h), quiet=False)
        comp = max(m.gpu, key=m.gpu.get)
        slow = max((c for c in xf if c != comp), key=lambda c: xf[c])
        area_col, area_label = "area_m2", "Membrane area (m²)"
        title += f" · {a.p_high:g}/{a.p_low:g} bar"

    df.to_csv(f"{a.out}.csv", index=False)
    plot_stage_cut_tradeoff(df, comp, area_col, patterns, slow=slow, title=title,
                            area_label=area_label).savefig(f"{a.out}.png")
    plot_purity_recovery(df, comp, patterns).savefig(f"{a.out}_purity_recovery.png")
    print(f"wrote {a.out}.csv, {a.out}.png, {a.out}_purity_recovery.png")


if __name__ == "__main__":
    main()
