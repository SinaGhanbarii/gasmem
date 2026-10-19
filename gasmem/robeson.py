"""Robeson upper bound: the empirical trade-off between permeability and
selectivity of polymer membranes.

    P_fast = k * alpha ** n      (k in Barrer, n < 0)

so the best selectivity reachable at a given fast-gas permeability is
alpha_max = (P_fast / k) ** (1 / n).

Constants are from Robeson (2008), Table 12: the 1991 ("prior") and 2008
("present") bounds.  Reference polymer points are from Tables 1, 2, 4 and 8 of
the same paper (pure-gas data close to the 2008 bound).

Robeson, L.M. (2008) 'The upper bound revisited', J. Membr. Sci. 320, 390-400.
doi:10.1016/j.memsci.2008.04.030
"""
from __future__ import annotations

import numpy as np

# pair: (k [Barrer], n)
UPPER_BOUND_1991 = {
    "O2/N2": (389_224.0, -5.800),
    "CO2/CH4": (1_073_700.0, -2.6264),
    "H2/N2": (52_918.0, -1.5275),
    "H2/CH4": (18_500.0, -1.2112),
    "He/N2": (12_500.0, -1.0242),
    "He/CH4": (5_002.0, -0.7857),
    "H2/CO2": (1_200.0, -1.9363),
}
UPPER_BOUND_2008 = {
    "O2/N2": (1_396_000.0, -5.666),
    "CO2/CH4": (5_369_140.0, -2.636),
    "H2/N2": (97_650.0, -1.4841),
    "H2/CH4": (27_200.0, -1.107),
    "He/N2": (19_890.0, -1.017),
    "He/CH4": (19_800.0, -0.809),
    "CO2/N2": (30_967_000.0, -2.888),
    "N2/CH4": (2_570.0, -4.507),
    "H2/CO2": (4_515.0, -2.302),
}

# (polymer, P_fast [Barrer], selectivity) close to the 2008 bound
REFERENCE_POLYMERS = {
    "O2/N2": [("Polyimide (BPDA-ODA)", 0.079, 19.8), ("Polyimide (BTDA-ODA)", 0.170, 14.2),
              ("Polypyrrolone 6FDA/PMDA-TAB", 1.01, 10.3), ("Polyimide (BADBSBF-BTDA)", 18.0, 9.0),
              ("PIM-7", 190.0, 4.5), ("PIM-1", 370.0, 4.0)],
    "CO2/CH4": [("Polypyrrolone 6FDA/PMDA-TAB", 3.13, 140.0), ("Polyimide TADATO/DSDA-DDBT", 45.0, 60.0),
                ("Polyimide 6FDA-TMPDA", 555.7, 22.7), ("Polyimide 6FDA-durene", 677.8, 20.18),
                ("PIM-1", 2300.0, 18.4), ("PTMSP", 29000.0, 4.46)],
    "H2/CH4": [("Sulfonated polyimide DAPHFDS(H)", 52.0, 325.0), ("Polyimide 6FDA-mMPD", 106.0, 121.0),
               ("Polyimide 6FDA-DDBT", 156.0, 78.8), ("Hyflon AD60X", 187.0, 61.7),
               ("Teflon AF-2400", 3300.0, 5.5), ("PTMSP", 17000.0, 1.13)],
    "CO2/N2": [("Polyphosphazene (MEEP)", 250.0, 62.5), ("Modified PDMS", 2000.0, 34.2),
               ("PIM-1", 2300.0, 25.0), ("PTMGP", 14000.0, 14.0), ("PTMSP", 29000.0, 10.7)],
}


def upper_bound_selectivity(pair: str, permeability_fast: float | np.ndarray, year: int = 2008):
    """Upper-bound selectivity at a given fast-gas permeability (Barrer)."""
    table = UPPER_BOUND_2008 if year == 2008 else UPPER_BOUND_1991
    if pair not in table:
        raise KeyError(f"no {year} upper bound for {pair}; available: {sorted(table)}")
    k, n = table[pair]
    return (np.asarray(permeability_fast, float) / k) ** (1.0 / n)


def distance_to_bound(pair: str, permeability_fast: float, selectivity: float,
                      year: int = 2008) -> float:
    """Ratio of a membrane's selectivity to the upper-bound selectivity at the
    same permeability (1 = on the bound, > 1 = above it)."""
    return float(selectivity / upper_bound_selectivity(pair, permeability_fast, year))


def plot_upper_bound(pair: str, points: dict | None = None, ax=None, show_reference=True,
                     prange=(1e-2, 1e5)):
    """Robeson plot for a gas pair.

    points : optional {label: (P_fast [Barrer], selectivity)} for your own membranes.
    """
    import matplotlib.pyplot as plt
    from .plotting import INK, INK2, SERIES

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
    P = np.logspace(np.log10(prange[0]), np.log10(prange[1]), 200)
    if pair in UPPER_BOUND_1991:
        ax.plot(P, upper_bound_selectivity(pair, P, 1991), color=INK2, ls="--", lw=1.4,
                label="1991 upper bound")
    ax.plot(P, upper_bound_selectivity(pair, P, 2008), color=INK, lw=1.8, label="2008 upper bound")
    if show_reference and pair in REFERENCE_POLYMERS:
        ref = REFERENCE_POLYMERS[pair]
        ax.scatter([r[1] for r in ref], [r[2] for r in ref], s=22, color=INK2, alpha=0.6,
                   label="polymers near the bound (Robeson 2008)", zorder=3)
    for i, (label, (p, a)) in enumerate((points or {}).items()):
        ax.scatter([p], [a], s=60, color=SERIES[i % len(SERIES)], edgecolor="white",
                   linewidth=1.5, zorder=4, label=label)
    fast, slow = pair.split("/")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*prange)
    ax.set_ylim(1, max(1000, ax.get_ylim()[1]))
    ax.set_xlabel(f"{fast} permeability (Barrer)")
    ax.set_ylabel(f"{pair} selectivity")
    ax.set_title(f"Robeson plot: {pair}")
    ax.legend(fontsize=8, loc="upper right")
    return ax.figure
