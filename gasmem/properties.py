"""Units, gas properties and illustrative membrane / feed presets.

Public API convention used throughout the toolkit
-------------------------------------------------
* pressures         : bar (absolute)
* molar flows       : mol/s
* permeances        : GPU (1 GPU = 1e-6 cm3(STP) cm-2 s-1 cmHg-1)
* membrane area     : m2
* temperature       : K
Everything is converted to SI internally.

The preset permeances below are *illustrative, order-of-magnitude* values chosen
to give representative selectivities. They are not data for any specific
commercial membrane - replace them with your own measured permeances.
"""
from __future__ import annotations

import numpy as np

R = 8.314462618          # J mol-1 K-1
BAR = 1.0e5              # Pa
ATM_BAR = 1.01325        # bar
GPU = 3.3464e-10         # mol m-2 s-1 Pa-1 per GPU
T_REF = 298.15           # K
NM3_PER_MOL = 0.022414   # Nm3 (0 degC, 1 atm) per mol


# Pure-gas data near 25 degC: molar mass (kg/mol), dynamic viscosity (Pa s),
# heat-capacity ratio cp/cv (ideal gas).
GASES: dict[str, dict[str, float]] = {
    "CO2": {"M": 44.010e-3, "mu": 1.49e-5, "k": 1.29},
    "N2":  {"M": 28.013e-3, "mu": 1.78e-5, "k": 1.40},
    "O2":  {"M": 31.999e-3, "mu": 2.06e-5, "k": 1.40},
    "CH4": {"M": 16.043e-3, "mu": 1.11e-5, "k": 1.31},
    "H2":  {"M": 2.016e-3,  "mu": 0.89e-5, "k": 1.41},
    "He":  {"M": 4.0026e-3, "mu": 1.99e-5, "k": 1.66},
    "Ar":  {"M": 39.948e-3, "mu": 2.27e-5, "k": 1.67},
    "H2O": {"M": 18.015e-3, "mu": 0.99e-5, "k": 1.33},
    "CO":  {"M": 28.010e-3, "mu": 1.77e-5, "k": 1.40},
    "H2S": {"M": 34.081e-3, "mu": 1.24e-5, "k": 1.32},
    "C2H4": {"M": 28.054e-3, "mu": 1.03e-5, "k": 1.24},
    "C2H6": {"M": 30.069e-3, "mu": 0.93e-5, "k": 1.19},
    "C3H8": {"M": 44.097e-3, "mu": 0.82e-5, "k": 1.13},
}
# Generic placeholders used by Membrane.binary(...) for dimensionless studies.
DEFAULT_GAS = {"M": 30.0e-3, "mu": 1.6e-5, "k": 1.40}


def gas(name: str) -> dict[str, float]:
    """Property record for a gas, falling back to generic values."""
    return GASES.get(name, DEFAULT_GAS)


def mixture_viscosity(names, y) -> float:
    """Herning-Zipperer mixing rule (adequate for light permanent gases)."""
    y = np.asarray(y, float)
    sq = np.array([np.sqrt(gas(n)["M"]) for n in names])
    mu = np.array([gas(n)["mu"] for n in names])
    return float(np.sum(y * mu * sq) / np.sum(y * sq))


def mixture_k(names, y) -> float:
    """Ideal-gas mixture heat-capacity ratio from mole-weighted cp."""
    y = np.asarray(y, float)
    k = np.array([gas(n)["k"] for n in names])
    cp = np.sum(y * k / (k - 1.0))           # in units of R
    return float(cp / (cp - 1.0))


def nm3h_to_mols(q_nm3_per_h: float) -> float:
    """Convert Nm3/h to mol/s."""
    return q_nm3_per_h / NM3_PER_MOL / 3600.0


def mols_to_nm3h(n_mol_per_s: float) -> float:
    """Convert mol/s to Nm3/h."""
    return n_mol_per_s * NM3_PER_MOL * 3600.0


BARRER = 3.35e-16        # mol m m-2 s-1 Pa-1 per Barrer (1e-10 cm3(STP) cm cm-2 s-1 cmHg-1)


def barrer_to_gpu(permeability_barrer: float, thickness_um: float) -> float:
    """Permeance (GPU) of a selective layer: 1 GPU = 1 Barrer / 1 um."""
    if thickness_um <= 0:
        raise ValueError("thickness must be positive")
    return permeability_barrer / thickness_um


# ---------------------------------------------------------------------------
# Presets taken from published papers.  Each entry records where the numbers
# come from; see LITERATURE for full references.  Permeances in GPU.
# ---------------------------------------------------------------------------
LITERATURE = {
    "Merkel2010": "Merkel, T.C., Lin, H., Wei, X. and Baker, R. (2010) 'Power plant post-combustion "
                  "carbon dioxide capture: an opportunity for membranes', J. Membr. Sci. 359, 126-139. "
                  "doi:10.1016/j.memsci.2009.10.041",
    "MTR2012": "Merkel, T. et al. (2012) Membrane Process to Capture CO2 from Coal-Fired Power Plant "
               "Flue Gas, Final Report, DOE award DE-NT0005312, Membrane Technology and Research Inc.",
    "Makaruk2010": "Makaruk, A., Miltner, M. and Harasek, M. (2010) 'Membrane biogas upgrading processes "
                   "for the production of natural gas substitute', Sep. Purif. Technol. 74, 83-92. "
                   "doi:10.1016/j.seppur.2010.05.010",
    "Pan1986": "Pan, C.Y. (1986) 'Gas separation by high-flux, asymmetric hollow-fiber membrane', "
               "AIChE J. 32, 2020-2027. doi:10.1002/aic.690321212",
    "Coker1998": "Coker, D.T., Freeman, B.D. and Fleming, G.K. (1998) 'Modeling multicomponent gas "
                 "separation using hollow-fiber membrane contactors', AIChE J. 44, 1289-1302. "
                 "doi:10.1002/aic.690440607",
    "Feng1999": "Feng, X., Ivory, J. and Rajan, V.S.V. (1999) 'Air separation by integrally asymmetric "
                "hollow-fiber membranes', AIChE J. 45, 2142-2152. doi:10.1002/aic.690451013",
    "Hosseini2015": "Hosseini, S.S. et al. (2015) 'Simulation and sensitivity analysis of transport in "
                    "asymmetric hollow fiber membrane permeators for air separation', RSC Adv. 5 "
                    "(issue 105). doi:10.1039/C5RA13943K",
    "Sidhoum1988": "Sidhoum, M., Sengupta, A. and Sirkar, K.K. (1988) 'Asymmetric cellulose acetate "
                   "hollow fibers: studies in gas permeation', AIChE J. 34, 417-425. "
                   "doi:10.1002/aic.690340309",
    "Robeson2008": "Robeson, L.M. (2008) 'The upper bound revisited', J. Membr. Sci. 320, 390-400. "
                   "doi:10.1016/j.memsci.2008.04.030",
}

_PAN_H2 = 2.84e-8 / GPU   # Pan (1986) Fig. 2: H2 Q/d = 2.84e-8 mol m-2 s-1 Pa-1

MEMBRANE_INFO: dict[str, dict] = {
    "Polaris CO2/N2 (Merkel 2010)": {
        "permeances": {"CO2": 1000.0, "N2": 20.0},
        "source": "Merkel2010",
        "note": "Base-case Polaris membrane used throughout the paper's design calculations "
                "(CO2 1000 gpu, CO2/N2 = 50)."},
    "Polaris pure-gas (MTR 2012)": {
        "permeances": {"CO2": 1300.0, "N2": 24.0, "CH4": 80.0, "H2": 110.0},
        "source": "MTR2012",
        "note": "Pure-gas roll average, 23 degC, 50 psig feed (Table 1 of the report). Mixed-gas "
                "CO2/N2 selectivity at 50 degC was only about 20-25."},
    "Polyimide CO2/CH4, a=35 (Makaruk 2010)": {
        "permeances": {"CO2": 100.0, "CH4": 100.0 / 35.0},
        "source": "Makaruk2010",
        "note": "CO2 permeance 100 GPU (Table 2); ideal CO2/CH4 selectivity 35, described as "
                "state of the art for polyimide hollow fibres."},
    "Cellulose acetate H2 recovery (Pan 1986)": {
        "permeances": {"H2": _PAN_H2, "N2": _PAN_H2 / 100 * 1.04, "CH4": _PAN_H2 / 100,
                       "Ar": _PAN_H2 / 100 * 2.71},
        "source": "Pan1986",
        "note": "Asymmetric cellulose acetate hollow fibre at 25 degC: H2/CH4 = 100, "
                "N2/CH4 = 1.04, Ar/CH4 = 2.71 (pure-gas values, Fig. 2)."},
    "Cellulose acetate CO2/hydrocarbons (Pan 1986)": {
        "permeances": {"CO2": 1.34e-8 / GPU, "CH4": 1.34e-8 / GPU / 36.0,
                       "C2H6": 1.34e-8 / GPU / 36.0 * 0.275, "C3H8": 1.34e-8 / GPU / 36.0 * 0.0536},
        "source": "Pan1986",
        "note": "Mixed-gas values fitted by Pan to one run at 10 degC (Fig. 6): CO2 Q/d = "
                "1.34e-8 mol/(m2 s Pa), CO2/CH4 = 36, C2H6/CH4 = 0.275, C3H8/CH4 = 0.0536."},
    "Cellulose acetate H2S/CO2 sour gas (Pan 1986)": {
        "permeances": {"H2S": 2.45e-8 / GPU, "CO2": 2.45e-8 / GPU, "CH4": 2.45e-8 / GPU / 18.5,
                       "N2": 2.45e-8 / GPU / 18.5, "C2H6": 2.45e-8 / GPU / 18.5 * 0.275},
        "source": "Pan1986",
        "note": "Mixed-gas values fitted by Pan to one run at 50 degC (Fig. 7): H2S Q/d = 2.45e-8 "
                "mol/(m2 s Pa), H2S/CH4 = CO2/CH4 = 18.5, N2/CH4 = 1. C2H6 (0.11 % of the feed) is "
                "not given; C2H6/CH4 = 0.275 from Fig. 6 is assumed."},
    "Polysulfone-like air (Coker 1998)": {
        "permeances": {"N2": 3.57, "O2": 20.0, "CO2": 60.0, "H2O": 1000.0},
        "source": "Coker1998",
        "note": "Table 2: permeances typical of polysulfone with a 0.1 um selective layer."},
    "Polysulfone-like H2/hydrocarbons (Coker 1998)": {
        "permeances": {"H2": 100.0, "C2H4": 3.03, "CH4": 2.86, "C2H6": 2.00, "C3H8": 1.89},
        "source": "Coker1998",
        "note": "Table 3. Component labels are missing from the available scan; they are assigned "
                "here in the order that matches the feed fractions and the text (CH4 is the "
                "intermediate component with a composition maximum). Check against the PDF."},
    "Asymmetric hollow fibre air (Feng 1999)": {
        "permeances": {"O2": 30.78e-10 / GPU, "N2": 5.7e-10 / GPU},
        "source": "Feng1999",
        "note": "23 degC pure-gas permeances as tabulated by Hosseini et al. (2015) from Feng's "
                "Fig. 2: O2 30.78e-10 and N2 5.7e-10 mol/(m2 s Pa), O2/N2 = 5.4."},
}

MEMBRANE_PRESETS: dict[str, dict[str, float]] = {k: v["permeances"] for k, v in MEMBRANE_INFO.items()}

FEED_INFO: dict[str, dict] = {
    "flue gas, 13% CO2 (Merkel 2010)": {"composition": {"CO2": 0.13, "N2": 0.87},
                                         "source": "Merkel2010"},
    "biogas, 60% CH4 (Makaruk 2010)": {"composition": {"CH4": 0.60, "CO2": 0.40},
                                       "source": "Makaruk2010"},
    "ammonia purge gas, NH3-free (Pan 1986)": {
        "composition": {"H2": 0.5178, "N2": 0.2469, "CH4": 0.1957, "Ar": 0.0396},
        "source": "Pan1986"},
    "acid gas with hydrocarbons (Pan 1986)": {
        "composition": {"CO2": 0.485, "CH4": 0.279, "C2H6": 0.1626, "C3H8": 0.0734},
        "source": "Pan1986"},
    "sour gas, 22% H2S (Pan 1986)": {
        "composition": {"H2S": 0.2209, "CO2": 0.0338, "CH4": 0.6605, "N2": 0.0837, "C2H6": 0.0011},
        "source": "Pan1986"},
    "humid air, 40 C 10 atm (Coker 1998)": {
        "composition": {"N2": 0.7841, "O2": 0.2084, "CO2": 0.0003, "H2O": 0.0072},
        "source": "Coker1998"},
    "refinery hydrotreater purge (Coker 1998)": {
        "composition": {"H2": 0.650, "C2H4": 0.025, "CH4": 0.210, "C2H6": 0.080, "C3H8": 0.035},
        "source": "Coker1998"},
    "air, 20.5% O2 (Feng 1999)": {"composition": {"O2": 0.205, "N2": 0.795},
                                  "source": "Feng1999"},
}

FEED_PRESETS: dict[str, dict[str, float]] = {k: v["composition"] for k, v in FEED_INFO.items()}

# Which feed goes with which membrane in the examples
PAIRINGS = {
    "Polaris CO2/N2 (Merkel 2010)": "flue gas, 13% CO2 (Merkel 2010)",
    "Polaris pure-gas (MTR 2012)": "flue gas, 13% CO2 (Merkel 2010)",
    "Polyimide CO2/CH4, a=35 (Makaruk 2010)": "biogas, 60% CH4 (Makaruk 2010)",
    "Cellulose acetate H2 recovery (Pan 1986)": "ammonia purge gas, NH3-free (Pan 1986)",
    "Cellulose acetate CO2/hydrocarbons (Pan 1986)": "acid gas with hydrocarbons (Pan 1986)",
    "Cellulose acetate H2S/CO2 sour gas (Pan 1986)": "sour gas, 22% H2S (Pan 1986)",
    "Polysulfone-like air (Coker 1998)": "humid air, 40 C 10 atm (Coker 1998)",
    "Polysulfone-like H2/hydrocarbons (Coker 1998)": "refinery hydrotreater purge (Coker 1998)",
    "Asymmetric hollow fibre air (Feng 1999)": "air, 20.5% O2 (Feng 1999)",
}


def describe(preset: str) -> str:
    """Source and notes for a membrane or feed preset."""
    info = MEMBRANE_INFO.get(preset) or FEED_INFO.get(preset)
    if info is None:
        raise KeyError(preset)
    return f"{preset}\n  {info.get('note', '')}\n  Source: {LITERATURE[info['source']]}"
