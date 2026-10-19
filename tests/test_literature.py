"""Regression tests against published results (see notebook 07 for the full comparison)."""
import numpy as np
import pytest
from scipy.optimize import brentq

from gasmem import (FEED_PRESETS, LITERATURE, MEMBRANE_INFO, MEMBRANE_PRESETS, PAIRINGS,
                    HollowFibreModule, Membrane, countercurrent, cross_flow, describe,
                    distance_to_bound, nm3h_to_mols, rate_module, solve_stage,
                    upper_bound_selectivity)
from gasmem.robeson import UPPER_BOUND_2008


def test_every_preset_has_a_reference_and_a_feed():
    for name, info in MEMBRANE_INFO.items():
        assert info["source"] in LITERATURE
        assert PAIRINGS[name] in FEED_PRESETS
        assert "Source:" in describe(name)
    for comp in FEED_PRESETS.values():
        assert sum(comp.values()) == pytest.approx(1.0, abs=1e-9)


def test_permeability_to_permeance():
    m = Membrane.from_permeability({"CO2": 100.0, "CH4": 4.0}, thickness_um=0.5)
    assert m.gpu["CO2"] == pytest.approx(200.0)
    assert m.selectivity("CO2", "CH4") == pytest.approx(25.0)


def test_robeson_bound_consistency():
    for pair, (k, n) in UPPER_BOUND_2008.items():
        assert distance_to_bound(pair, k * 10.0 ** n, 10.0) == pytest.approx(1.0)
    # 6FDA-durene sits just below the 2008 bound (Robeson 2008, Table 2)
    assert 0.5 < distance_to_bound("CO2/CH4", 677.8, 20.18) < 1.0
    # the 2008 bound lies above the 1991 bound
    assert upper_bound_selectivity("O2/N2", 10.0, 2008) > upper_bound_selectivity("O2/N2", 10.0, 1991)


def test_pan_1986_measured_point_is_closest_to_cross_flow():
    m = Membrane(MEMBRANE_PRESETS["Cellulose acetate H2 recovery (Pan 1986)"])
    xf = FEED_PRESETS["ammonia purge gas, NH3-free (Pan 1986)"]
    y = {p: solve_stage(m, xf, 0.55, 69.64, 11.23, p).y_perm[0]
         for p in ("cross_flow", "cocurrent", "well_mixed")}
    measured = 0.866
    assert abs(y["cross_flow"] - measured) < 0.03
    assert abs(y["cross_flow"] - measured) < abs(y["cocurrent"] - measured)


@pytest.mark.parametrize("pattern,purity,area_mm2", [("cross_flow", 0.289, 11.0),
                                                      ("countercurrent", 0.406, 6.8)])
def test_merkel_2010_fig6(pattern, purity, area_mm2):
    m = Membrane(MEMBRANE_PRESETS["Polaris CO2/N2 (Merkel 2010)"])
    xf = FEED_PRESETS["flue gas, 13% CO2 (Merkel 2010)"]
    F = 500 / 0.022414
    th = brentq(lambda t: solve_stage(m, xf, t, 1.1, 0.22, pattern, F).x_ret[0] - 0.021,
                0.05, 0.9, xtol=1e-7)
    r = solve_stage(m, xf, th, 1.1, 0.22, pattern, F)
    assert r.y_perm[0] == pytest.approx(purity, abs=0.01)
    assert r.area / 1e6 == pytest.approx(area_mm2, rel=0.08)


def test_merkel_2010_pressure_ratio_limit():
    r = cross_flow(Membrane({"CO2": 1000, "N2": 0.1}), [0.10, 0.90], 0.01, 5.0, 1.0, profile=False)
    assert 0.48 < r.y_perm[0] <= 0.50


def test_coker_1998_refinery_module_stage_cut():
    m = Membrane(MEMBRANE_PRESETS["Polysulfone-like H2/hydrocarbons (Coker 1998)"])
    xf = FEED_PRESETS["refinery hydrotreater purge (Coker 1998)"]
    mod = HollowFibreModule(500_000, length=0.8, d_inner=150e-6, d_outer=300e-6)
    r = rate_module(m, xf, nm3h_to_mols(283.2), 42.4, 7.9, mod, "countercurrent",
                    pressure_drop=False, T=323.15)
    assert r.theta == pytest.approx(0.946, abs=0.003)
    ch4 = r.profile["x"][m.components.index("CH4")]
    assert 0.05 < r.profile["area_fraction"][np.argmax(ch4)] < 0.6     # interior maximum


def test_feng_1999_productivity_halves_from_95_to_98_percent():
    m = Membrane(MEMBRANE_PRESETS["Asymmetric hollow fibre air (Feng 1999)"])
    xf = FEED_PRESETS["air, 20.5% O2 (Feng 1999)"]

    def productivity(purity):
        th = brentq(lambda t: countercurrent(m, xf, t, 7.83, 0.933, profile=False).x_ret[1] - purity,
                    0.25, 0.9, xtol=1e-7)
        r = countercurrent(m, xf, th, 7.83, 0.933, profile=False)
        return r.retentate_flow * r.x_ret[1] / r.area

    assert 0.4 < productivity(0.98) / productivity(0.95) < 0.65


# ---------------------------------------------------------------- digitised data
from gasmem.datasets import available, load  # noqa: E402


def test_digitised_datasets_load():
    s = available()
    assert s.points.sum() == 338
    for src in ("Pan1986", "Feng1999", "Makaruk2010"):
        d = load(src)
        assert {"figure", "quantity", "series", "x", "y", "px", "py"} <= set(d.columns)


def _rms(df, solve, value):
    cache, err = {}, []
    for row in df.itertuples():
        k = round(row.x, 5)
        if k not in cache:
            cache[k] = solve(row.x)
        err.append(value(cache[k], row) - row.y)
    return float(np.sqrt(np.mean(np.square(err))))


def test_pan_1986_fig3_impurities_predicted_by_cross_flow():
    m = Membrane(MEMBRANE_PRESETS["Cellulose acetate H2 recovery (Pan 1986)"])
    xf = FEED_PRESETS["ammonia purge gas, NH3-free (Pan 1986)"]
    d = load("Pan1986").query("figure == 'Fig. 3'")
    rms = _rms(d, lambda t: cross_flow(m, xf, t, 69.64, 11.23, profile=False),
               lambda r, row: 100 * r.y_perm[r.components.index(row.quantity.split()[1])])
    assert rms < 0.7


def test_pan_1986_fig6_acid_gas():
    m = Membrane(MEMBRANE_PRESETS["Cellulose acetate CO2/hydrocarbons (Pan 1986)"])
    xf = FEED_PRESETS["acid gas with hydrocarbons (Pan 1986)"]
    d = load("Pan1986").query("figure == 'Fig. 6'")
    rms = _rms(d, lambda t: cross_flow(m, xf, t, 35.28, 0.928, profile=False),
               lambda r, row: 100 * r.y_perm[r.components.index(row.quantity.split()[1])])
    assert rms < 0.6


@pytest.mark.parametrize("series,pattern", [("bore-side feed, countercurrent", "countercurrent"),
                                            ("bore-side feed, cocurrent", "cocurrent")])
def test_feng_1999_figs4_5_matching_flow_pattern(series, pattern):
    m = Membrane(MEMBRANE_PRESETS["Asymmetric hollow fibre air (Feng 1999)"])
    d = load("Feng1999").query("figure in ['Fig. 4', 'Fig. 5'] and series == @series")
    rms = _rms(d, lambda t: solve_stage(m, [0.205, 0.795], t, 7.828, 0.933, pattern),
               lambda r, row: 100 * (r.x_ret[1] if "N2" in row.quantity else r.y_perm[0]))
    assert rms < 0.6
