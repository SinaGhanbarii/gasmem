"""Tests for the v1.2 additions: permeate sweep, shell-side pressure drop,
optimisation and the own-data tools."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_bvp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gasmem import (BAR, FEED_PRESETS, MEMBRANE_PRESETS, HollowFibreModule,  # noqa: E402
                    Membrane, StageSpec, nm3h_to_mols, optimise, rate_module, single_stage,
                    solve_stage, stage_cut_for_spec, two_stage_stripping)
from gasmem.properties import mixture_viscosity  # noqa: E402
from gasmem.userdata import compare, fit_permeances, read, template  # noqa: E402

CO2N2 = Membrane({"CO2": 1000, "N2": 20})
FLUE = np.array([0.13, 0.87])


# --- sweep ----------------------------------------------------------------
def test_sweep_countercurrent_matches_independent_bvp():
    th, S, z = 0.12, 0.1, np.array([0.0, 1.0])
    ph, pl, Q = 1.2 * BAR, 0.2 * BAR, CO2N2.Q
    r = solve_stage(CO2N2, FLUE, th, 1.2, 0.2, "countercurrent", 1.0,
                    sweep_flow=S, sweep_composition={"N2": 1})

    def f(s, Y, p):
        N, P = Y[:2], Y[2:]
        J = Q[:, None] * (ph * N / N.sum(0) - pl * P / P.sum(0))
        return np.vstack([-p[0] * J, -p[0] * J])

    def bc(Ya, Yb, p):
        return np.array([Ya[0] - FLUE[0], Ya[1] - FLUE[1], Yb[2] - 1e-30, Yb[3] - S,
                         Yb[0] + Yb[1] - (1 - th)])
    s = np.linspace(0, 1, 50)
    Y0 = np.vstack([np.outer(FLUE, 1 - th * s), np.outer([0.1, 0.12], 1 - s) + np.outer(S * z, s)])
    sol = solve_bvp(f, bc, s, Y0, p=[50.0], tol=1e-8, max_nodes=100000)
    assert sol.status == 0
    assert r.area == pytest.approx(sol.p[0], rel=1e-5)
    assert r.x_ret[0] == pytest.approx(sol.y[0, -1] / sol.y[:2, -1].sum(), abs=1e-7)


@pytest.mark.parametrize("pattern", ["countercurrent", "cocurrent"])
def test_sweep_balances_and_helps(pattern):
    r0 = solve_stage(CO2N2, FLUE, 0.12, 1.2, 0.2, pattern, 1.0)
    r1 = solve_stage(CO2N2, FLUE, 0.12, 1.2, 0.2, pattern, 1.0,
                     sweep_flow=0.1, sweep_composition={"N2": 1})
    assert r1.mass_balance_error() < 1e-9
    assert r1.permeate_flow == pytest.approx(0.12 + 0.1)
    assert r1.recovery("CO2") > r0.recovery("CO2")
    assert r1.area < r0.area


def test_tiny_sweep_tends_to_no_sweep():
    r0 = solve_stage(CO2N2, FLUE, 0.12, 1.2, 0.2, "countercurrent", 1.0)
    r1 = solve_stage(CO2N2, FLUE, 0.12, 1.2, 0.2, "countercurrent", 1.0,
                     sweep_flow=1e-6, sweep_composition={"N2": 1})
    assert r1.area == pytest.approx(r0.area, rel=1e-3)


def test_sweep_needs_a_permeate_flow_path():
    with pytest.raises(ValueError):
        solve_stage(CO2N2, FLUE, 0.12, 1.2, 0.2, "cross_flow", 1.0,
                    sweep_flow=0.1, sweep_composition={"N2": 1})


def test_air_sweep_back_permeation_and_flowsheet_balance():
    m = Membrane({"CO2": 1000, "N2": 20, "O2": 50})
    xf = {"CO2": 0.13, "N2": 0.87, "O2": 0.0}
    air = {"N2": 0.79, "O2": 0.21}
    r = solve_stage(m, xf, 0.12, 1.2, 0.2, "countercurrent", 1.0, sweep_flow=0.1,
                    sweep_composition=air)
    assert r.x_ret[2] > 0                      # oxygen entered the retentate
    assert r.mass_balance_error() < 1e-9
    fs = two_stage_stripping(m, xf, 10, StageSpec(0.12, 1.2, 0.2, "countercurrent"),
                             StageSpec(0.05, 1.2, 0.2, "countercurrent", 0.1, air))
    assert fs.converged and "sweep 2" in fs.streams
    assert fs.mass_balance_error() < 1e-6


@pytest.mark.parametrize("p_high,p_low", [(1.1, 0.22), (1.2, 0.2), (2.0, 0.2)])
@pytest.mark.parametrize("theta", [0.05, 0.12, 0.2])
def test_air_sweep_robust(p_high, p_low, theta):
    m = Membrane({"CO2": 1000, "N2": 20, "O2": 50})
    r = solve_stage(m, {"CO2": 0.13, "N2": 0.87, "O2": 0.0}, theta, p_high, p_low,
                    "countercurrent", 1.0, sweep_flow=0.1,
                    sweep_composition={"N2": 0.79, "O2": 0.21})
    assert r.mass_balance_error() < 1e-9
    assert 0 < r.x_ret[2] < 0.021


# --- shell-side pressure drop --------------------------------------------
@pytest.mark.parametrize("pattern", ["cross_flow", "cocurrent", "countercurrent"])
def test_shell_pressure_drop_matches_analytic_limit(pattern):
    # negligible permeation: p_in^2 - p_out^2 = 2 K_s mu F
    m = Membrane({"CO2": 1e-4, "CH4": 1e-4 / 35})
    xf, F = np.array([0.4, 0.6]), 50.0
    mod = HollowFibreModule.for_area(1000, packing_fraction=0.6, shell_friction_factor=20)
    mu = mixture_viscosity(m.components, xf)
    dp = (10 * BAR - np.sqrt((10 * BAR) ** 2 - 2 * mod.shell_constant() * mu * F)) / BAR
    r = rate_module(m, xf, F, 10, 1, mod, pattern)
    assert r.extra["shell_dp_bar"] == pytest.approx(dp, rel=1e-5)


def test_shell_pressure_drop_lowers_stage_cut():
    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
    F = nm3h_to_mols(1000)
    a = rate_module(m, xf, F, 10, 1, HollowFibreModule.for_area(1744), "cross_flow")
    b = rate_module(m, xf, F, 10, 1, HollowFibreModule.for_area(
        1744, packing_fraction=0.6, shell_friction_factor=50), "cross_flow")
    assert a.extra["shell_dp_bar"] == 0.0
    assert b.extra["shell_dp_bar"] > 0.1
    assert b.theta < a.theta


def test_packing_fraction_validated():
    with pytest.raises(ValueError):
        HollowFibreModule.for_area(100, packing_fraction=0.95)


# --- optimisation -----------------------------------------------------------
BIO_M = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
BIO_X = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]


def test_stage_cut_for_spec_hits_target():
    r = stage_cut_for_spec(BIO_M, BIO_X, "CH4", 0.98, 10, 1, "cross_flow")
    assert r.x_ret[1] == pytest.approx(0.98, abs=1e-8)
    r = stage_cut_for_spec(BIO_M, BIO_X, "CO2", 0.9, 10, 1, "cross_flow",
                           stream="permeate", spec="recovery")
    assert r.recovery("CO2") == pytest.approx(0.9, abs=1e-8)


def test_optimise_single_stage_meets_constraint():
    F = nm3h_to_mols(1000)

    def design(theta, p_high):
        return single_stage(BIO_M, BIO_X, F, StageSpec(theta, p_high, 1.0, "cross_flow"))
    o = optimise(design, {"theta": (0.3, 0.7), "p_high": (5, 20)}, "slow-gas product", "CH4",
                 purity=0.98, objective="area", n_scan=16)
    assert o.feasible
    assert o.purity >= 0.98 - 1e-6
    # minimum area at fixed purity: highest pressure, constraint active
    assert o.variables["p_high"] == pytest.approx(20, rel=1e-3)
    assert o.purity == pytest.approx(0.98, abs=1e-4)


# --- own data ------------------------------------------------------------------
def test_template_and_percent_check(tmp_path):
    t = template(["CO2", "CH4"], tmp_path / "t.csv")
    assert {"x_feed_CO2", "y_perm_CH4", "stage_cut", "area_m2"} <= set(t.columns)
    bad = pd.DataFrame({"p_feed_bar": [10], "p_perm_bar": [1], "x_feed_CO2": [40],
                        "x_feed_CH4": [60], "stage_cut": [0.4]})
    with pytest.raises(ValueError, match="fractions"):
        read(bad)


def _synthetic(m, pattern):
    rows = []
    for th, ph in [(0.3, 8), (0.45, 10), (0.6, 12), (0.5, 6)]:
        r = solve_stage(m, BIO_X, th, ph, 1.0, pattern, 2.0)
        rows.append({"p_feed_bar": ph, "p_perm_bar": 1.0, "feed_flow_mol_s": 2.0,
                     "x_feed_CO2": 0.4, "x_feed_CH4": 0.6, "stage_cut": th, "area_m2": r.area,
                     "y_perm_CO2": r.y_perm[0], "x_ret_CH4": r.x_ret[1]})
    return pd.DataFrame(rows)


def test_compare_reproduces_its_own_predictions():
    data = _synthetic(BIO_M, "cross_flow")
    c = compare(BIO_M, data, "cross_flow")
    assert not c.failed
    assert c.table["error"].abs().max() < 1e-6
    c2 = compare(BIO_M, data.drop(columns="stage_cut").assign(stage_cut=np.nan), "cross_flow")
    assert set(c2.table["mode"]) == {"rating"}


def test_fit_recovers_permeances():
    true = Membrane({"CO2": 80.0, "CH4": 80.0 / 25})
    data = _synthetic(true, "cross_flow")
    fitted, c, info = fit_permeances(BIO_M, data, "cross_flow")
    assert fitted.gpu["CO2"] == pytest.approx(80.0, rel=1e-4)
    assert fitted.selectivity("CO2", "CH4") == pytest.approx(25.0, rel=1e-4)
