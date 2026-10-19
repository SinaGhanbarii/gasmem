"""Verification suite: analytical solutions, limiting cases, balances and
an independent discretised counter-current solver."""
import numpy as np
import pytest

from gasmem import (FEED_PRESETS, MEMBRANE_PRESETS, HollowFibreModule, InfeasibleError,
                    Membrane, StageSpec, cocurrent, compression_power, countercurrent,
                    cross_flow, local_permeate, rate_module, single_stage, solve_stage,
                    three_stage, two_stage_enriching, two_stage_stripping, weller_steiner_theta,
                    well_mixed, well_mixed_binary)
from gasmem.properties import BAR, R, T_REF

PATTERNS = ["well_mixed", "cross_flow", "cocurrent", "countercurrent"]
BIN = Membrane.binary(10.0)
XB = [0.209, 0.791]


# ---------------------------------------------------------------- analytical
@pytest.mark.parametrize("theta", [0.05, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("alpha,phi", [(10, 4), (50, 10), (5, 50)])
def test_well_mixed_closed_form_matches_multicomponent_solver(theta, alpha, phi):
    cf = well_mixed_binary(alpha, phi, 0.3, theta)
    r = well_mixed(Membrane.binary(alpha), [0.3, 0.7], theta, phi, 1.0)
    assert r.y_perm[0] == pytest.approx(cf["y"], abs=1e-10)
    assert r.x_ret[0] == pytest.approx(cf["x"], abs=1e-10)
    A_dimless = r.area * Membrane.binary(alpha).Q[0] * phi * BAR
    assert A_dimless == pytest.approx(cf["area_dimless"], rel=1e-9)


@pytest.mark.parametrize("theta", [0.05, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("alpha,phi,xf", [(10, 4, 0.209), (30, 20, 0.5), (4, 100, 0.1)])
def test_cross_flow_matches_weller_steiner(theta, alpha, phi, xf):
    xo, yp = weller_steiner_theta(alpha, phi, xf, theta)
    r = cross_flow(Membrane.binary(alpha), [xf, 1 - xf], theta, phi, 1.0)
    assert r.x_ret[0] == pytest.approx(xo, rel=1e-6, abs=1e-12)
    assert r.y_perm[0] == pytest.approx(yp, rel=1e-7)


# ------------------------------------------------------------- consistency
def _cases():
    for mk, fk in [("Polyimide CO2/CH4, a=35 (Makaruk 2010)", "biogas, 60% CH4 (Makaruk 2010)"),
                   ("Polaris CO2/N2 (Merkel 2010)", "flue gas, 13% CO2 (Merkel 2010)"),
                   ("Cellulose acetate H2 recovery (Pan 1986)", "ammonia purge gas, NH3-free (Pan 1986)")]:
        yield Membrane(MEMBRANE_PRESETS[mk]), FEED_PRESETS[fk]
    yield BIN, XB


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("theta", [0.1, 0.5, 0.85])
def test_mass_balance_and_bounds(pattern, theta):
    for m, xf in _cases():
        r = solve_stage(m, xf, theta, 12.0, 1.0, pattern)
        assert r.mass_balance_error() < 1e-8
        assert np.all(r.y_perm >= 0) and np.all(r.x_ret >= -1e-12)
        assert r.area > 0
        assert abs(r.y_perm.sum() - 1) < 1e-12 and abs(r.x_ret.sum() - 1) < 1e-12


@pytest.mark.parametrize("pattern", PATTERNS)
def test_zero_stage_cut_limit_is_local_permeate(pattern):
    y_loc, _ = local_permeate(np.array(XB), BIN.Q, 4 * BAR, 1 * BAR)
    r = solve_stage(BIN, XB, 1e-5, 4.0, 1.0, pattern)
    assert r.y_perm[0] == pytest.approx(y_loc[0], abs=2e-5)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_selectivity_means_no_separation(pattern):
    m = Membrane({"A": 50, "B": 50})
    r = solve_stage(m, [0.3, 0.7], 0.5, 10.0, 1.0, pattern)
    assert r.y_perm[0] == pytest.approx(0.3, abs=1e-8)
    assert r.x_ret[0] == pytest.approx(0.3, abs=1e-8)


@pytest.mark.parametrize("theta", [0.2, 0.5, 0.8])
def test_flow_pattern_ranking(theta):
    """Permeate purity: counter-current >= cross-flow >= co-current >= well-mixed."""
    y = {p: solve_stage(BIN, XB, theta, 4.0, 1.0, p).y_perm[0] for p in PATTERNS}
    assert y["countercurrent"] >= y["cross_flow"] - 1e-9
    assert y["cross_flow"] >= y["cocurrent"] - 1e-9
    assert y["cocurrent"] >= y["well_mixed"] - 1e-9


def test_purity_falls_and_recovery_rises_with_stage_cut():
    for p in PATTERNS:
        res = [solve_stage(BIN, XB, th, 4.0, 1.0, p) for th in np.linspace(0.05, 0.9, 10)]
        y = [r.y_perm[0] for r in res]
        rec = [r.recovery(0) for r in res]
        assert np.all(np.diff(y) < 0) and np.all(np.diff(rec) > 0)


def test_cocurrent_pinches_at_pressure_ratio_limit():
    """In co-current flow the feed side cannot be stripped below roughly y/phi
    (the fast-gas driving force vanishes against the bulk permeate), whereas
    counter-current flow keeps stripping."""
    m, phi = Membrane.binary(200.0), 2.0
    co = cocurrent(m, [0.05, 0.95], 0.9, phi, 1.0)
    cc = countercurrent(m, [0.05, 0.95], 0.9, phi, 1.0)
    assert co.x_ret[0] >= 0.9 * co.y_perm[0] / phi
    assert cc.x_ret[0] < 0.1 * co.x_ret[0]


# ------------------------------------------- independent counter-current check
def _discretised_countercurrent(m, xf, area, ph, pl, cells=4000, iters=400):
    """First-order cell model iterated on the permeate profile (independent of
    the shooting code).  Returns (theta, y_perm)."""
    Q, n = m.Q, m.n
    dA = area / cells
    xf = np.asarray(xf, float)
    y_prof = np.tile(local_permeate(xf, Q, ph, pl)[0], (cells, 1))
    for _ in range(iters):
        N = xf.copy()
        J = np.empty((cells, n))
        for k in range(cells):
            x = np.clip(N, 0, None) / np.clip(N, 0, None).sum()
            J[k] = Q * (ph * x - pl * y_prof[k])
            N = N - J[k] * dA
        P = np.cumsum(J[::-1] * dA, axis=0)[::-1]      # permeate flows toward cell 0
        y_new = P / P.sum(axis=1, keepdims=True)
        y_new[-1] = local_permeate(np.clip(N, 0, None) / N.sum(), Q, ph, pl)[0]
        if np.max(np.abs(y_new - y_prof)) < 1e-11:
            break
        y_prof = 0.5 * y_prof + 0.5 * y_new
    return P[0].sum(), P[0] / P[0].sum()


def test_countercurrent_against_discretised_model():
    m = Membrane({"CO2": 100.0, "CH4": 4.0, "N2": 5.0})
    xf = m.vector({"CO2": 0.4, "CH4": 0.58, "N2": 0.02})
    r = countercurrent(m, xf, 0.45, 10.0, 1.0)
    th, y = _discretised_countercurrent(m, xf, r.area, 10 * BAR, 1 * BAR, cells=1500)
    assert th == pytest.approx(0.45, abs=2e-3)
    assert np.allclose(y, r.y_perm, atol=2e-3)


# ---------------------------------------------------------------- rating
@pytest.mark.parametrize("pattern", PATTERNS)
def test_rating_inverts_design(pattern):
    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
    d = solve_stage(m, xf, 0.4, 15.0, 1.0, pattern, 2.0)
    r = rate_module(m, xf, 2.0, 15.0, 1.0, d.area, pattern)
    assert r.theta == pytest.approx(0.4, abs=1e-6)
    assert np.allclose(r.y_perm, d.y_perm, atol=1e-6)


@pytest.mark.parametrize("pattern", ["cocurrent", "countercurrent"])
def test_bore_pressure_drop(pattern):
    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
    d = solve_stage(m, xf, 0.4, 15.0, 1.0, pattern, 2.0)
    thetas = []
    for di in [2e-3, 200e-6, 100e-6]:
        mod = HollowFibreModule.for_area(d.area, 1.0, di, 2 * di)
        r = rate_module(m, xf, 2.0, 15.0, 1.0, mod, pattern)
        assert r.mass_balance_error() < 1e-7
        assert r.extra["bore_dp_bar"] >= 0
        thetas.append(r.theta)
    assert thetas[0] == pytest.approx(0.4, abs=1e-3)         # wide bores: ~ideal
    assert thetas[0] > thetas[1] > thetas[2]                  # narrower: lower stage cut


# ---------------------------------------------------------------- flowsheets
def test_single_stage_flowsheet_equals_stage():
    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    xf = FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"]
    fs = single_stage(m, xf, 3.0, StageSpec(0.45, 15, 1))
    st = solve_stage(m, xf, 0.45, 15, 1, "countercurrent", 3.0)
    assert fs.purity("retentate 1", "CH4") == pytest.approx(st.x_ret[1], rel=1e-9)
    assert fs.total_area == pytest.approx(st.area, rel=1e-9)


@pytest.mark.parametrize("builder,args", [
    (two_stage_enriching, (StageSpec(0.5, 15, 1, "cross_flow"), StageSpec(0.6, 15, 1, "cross_flow"))),
    (two_stage_stripping, (StageSpec(0.4, 15, 1, "cross_flow"), StageSpec(0.15, 15, 1, "cross_flow"))),
    (three_stage, (StageSpec(0.35, 15, 4, "cross_flow"), StageSpec(0.15, 15, 1, "cross_flow"),
                   StageSpec(0.7, 4, 1, "cross_flow"))),
])
def test_flowsheets_converge_and_balance(builder, args):
    m = Membrane(MEMBRANE_PRESETS["Polyimide CO2/CH4, a=35 (Makaruk 2010)"])
    fs = builder(m, FEED_PRESETS["biogas, 60% CH4 (Makaruk 2010)"], 3.0, *args)
    assert fs.converged
    assert fs.mass_balance_error() < 1e-8
    assert fs.total_power_kW > 0


def test_compression_power_single_stage_formula():
    n, k = 1.0, 1.4
    W = compression_power(n, 1.0, 3.0, k, T_REF, eta=1.0, max_ratio=4.0)
    assert W == pytest.approx(n * R * T_REF * k / (k - 1) * (3.0 ** ((k - 1) / k) - 1), rel=1e-12)
    # intercooled two-stage needs less than single-stage for the same ratio
    assert compression_power(n, 1, 15, k, max_ratio=4) < compression_power(n, 1, 15, k, max_ratio=20)
