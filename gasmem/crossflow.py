"""Model 2 - cross-flow (feed in plug flow, permeate leaves each element
without mixing along the module).

Numerical model (any number of components)
------------------------------------------
Using the retentate flow L as the independent variable (L runs from F down
to (1-theta)F), each element releases permeate of the *local* composition
y(x) given by `local_permeate`:

    dN_i/dL = y_i            dA/dL = -1/Jt

so the stage cut is imposed directly and the membrane area is an output.

Weller-Steiner analytical solution (binary)
-------------------------------------------
Weller & Steiner (1950) integrated the binary cross-flow equations in closed
form; the form below follows Geankoplis, *Transport Processes and Separation
Process Principles*, section 13.7.  It is given the retentate composition x_o
and returns the stage cut; `weller_steiner_theta` inverts it for a target
stage cut.  The numerical model reproduces it (see tests/).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .core import Membrane, StageResult, check_inputs, local_permeate


def cross_flow(membrane: Membrane, x_feed, theta: float, p_high: float, p_low: float,
               feed_flow: float = 1.0, profile: bool = True, rtol: float = 1e-9) -> StageResult:
    """Cross-flow stage at a specified stage cut (pressures in bar)."""
    xF, ph, pl = check_inputs(membrane, x_feed, theta, p_high, p_low)
    Q, n, F = membrane.Q, membrane.n, feed_flow

    def rhs(L, s):
        N = np.clip(s[:n], 0.0, None)
        y, Jt = local_permeate(N / N.sum(), Q, ph, pl)
        return np.concatenate([y, [-1.0 / Jt]])

    s0 = np.concatenate([F * xF, [0.0]])
    sol = solve_ivp(rhs, (F, (1.0 - theta) * F), s0, method="LSODA", rtol=rtol,
                    atol=1e-12 * F, dense_output=profile)
    if not sol.success:
        raise RuntimeError(sol.message)
    N_end = np.clip(sol.y[:n, -1], 0.0, None)
    x_ret = N_end / N_end.sum()
    perm = F * xF - N_end
    y_perm = perm / perm.sum()
    area = float(sol.y[n, -1])

    prof = None
    if profile:
        L = np.linspace(F, (1.0 - theta) * F, 200)
        S = sol.sol(L)
        N = np.clip(S[:n], 0, None)
        x = N / N.sum(axis=0)
        y_loc = np.array([local_permeate(x[:, k], Q, ph, pl)[0] for k in range(L.size)]).T
        P_cum = F * xF[:, None] - N
        with np.errstate(invalid="ignore", divide="ignore"):
            y_cum = P_cum / P_cum.sum(axis=0)
        y_cum[:, 0] = y_loc[:, 0]
        prof = {"area_fraction": S[n] / area, "theta_local": (F - L) / F,
                "x": x, "y_local": y_loc, "y_collected": y_cum}
    return StageResult("cross_flow", membrane.components, theta, F, xF, x_ret, y_perm,
                       area, p_high, p_low, profile=prof)


# ---------------------------------------------------------------------------
# Weller-Steiner closed form (binary)
# ---------------------------------------------------------------------------
def _ws_constants(alpha: float, r: float):
    D = 0.5 * ((1.0 - alpha) * r + alpha)
    Fc = -0.5 * ((1.0 - alpha) * r - 1.0)
    E = alpha / 2.0 - D * Fc
    R = 1.0 / (2.0 * D - 1.0)
    S = (alpha * (D - 1.0) + Fc) / ((2.0 * D - 1.0) * (alpha / 2.0 - Fc))
    T = 1.0 / (1.0 - D - E / Fc)
    return D, E, Fc, R, S, T


def weller_steiner(alpha: float, phi: float, x_feed: float, x_ret: float):
    """Binary cross-flow: stage cut and permeate composition for a given
    retentate composition x_ret (mole fraction of the fast gas A).

    Returns (theta, y_permeate).
    """
    r = 1.0 / phi
    D, E, Fc, R, S, T = _ws_constants(alpha, r)

    def u(x):
        i = x / (1.0 - x)
        return -D * i + np.sqrt(D * D * i * i + 2.0 * E * i + Fc * Fc)

    uf, uo = u(x_feed), u(x_ret)
    rhs = (((uf - E / D) / (uo - E / D)) ** R
           * ((uf - alpha + Fc) / (uo - alpha + Fc)) ** S
           * ((uf - Fc) / (uo - Fc)) ** T)
    theta = 1.0 - rhs * (1.0 - x_feed) / (1.0 - x_ret)
    y = (x_feed - x_ret * (1.0 - theta)) / theta
    return float(theta), float(y)


def weller_steiner_theta(alpha: float, phi: float, x_feed: float, theta: float):
    """Invert `weller_steiner` for a target stage cut.  Returns (x_ret, y_perm)."""
    g = lambda xo: weller_steiner(alpha, phi, x_feed, xo)[0] - theta
    hi = x_feed * (1.0 - 1e-12)
    lo = x_feed * 1e-9
    while g(lo) < 0 and lo > 1e-300:     # theta too large for lo: go lower
        lo *= 1e-3
    xo = brentq(g, lo, hi, xtol=1e-15, rtol=1e-13)
    return float(xo), weller_steiner(alpha, phi, x_feed, xo)[1]
