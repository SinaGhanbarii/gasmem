"""Model 1 - complete mixing (well-mixed) on both sides of the membrane.

Both sides are perfectly mixed, so the whole membrane sees the *outlet*
compositions: retentate x and permeate y.

Binary closed form
------------------
With alpha = Q_A/Q_B and phi = p_h/p_l,

    y/(1-y) = alpha (x - y/phi) / ((1-x) - (1-y)/phi)        (rate equation)
    x_F     = theta y + (1-theta) x                          (mass balance)

Eliminating x gives a quadratic  a y^2 + b y + c = 0  (coefficients in
`well_mixed_binary`), so y, x and the membrane area follow without iteration.

Multicomponent
--------------
Combining the component balance with the rate equation gives

    y_i = [Q_i p_h x_F,i / (1-theta)] / [Jt + Q_i p_l + Q_i p_h theta/(1-theta)]

where Jt = theta F / A is the total flux.  sum(y) = 1 has one positive root
in Jt, so the multicomponent case reduces to a single scalar equation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .core import InfeasibleError, Membrane, StageResult, check_inputs


def well_mixed_binary(alpha: float, phi: float, x_feed: float, theta: float):
    """Closed-form binary well-mixed solution (dimensionless).

    Parameters
    ----------
    alpha : selectivity of the fast gas A over B
    phi   : pressure ratio p_h/p_l (np.inf allowed: zero permeate pressure)
    x_feed: feed mole fraction of A
    theta : stage cut

    Returns
    -------
    dict with y (permeate A), x (retentate A), area_dimless = A Q_A p_h / F
    """
    if not (0 < theta < 1):
        raise ValueError("theta must be in (0, 1)")
    r = 0.0 if np.isinf(phi) else 1.0 / phi
    # x = a0 + a1 y from the mass balance
    a0 = x_feed / (1.0 - theta)
    a1 = -theta / (1.0 - theta)
    # rate equation  y[(1-x) - (1-y) r] - alpha (1-y)(x - y r) = 0, expanded:
    #   y(1 - a0 - r) + y^2(r - a1) - alpha[(a0) + y(a1 - r - a0) + y^2(r - a1)] = 0
    c2 = (r - a1) * (1.0 - alpha)
    c1 = (1.0 - a0 - r) - alpha * (a1 - r - a0)
    c0 = -alpha * a0
    roots = np.roots([c2, c1, c0]) if abs(c2) > 1e-300 else np.array([-c0 / c1])
    for y in np.real(roots[np.abs(np.imag(roots)) < 1e-12]):
        x = a0 + a1 * y
        if -1e-12 <= y <= 1 + 1e-12 and -1e-12 <= x <= 1 + 1e-12:
            dfA, dfB = x - y * r, (1 - x) - (1 - y) * r
            if dfA >= -1e-12 and dfB >= -1e-12:
                y, x = float(np.clip(y, 0, 1)), float(np.clip(x, 0, 1))
                # total flux / (Q_A p_h) = (x - r y)/y  (from J_A = y Jt)
                jt = (x - r * y) / y if y > 0 else np.nan
                return {"y": y, "x": x, "area_dimless": theta / jt if jt > 0 else np.inf,
                        "coefficients": (c2, c1, c0)}
    raise InfeasibleError("no physical root for these conditions")


def well_mixed(membrane: Membrane, x_feed, theta: float, p_high: float, p_low: float,
               feed_flow: float = 1.0) -> StageResult:
    """Multicomponent well-mixed stage at a specified stage cut.

    Pressures in bar, feed_flow in mol/s.  Returns a StageResult (area in m2).
    """
    xF, ph, pl = check_inputs(membrane, x_feed, theta, p_high, p_low)
    Q = membrane.Q
    c = Q * ph * xF / (1.0 - theta)
    d = Q * pl + Q * ph * theta / (1.0 - theta)
    f = lambda J: np.sum(c / (J + d)) - 1.0
    hi = c.sum()                       # f(hi) < 0 because d > 0
    if f(0.0) <= 0:
        raise InfeasibleError("no driving force")
    Jt = brentq(f, 0.0, hi, xtol=1e-15 * hi, rtol=4e-15)
    y = c / (Jt + d)
    y /= y.sum()
    x = (xF - theta * y) / (1.0 - theta)
    x = np.clip(x, 0.0, None)
    x /= x.sum()
    area = theta * feed_flow / Jt
    return StageResult("well_mixed", membrane.components, theta, feed_flow, xF, x, y,
                       area, p_high, p_low, extra={"total_flux": Jt})
