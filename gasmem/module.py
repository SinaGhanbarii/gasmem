"""Model 4a - rating a hollow-fibre module of fixed size, with optional
permeate-side (bore) pressure drop.

In *design* mode (wellmixed.py, crossflow.py, hollowfiber.py) the stage cut is
specified and the area is the answer.  Here the module is fixed and the stage
cut is the *result*.  This is how stage cut is changed in practice: by changing
the feed flow (or the pressures) to a module that already exists.

Configuration
-------------
Feed on the shell side, permeate inside the fibre bores, sweepless.  The bore
is closed at one end and open at the other.  Bore pressure drop follows
Hagen-Poiseuille for a compressible ideal gas,

    dp/dx = -128 mu R T q / (pi d_i^4 n_f p)

where q is the molar permeate flow carried by all fibres at that point.  With
z = x / L the active-length fraction, |dp/dz| = K mu q / p, K = 128 R T L /
(pi d_i^4 n_f).  The mixture viscosity is evaluated locally (Herning-Zipperer).
Feed-side (shell) pressure drop is neglected.

Equations in z (A = total area):
    dN_i/dz = -A J_i
    co-current      dP_i/dz = +A J_i     dp/dz = -K mu sum(P)/p   (outlet z=1)
    counter-current dP_i/dz = -A J_i     dp/dz = +K mu sum(P)/p   (outlet z=0)
Counter-current is solved by shooting from the closed end (z = 1): unknown
retentate composition, retentate flow and closed-end bore pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, root

from .core import InfeasibleError, Membrane, StageResult, local_permeate
from .crossflow import cross_flow
from .hollowfiber import _perm_comp
from .properties import BAR, R, T_REF, mixture_viscosity
from .wellmixed import well_mixed


_DEBUG = False
_DZ = 1e-7   # length fraction of the analytic first element at the closed end


@dataclass
class HollowFibreModule:
    """Geometry of a hollow-fibre module (SI units: m)."""

    n_fibres: float
    length: float = 1.0          # active length, m
    d_inner: float = 200e-6      # bore diameter, m
    d_outer: float = 400e-6      # outer diameter, m (area basis: outer skin)

    @property
    def area(self) -> float:
        return float(self.n_fibres * np.pi * self.d_outer * self.length)

    def bore_constant(self, T: float = T_REF) -> float:
        """K in |dp/dz| = K mu q / p  (Pa^2 per (Pa s * mol/s))."""
        return 128.0 * R * T * self.length / (np.pi * self.d_inner ** 4 * self.n_fibres)

    @classmethod
    def for_area(cls, area: float, length: float = 1.0, d_inner: float = 200e-6,
                 d_outer: float = 400e-6) -> "HollowFibreModule":
        """Module with the given outer area, fibre length and diameters."""
        return cls(area / (np.pi * d_outer * length), length, d_inner, d_outer)

    def __repr__(self) -> str:
        return (f"HollowFibreModule(area={self.area:.4g} m2, n_fibres={self.n_fibres:.4g}, "
                f"L={self.length:g} m, d_i={self.d_inner*1e6:.0f} um, "
                f"d_o={self.d_outer*1e6:.0f} um)")


def rate_module(membrane: Membrane, x_feed, feed_flow: float, p_high: float,
                p_perm_outlet: float, module: HollowFibreModule | float,
                pattern: str = "countercurrent", pressure_drop: bool = True,
                T: float = T_REF, profile: bool = True, rtol: float = 1e-8) -> StageResult:
    """Performance of a fixed module.  The stage cut is an output.

    Parameters
    ----------
    module : HollowFibreModule, or a plain area in m2 (then no pressure drop)
    pattern : 'well_mixed', 'cross_flow', 'cocurrent' or 'countercurrent'
    pressure_drop : include bore-side pressure drop (co/counter-current only)
    p_perm_outlet : permeate pressure at the bore outlet, bar
    """
    xF = membrane.vector(x_feed)
    if isinstance(module, (int, float)):
        A, K, pressure_drop = float(module), 0.0, False
    else:
        A, K = module.area, module.bore_constant(T)
    if not pressure_drop:
        K = 0.0
    Q, n, F = membrane.Q, membrane.n, feed_flow
    ph, pout = p_high * BAR, p_perm_outlet * BAR
    names = membrane.components

    # -- well-mixed and cross-flow: invert the design models ---------------
    if pattern in ("well_mixed", "cross_flow"):
        if pattern == "cross_flow":
            return _rate_cross_flow(membrane, xF, F, p_high, p_perm_outlet, A, profile, rtol)
        f = lambda th: well_mixed(membrane, xF, th, p_high, p_perm_outlet, F).area - A
        th = brentq(f, 1e-9, 1 - 1e-9, xtol=1e-13)
        res = well_mixed(membrane, xF, th, p_high, p_perm_outlet, F)
        res.extra["rating"] = True
        return res

    def mu(y):
        return mixture_viscosity(names, y)

    sgn = 1.0 if pattern == "cocurrent" else -1.0
    p_of_z = [None]   # counter-current: prescribed bore-pressure profile

    def rhs(z, s):
        N, P = s[:n], s[n:2 * n]
        if p_of_z[0] is not None:
            p, dp = p_of_z[0](z), 0.0
        else:
            p = s[2 * n]
        x, y = _perm_comp(N, P, Q, ph, p, F)
        J = Q * (ph * x - p * y)
        if p_of_z[0] is None:
            Ps = max(P.sum(), 0.0)
            dp = -sgn * K * mu(y) * Ps / max(p, 1e-6 * pout) if K > 0 else 0.0
        return np.concatenate([-A * J, sgn * A * J, [dp]])

    def depleted(z, s):
        # stop if a feed flow is exhausted or the bore pressure collapses
        # (the guess is then unphysical; the root finders treat it as such)
        return min(np.min(s[:n]) / F, s[2 * n] / pout - 0.25)
    depleted.terminal = True

    def seed(N, p, sign):
        """First element next to the closed bore end, released at the local
        (cross-flow) composition; removes the 0/0 start of y = P/sum(P)."""
        y, Jt = local_permeate(N / N.sum(), Q, ph, p)
        dP = y * Jt * A * _DZ
        return np.concatenate([N + sign * dP, dP, [p]])

    if pattern == "cocurrent":
        def run(p0, dense=False):
            s0 = seed(F * xF, p0, -1.0)
            return solve_ivp(rhs, (_DZ, 1.0), s0, method="RK45", rtol=rtol,
                             atol=1e-14 * F, events=depleted, dense_output=dense)

        if K == 0.0:
            p0 = pout
        else:
            def g(p0):
                sol = run(p0)
                return (sol.y[2 * n, -1] if sol.status == 0 else -pout) - pout
            # expand the bracket upwards from the outlet pressure; values close to
            # p_high are avoided because the flux (and the ODE) becomes stiff there
            lo, hi = pout, pout * 1.05
            while g(hi) < 0:
                lo, hi = hi, pout + 2.0 * (hi - pout)
                if hi > 0.98 * ph:
                    raise InfeasibleError("bore pressure drop too large for this module")
            p0 = brentq(g, lo, hi, xtol=1e-10 * pout)
        sol = run(p0, profile)
        if sol.status != 0:
            raise InfeasibleError("a feed component was exhausted inside the module")
        N_end, P_end = sol.y[:n, -1], sol.y[n:2 * n, -1]
        p_closed = p0
        z_feed_end = 0.0
    elif pattern == "countercurrent":
        # Outer loop: successive substitution on the bore-pressure profile.
        # Inner problem: counter-current shooting with that profile fixed.
        # With p(z) known the bore momentum balance integrates in closed form,
        #     p(z)^2 = p_out^2 + 2 K int_0^z mu(y) sum(P) dz'.
        zg = np.linspace(0.0, 1.0, 201)
        pg = np.full_like(zg, pout)
        spline = [CubicSpline(zg, pg)]       # smooth p(z) keeps the shooting accurate
        p_of_z[0] = lambda z: float(spline[0](z))

        def run(xR, LR, pc=None, dense=False):
            s0 = seed(LR * xR, p_of_z[0](1.0), +1.0)
            return solve_ivp(rhs, (1.0 - _DZ, 0.0), s0, method="RK45", rtol=rtol,
                             atol=1e-14 * F, events=depleted, dense_output=dense)

        cf = _rate_cross_flow(membrane, xF, F, p_high, p_perm_outlet, A, False, rtol)
        xR, LR = cf.x_ret, cf.retentate_flow
        omega, last = 0.7, np.inf
        for it in range(200):
            xR, LR = _shoot_cc_rating(run, xF, F, n, xR, LR)
            if K == 0.0:
                break
            sol = run(xR, LR, dense=True)
            S = sol.sol(np.clip(zg, 0.0, 1.0 - _DZ))
            Ps = np.clip(S[n:2 * n].sum(axis=0), 0, None)
            mug = np.array([mu(_perm_comp(S[:n, k], S[n:2 * n, k], Q, ph, pg[k], F)[1])
                            for k in range(zg.size)])
            integrand = mug * Ps
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                                    * np.diff(zg))])
            p_new = np.sqrt(pout ** 2 + 2.0 * K * cum)
            if p_new[-1] > 0.98 * ph:
                raise InfeasibleError("bore pressure drop too large for this module")
            change = np.max(np.abs(p_new - pg)) / pout
            if _DEBUG: print("iter", it, change, omega, p_new[-1] / BAR, xR[0], LR)
            if change > last:              # oscillating: damp harder
                omega = max(0.2, 0.5 * omega)
            last = change
            pg[:] = pg + omega * (p_new - pg)
            spline[0] = CubicSpline(zg, pg)
            if change < 1e-8:
                break
        else:
            raise InfeasibleError("counter-current rating: bore-pressure iteration did not converge")
        sol = run(xR, LR, dense=profile)
        N_end, P_end = LR * xR, sol.y[n:2 * n, -1]
        p_closed = pg[-1]
        z_feed_end = 1.0
    else:
        raise ValueError(f"unknown flow pattern {pattern!r}")

    x_ret = np.clip(N_end, 0, None); x_ret /= x_ret.sum()
    y_perm = np.clip(P_end, 0, None); y_perm /= y_perm.sum()
    theta = float(P_end.sum() / F)
    prof = None
    if profile:
        z = np.linspace(_DZ, 1.0 - _DZ, 200)
        S = sol.sol(z)
        pz = (np.array([p_of_z[0](zz) for zz in z]) if p_of_z[0] is not None
              else S[2 * n])
        xs, ys = zip(*[_perm_comp(S[:n, k], S[n:2 * n, k], Q, ph, pz[k], F)
                       for k in range(z.size)])
        prof = {"area_fraction": z, "x": np.array(xs).T, "y": np.array(ys).T,
                "permeate_flow": S[n:2 * n].sum(axis=0),
                "p_perm_bar": pz / BAR}
    res = StageResult(pattern, names, theta, F, xF, x_ret, y_perm, A, p_high,
                      p_perm_outlet, profile=prof,
                      extra={"rating": True, "pressure_drop": bool(K > 0),
                             "p_perm_closed_end_bar": p_closed / BAR,
                             "bore_dp_bar": (p_closed - pout) / BAR})
    return res


def _rate_cross_flow(membrane, xF, F, p_high, p_low, A, profile, rtol):
    Q, n, ph, pl = membrane.Q, membrane.n, p_high * BAR, p_low * BAR

    def rhs(z, N):
        x = np.clip(N, 0, None)
        if x.sum() <= 1e-12 * F:          # feed exhausted: nothing left to permeate
            return np.zeros_like(N)
        y, Jt = local_permeate(x / x.sum(), Q, ph, pl)
        return -A * y * Jt

    def gone(z, N):
        return N.sum() - 1e-9 * F
    gone.terminal = True
    sol = solve_ivp(rhs, (0, 1), F * xF, method="RK45", rtol=rtol, atol=1e-14 * F,
                    events=gone, dense_output=profile)
    if sol.status != 0:
        raise InfeasibleError("module area exceeds what the feed can supply")
    N = sol.y[:, -1]
    th = 1 - N.sum() / F
    x_ret = N / N.sum()
    perm = F * xF - N
    prof = None
    if profile:
        z = np.linspace(0, 1, 200)
        Nz = sol.sol(z)
        x = Nz / Nz.sum(axis=0)
        prof = {"area_fraction": z, "x": x,
                "y_local": np.array([local_permeate(x[:, k], Q, ph, pl)[0]
                                     for k in range(z.size)]).T}
    return StageResult("cross_flow", membrane.components, float(th), F, xF, x_ret,
                       perm / perm.sum(), A, p_high, p_low, profile=prof,
                       extra={"rating": True})


def _shoot_cc_rating(run, xF, F, n, xR0, LR0):
    """Counter-current rating shooting for a given bore-pressure profile.
    Unknowns: log-ratios of x_R (relative to its most abundant species) and
    ln(L_R/F).  Residuals: feed-end component flows."""
    ref = int(np.argmax(xR0))
    idx = [i for i in range(n) if i != ref]
    floor = 1e-300

    def unpack(v):
        w = np.ones(n)
        w[idx] = np.exp(np.clip(v[:n - 1], -690, 690))
        return w / w.sum(), F * np.exp(min(v[n - 1], 0.0))

    def resid(v):
        xR, LR = unpack(v)
        sol = run(xR, LR)
        if sol.status != 0:
            return np.full(n, 1e3)
        return sol.y[:n, -1] / F / np.maximum(xF, floor) - 1.0

    v0 = np.concatenate([np.log(np.maximum(xR0[idx], floor) / xR0[ref]),
                         [np.log(LR0 / F)]])
    best = None
    for method, opts in (("hybr", {"xtol": 1e-12}), ("lm", {"xtol": 1e-13})):
        res = root(resid, v0, method=method, options=opts)
        err = np.max(np.abs(resid(res.x)))
        if best is None or err < best[1]:
            best = (res.x, err)
        if err < 1e-8:
            break
    if best[1] > 1e-6:
        raise InfeasibleError("counter-current rating: shooting failed")
    return unpack(best[0])


def feed_flow_for_stage_cut(membrane: Membrane, x_feed, theta: float, p_high: float,
                            p_perm_outlet: float, module, pattern: str = "countercurrent",
                            pressure_drop: bool = True, T: float = T_REF) -> StageResult:
    """Feed flow that gives the target stage cut in a fixed module."""
    area = module.area if isinstance(module, HollowFibreModule) else float(module)
    # initial bracket from the ideal (no pressure drop) design area
    from .hollowfiber import countercurrent as _cc
    A1 = _cc(membrane, x_feed, theta, p_high, p_perm_outlet, 1.0, profile=False).area
    F0 = area / A1
    def g(lnF):
        try:
            return rate_module(membrane, x_feed, np.exp(lnF), p_high, p_perm_outlet, module,
                               pattern, pressure_drop, T, profile=False).theta - theta
        except (InfeasibleError, ValueError, FloatingPointError):
            return 1.0 - theta          # feed flow so small that the module exhausts it
    a, b = np.log(F0) - 0.5, np.log(F0) + 0.5
    while g(a) < 0:
        a -= 1.0
    while g(b) > 0:
        b += 1.0
    lnF = brentq(g, a, b, xtol=1e-10)
    return rate_module(membrane, x_feed, np.exp(lnF), p_high, p_perm_outlet, module,
                       pattern, pressure_drop, T)
