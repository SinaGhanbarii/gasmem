"""Model 3 - co-current and counter-current plug flow (sweepless hollow-fibre
module), any number of components.

Both sides are in plug flow.  The permeate side starts with zero flow at its
closed end; there the permeate composition equals the local (cross-flow) value.

Co-current (design mode, theta given)
    Permeate flows with the feed.  An initial-value problem in the retentate
    flow L from F to (1-theta)F:
        dN_i/dL = J_i/Jt,   dP_i/dL = -J_i/Jt,   dA/dL = -1/Jt
    with y = P/sum(P).  If the total flux falls to zero before the requested
    stage cut is reached, the stage cut is infeasible (pressure-ratio limit).

Counter-current (design mode, theta given)
    Permeate flows against the feed and leaves at the feed end, so it is a
    two-point boundary-value problem.  It is solved by *shooting from the
    closed (retentate) end*: guess the retentate composition x_R, start with
    N = (1-theta)F x_R and P = 0, integrate back to L = F and adjust x_R until
    the feed composition is recovered.  Binary cases use a bracketed scalar
    root (Brent); multicomponent cases use log-ratio unknowns and a Newton-type
    solver started from the cross-flow solution.

Permeate sweep (optional, both patterns)
    A sweep gas of flow S and composition z enters the permeate channel at its
    otherwise closed end (feed end for co-current, retentate end for
    counter-current).  The permeate side then starts at P = S z instead of 0,
    so no seed element is needed.  The stage cut keeps its meaning as the
    fraction of the feed that crosses the membrane, theta = 1 - L_R/F; the
    permeate outlet carries theta F + S.  Components of the sweep that are
    absent from the feed may permeate backwards into the retentate.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from .core import InfeasibleError, Membrane, StageResult, check_inputs, local_permeate
from .crossflow import cross_flow

_METHOD = "RK45"  # non-stiff once the closed end is seeded; LSODA crawls here
_SEED = 1e-7   # first analytic element, as a fraction of the permeate flow


def _perm_comp(N, P, Q, ph, pl, F):
    """Feed-side x and permeate-side (bulk) y at a point."""
    x = np.clip(N, 0.0, None)
    x = x / x.sum()
    y = np.clip(P, 0.0, None)
    ys = y.sum()
    if ys <= 0.0:                     # exactly at a closed end
        return x, local_permeate(x, Q, ph, pl)[0]
    return x, y / ys


def _sweep_vector(membrane, sweep_flow, sweep_composition, F):
    """Sweep component flows (mol/s) or None."""
    if not sweep_flow:
        return None
    if sweep_flow < 0:
        raise ValueError("sweep_flow must be non-negative")
    if sweep_composition is None:
        raise ValueError("give sweep_composition with sweep_flow, e.g. {'N2': 1.0}")
    return sweep_flow * membrane.vector(sweep_composition)


def _start(N, Q, ph, pl, dL, sign, sweep):
    """Start of the integration at the permeate inlet: the analytic seed for a
    closed end, or the sweep flows.  Returns (state0, dL used)."""
    if sweep is None:
        N0, P0, A0 = _seed(N, Q, ph, pl, dL, sign)
        return np.concatenate([N0, P0, [A0]]), dL
    return np.concatenate([N, sweep, [0.0]]), 0.0


def _seed(N, Q, ph, pl, dL, sign):
    """Analytic first element next to the closed permeate end.

    Over a tiny retentate change dL the permeate released has the local
    (cross-flow) composition, so the singular start P = 0 is replaced by
    P = y_local dL.  This removes the 0/0 in y = P/sum(P) and keeps the
    integration smooth.  Returns (N_new, P_new, dA)."""
    x = N / N.sum()
    y, Jt = local_permeate(x, Q, ph, pl)
    return N + sign * y * dL, y * dL, dL / Jt


# ---------------------------------------------------------------------------
# co-current
# ---------------------------------------------------------------------------
def cocurrent(membrane: Membrane, x_feed, theta: float, p_high: float, p_low: float,
              feed_flow: float = 1.0, profile: bool = True, rtol: float = 1e-9,
              sweep_flow: float = 0.0, sweep_composition=None) -> StageResult:
    """Co-current plug flow at a specified stage cut (pressures in bar).

    sweep_flow (mol/s) and sweep_composition (mapping) add a permeate sweep
    entering at the feed end."""
    xF, ph, pl = check_inputs(membrane, x_feed, theta, p_high, p_low)
    Q, n, F = membrane.Q, membrane.n, feed_flow
    sweep = _sweep_vector(membrane, sweep_flow, sweep_composition, F)
    jscale = Q.max() * ph

    def fluxes(s):
        x, y = _perm_comp(s[:n], s[n:2 * n], Q, ph, pl, F)
        J = Q * (ph * x - pl * y)
        return J, J.sum(), x, y

    def rhs(L, s):
        J, Jt, _, _ = fluxes(s)
        Jt = max(Jt, 1e-14 * jscale)
        return np.concatenate([J / Jt, -J / Jt, [-1.0 / Jt]])

    def no_flux(L, s):
        return fluxes(s)[1] - 1e-9 * jscale
    no_flux.terminal, no_flux.direction = True, -1

    s0, dL = _start(F * xF, Q, ph, pl, _SEED * theta * F, -1.0, sweep)
    sol = solve_ivp(rhs, (F - dL, (1 - theta) * F), s0, method=_METHOD, rtol=rtol,
                    atol=1e-14 * F, events=no_flux, dense_output=profile)
    if sol.status == 1:
        th_max = (F - sol.t[-1]) / F
        raise InfeasibleError(f"co-current: total flux vanishes at stage cut {th_max:.4f} "
                              f"(< requested {theta:.4f}); pressure-ratio limited")
    if not sol.success:
        raise RuntimeError(sol.message)
    N_end, P_end, area = sol.y[:n, -1], sol.y[n:2 * n, -1], float(sol.y[2 * n, -1])
    x_ret = np.clip(N_end, 0, None); x_ret /= x_ret.sum()
    y_perm = np.clip(P_end, 0, None); y_perm /= y_perm.sum()

    prof = None
    if profile:
        L = np.linspace(F - dL, (1 - theta) * F, 200)
        S = sol.sol(L)
        xs, ys = zip(*[fluxes(S[:, k])[2:] for k in range(L.size)])
        prof = {"area_fraction": S[2 * n] / area, "theta_local": (F - L) / F,
                "x": np.array(xs).T, "y": np.array(ys).T,
                "permeate_flow": S[n:2 * n].sum(axis=0)}
    return StageResult("cocurrent", membrane.components, theta, F, xF, x_ret, y_perm,
                       area, p_high, p_low, profile=prof,
                       sweep_flow=float(sweep_flow or 0.0),
                       sweep_x=None if sweep is None else sweep / sweep.sum())


# ---------------------------------------------------------------------------
# counter-current
# ---------------------------------------------------------------------------
def _cc_shoot(xR, Q, ph, pl, F, theta, n, dense=False, rtol=1e-9, sweep=None, check=None):
    """Integrate from the closed (retentate) end back to the feed end."""
    jscale = Q.max() * ph

    def fluxes(s):
        x, y = _perm_comp(s[:n], s[n:2 * n], Q, ph, pl, F)
        J = Q * (ph * x - pl * y)
        return J, J.sum(), x, y

    def rhs(L, s):
        J, Jt, _, _ = fluxes(s)
        Jt = max(Jt, 1e-14 * jscale)
        return np.concatenate([J / Jt, J / Jt, [1.0 / Jt]])

    def no_flux(L, s):
        return fluxes(s)[1] - 1e-9 * jscale
    no_flux.terminal, no_flux.direction = True, -1

    chk = slice(None) if check is None else check

    def depleted(L, s):              # a feed-side flow would turn negative:
        return np.min(s[:n][chk])    # the guessed x_R is unphysical
    depleted.terminal, depleted.direction = True, -1

    LR = (1 - theta) * F
    s0, dL = _start(LR * xR, Q, ph, pl, _SEED * theta * F, +1.0, sweep)
    opts = {"atol": 1e-300 * F}
    if sweep is not None:      # area starts at exactly 0: own tolerance, explicit first step
        opts = {"atol": np.r_[np.full(2 * n, 1e-300 * F), 1e-12],
                "first_step": 1e-8 * (F - LR)}
    sol = solve_ivp(rhs, (LR + dL, F), s0, method=_METHOD, rtol=rtol,
                    events=(no_flux, depleted), dense_output=dense, **opts)
    ok = sol.success and sol.status == 0
    sol.depleted = (int(np.argmin(sol.y[:n, -1]))
                    if sol.status == 1 and sol.t_events[1].size else None)
    return sol, ok, fluxes


def _bracket_increasing(g, t0, step=1.0, maxit=80):
    """Bracket the root of an increasing function g starting from t0.
    nan values (failed integrations) are stepped over towards t0."""
    g0 = g(t0)
    if not np.isfinite(g0):
        for d in (0.1, -0.1, 0.5, -0.5, 2, -2):
            g0 = g(t0 + d)
            if np.isfinite(g0):
                t0 += d
                break
        else:
            raise InfeasibleError("counter-current: shooting failed near the initial guess")
    if g0 == 0:
        return t0, t0, 0.0, 0.0
    direction = -1.0 if g0 > 0 else 1.0
    t_prev, g_prev = t0, g0
    for _ in range(maxit):
        t_new = t_prev + direction * step
        g_new = g(t_new)
        if not np.isfinite(g_new):
            step *= 0.5
            continue
        if g_new * g_prev <= 0:
            a, b = sorted((t_prev, t_new))
            ga, gb = (g_prev, g_new) if t_prev < t_new else (g_new, g_prev)
            return a, b, ga, gb
        t_prev, g_prev = t_new, g_new
        step *= 2.0
    raise InfeasibleError("counter-current: could not bracket the retentate composition")


def _cc_multicomponent(membrane, xF, theta, p_high, p_low, F, guess, rtol, sweep=None):
    """Shooting for n > 2: unknowns u_i = ln(x_R,i / x_R,ref), where ref is the
    most abundant retentate species of the guess.  If a direct solve fails
    (typically at very high stage cut, when a fast gas is stripped to trace
    level), the stage cut is approached by continuation from a lower value."""
    Q, n = membrane.Q, membrane.n
    from .properties import BAR
    ph, pl = p_high * BAR, p_low * BAR
    floor = 1e-300
    # a species that enters only with the sweep must reach exactly zero at the
    # feed end; it may cross zero during the shooting without ending it
    check = (xF > 1e-14) if sweep is not None else None

    def solve_at(th, gx):
        ref = int(np.argmax(gx))
        idx = [i for i in range(n) if i != ref]

        def to_x(u):
            v = np.zeros(n)
            v[idx] = np.clip(u, -690, 690)
            w = np.exp(v - v.max())          # overflow-safe normalisation
            return w / w.sum()

        def resid(u):
            sol, ok, _ = _cc_shoot(to_x(u), Q, ph, pl, F, th, n, rtol=rtol, sweep=sweep,
                                   check=check)
            if not ok:
                return np.full(n - 1, 1e3)
            Nf = sol.y[:n, -1] / F
            # a component absent from the feed (it can enter from the sweep)
            # is matched in absolute terms
            out = np.where(xF[idx] > 1e-14, Nf[idx] / np.maximum(xF[idx], floor) - 1.0,
                           (Nf[idx] - xF[idx]) * 1e3)
            return out if np.all(np.isfinite(out)) else np.full(n - 1, 1e3)

        u0 = np.log(np.maximum(gx[idx], floor) / gx[ref])
        best = None
        for method, opts in (("hybr", {"xtol": 1e-11}),
                             ("lm", {"xtol": 1e-12, "ftol": 1e-14})):
            res = root(resid, u0, method=method, options=opts)
            err = np.max(np.abs(resid(res.x)))
            if best is None or err < best[1]:
                best = (to_x(res.x), err)
            if err < 1e-8:
                break
        return best

    xR, err = solve_at(theta, guess)
    if err < 1e-7:
        return xR
    # continuation in stage cut from a comfortable starting point
    th_list = np.linspace(min(0.3, 0.5 * theta), theta, 8)
    gx = cross_flow(membrane, xF, th_list[0], p_high, p_low, F, profile=False).x_ret
    if sweep is not None:      # keep every species present in the starting guess
        gx = 0.98 * gx / gx.sum() + 0.02 * sweep / sweep.sum()
    for th in th_list:
        gx, err = solve_at(th, gx)
        if err > 1e-6:
            raise InfeasibleError(f"counter-current shooting failed at stage cut {th:.3f}")
    return gx


def _cc_backward(membrane, xF, theta, ph, pl, F, guess, rtol, p_high, p_low, sweep=None):
    """Shooting from the closed (retentate) end; returns x_R."""
    Q, n = membrane.Q, membrane.n
    if n > 2 or (sweep is not None and np.any(xF <= 0)):
        return _cc_multicomponent(membrane, xF, theta, p_high, p_low, F, guess, rtol, sweep)

    def g(t):          # t = logit of x_R[0]
        with np.errstate(over="ignore"):
            x0 = 1.0 / (1.0 + np.exp(-t))
        if x0 <= 1e-290 or x0 >= 1 - 1e-15:
            return np.nan                     # outside double precision
        sol, ok, _ = _cc_shoot(np.array([x0, 1 - x0]), Q, ph, pl, F, theta, n, rtol=rtol,
                               sweep=sweep)
        if sol.depleted is not None:
            # component 0 exhausted -> x_R[0] too small (residual -x_F0);
            # component 1 exhausted -> x_R[0] too large (residual 1-x_F0)
            return -xF[0] if sol.depleted == 0 else 1.0 - xF[0]
        if not ok:
            return np.nan
        return sol.y[0, -1] / F - xF[0]

    fast = int(np.argmax(Q))
    t_lim = np.log(1e-280) * (1 if fast == 0 else -1)   # fast gas at 1e-280
    if (g(t_lim) > 0) if fast == 0 else (g(t_lim) < 0):
        if sweep is not None:
            raise InfeasibleError("counter-current with sweep: the fast gas is stripped "
                                  "completely; reduce the stage cut or the sweep")
        raise _Stripped()
    t0 = np.log(max(guess[0], 1e-300) / max(guess[1], 1e-300))
    a, b, ga, gb = _bracket_increasing(g, t0, step=2.0)
    t = a if a == b else brentq(g, a, b, xtol=1e-10, rtol=1e-12)
    x0 = 1.0 / (1.0 + np.exp(-t))
    xR = np.array([x0, 1 - x0])
    # reject a "root" sitting on the depletion discontinuity
    sol, ok, _ = _cc_shoot(xR, Q, ph, pl, F, theta, n, rtol=rtol, sweep=sweep)
    if not ok or abs(sol.y[0, -1] / F - xF[0]) > 1e-7:
        raise InfeasibleError("backward shooting did not converge")
    return xR


class _Stripped(Exception):
    """The fast gas is stripped from the retentate below double precision."""


def _cc_stripped_binary(membrane, xF, theta, ph, pl, F, dense):
    """Binary counter-current limit in which the fast gas is completely
    removed before the retentate end (x_R,fast < 1e-280, i.e. zero in double
    precision).  Then y_out = x_F/theta for the fast gas, the module is
    integrated from the feed end until the fast gas is exhausted, and the
    remaining area permeates pure slow gas at flux Q_slow (p_h - p_l).
    Returns (x_ret, y_perm, area, sol, fluxes, info)."""
    Q, n = membrane.Q, 2
    i, j = int(np.argmax(Q)), int(np.argmin(Q))
    x_ret = np.zeros(2); x_ret[j] = 1.0
    y_perm = (xF - (1 - theta) * x_ret) / theta

    def fluxes(s):
        x, y = _perm_comp(s[:n], s[n:2 * n], Q, ph, pl, F)
        J = Q * (ph * x - pl * y)
        return J, J.sum(), x, y

    def rhs(L, s):
        J, Jt, _, _ = fluxes(s)
        return np.concatenate([J / Jt, J / Jt, [-1.0 / Jt]])

    def gone(L, s):
        return s[i] - 1e-12 * F * xF[i]
    gone.terminal, gone.direction = True, -1

    s0 = np.concatenate([F * xF, theta * F * y_perm, [0.0]])
    sol = solve_ivp(rhs, (F, (1 - theta) * F), s0, method=_METHOD, rtol=1e-9,
                    atol=1e-30 * F, events=gone, dense_output=dense)
    L_strip = sol.t[-1]
    A_strip = float(sol.y[2 * n, -1])
    A_tail = (L_strip - (1 - theta) * F) / (Q[j] * (ph - pl))
    area = A_strip + A_tail
    info = {"shooting": "stripped limit", "fast_gas_exhausted_at_area_fraction": A_strip / area}
    return x_ret, y_perm, area, sol, fluxes, info


def countercurrent(membrane: Membrane, x_feed, theta: float, p_high: float, p_low: float,
                   feed_flow: float = 1.0, profile: bool = True, rtol: float = 1e-9,
                   x_ret_guess=None, sweep_flow: float = 0.0,
                   sweep_composition=None) -> StageResult:
    """Counter-current plug flow at a specified stage cut (pressures in bar).

    x_ret_guess : optional starting retentate composition for the shooting
                  (e.g. from a nearby solution); default is the cross-flow result.
    sweep_flow, sweep_composition : permeate sweep (mol/s, mapping) entering at
                  the retentate end and leaving with the permeate at the feed end.
    """
    xF, ph, pl = check_inputs(membrane, x_feed, theta, p_high, p_low)
    Q, n, F = membrane.Q, membrane.n, feed_flow
    sweep = _sweep_vector(membrane, sweep_flow, sweep_composition, F)
    if x_ret_guess is not None:
        guess = np.asarray(x_ret_guess, float)
    else:
        guess = cross_flow(membrane, xF, theta, p_high, p_low, F, profile=False).x_ret
    if sweep is not None:      # keep every species present in the starting guess
        guess = 0.98 * guess / guess.sum() + 0.02 * sweep / sweep.sum()

    info = {"shooting": "backward"}
    try:
        xR = _cc_backward(membrane, xF, theta, ph, pl, F, guess, rtol, p_high, p_low, sweep)
        sol, ok, fluxes = _cc_shoot(xR, Q, ph, pl, F, theta, n, dense=profile, rtol=rtol,
                                    sweep=sweep,
                                    check=(xF > 1e-14) if sweep is not None else None)
        if not ok:
            raise InfeasibleError("final backward shot failed")
        P_end, area = sol.y[n:2 * n, -1], float(sol.y[2 * n, -1])
        y_perm = np.clip(P_end, 0, None); y_perm /= y_perm.sum()
        x_ret = xR
        method = "backward"
    except _Stripped:
        x_ret, y_perm, area, sol, fluxes, info = _cc_stripped_binary(membrane, xF, theta, ph,
                                                                     pl, F, profile)
        method = "stripped"

    prof = None
    if profile:
        # report in the flow direction of the feed (area fraction 0 = feed end)
        if method == "backward":
            L = np.linspace(F, sol.t[0], 200)
            S = sol.sol(L)
            frac = 1.0 - S[2 * n] / area
        else:
            # forward part (until the fast gas is exhausted) + pure slow-gas tail
            L = np.linspace(F, sol.t[-1], 200)
            S = sol.sol(L)
            frac = S[2 * n] / area
        xs, ys = zip(*[fluxes(S[:, k])[2:] for k in range(L.size)])
        prof = {"area_fraction": frac, "theta_local": (F - L) / F,
                "x": np.array(xs).T, "y": np.array(ys).T,
                "permeate_flow": S[n:2 * n].sum(axis=0)}
    return StageResult("countercurrent", membrane.components, theta, F, xF, x_ret, y_perm,
                       area, p_high, p_low, profile=prof, extra=info,
                       sweep_flow=float(sweep_flow or 0.0),
                       sweep_x=None if sweep is None else sweep / sweep.sum())
