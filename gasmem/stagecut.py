"""One entry point for all flow patterns, plus stage-cut sweeps and the
binary limits that explain them."""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .core import InfeasibleError, Membrane, StageResult, local_permeate
from .crossflow import cross_flow
from .hollowfiber import cocurrent, countercurrent
from .properties import BAR
from .wellmixed import well_mixed

PATTERNS = ("well_mixed", "cross_flow", "cocurrent", "countercurrent")
LABELS = {"well_mixed": "Well-mixed", "cross_flow": "Cross-flow",
          "cocurrent": "Co-current", "countercurrent": "Counter-current"}
_SOLVERS = {"well_mixed": well_mixed, "cross_flow": cross_flow,
            "cocurrent": cocurrent, "countercurrent": countercurrent}


def solve_stage(membrane: Membrane, x_feed, theta: float, p_high: float, p_low: float,
                pattern: str = "countercurrent", feed_flow: float = 1.0,
                profile: bool = False, x_ret_guess=None, sweep_flow: float = 0.0,
                sweep_composition=None) -> StageResult:
    """Single membrane stage at a given stage cut.

    pattern : 'well_mixed' | 'cross_flow' | 'cocurrent' | 'countercurrent'
    Pressures in bar, feed_flow in mol/s.
    sweep_flow (mol/s) and sweep_composition (e.g. {'N2': 1}) add a permeate
    sweep gas; only for 'cocurrent' and 'countercurrent', where the permeate
    has a flow path.  The stage cut is still the fraction of the feed that
    permeates; the permeate outlet carries theta*feed_flow + sweep_flow.
    """
    try:
        f = _SOLVERS[pattern]
    except KeyError:
        raise ValueError(f"pattern must be one of {PATTERNS}") from None
    sw = {}
    if sweep_flow:
        if pattern not in ("cocurrent", "countercurrent"):
            raise ValueError("a sweep needs a permeate flow path: use 'cocurrent' or "
                             "'countercurrent'")
        sw = {"sweep_flow": sweep_flow, "sweep_composition": sweep_composition}
    if pattern == "well_mixed":
        return f(membrane, x_feed, theta, p_high, p_low, feed_flow)
    if pattern == "countercurrent" and x_ret_guess is not None:
        try:
            return f(membrane, x_feed, theta, p_high, p_low, feed_flow, profile=profile,
                     x_ret_guess=x_ret_guess, **sw)
        except InfeasibleError:
            pass                       # fall back to the default starting guess
    return f(membrane, x_feed, theta, p_high, p_low, feed_flow, profile=profile, **sw)


def sweep_stage_cut(membrane: Membrane, x_feed, p_high: float, p_low: float,
                    thetas: Iterable[float] = np.linspace(0.02, 0.95, 32),
                    patterns: Iterable[str] = PATTERNS, feed_flow: float = 1.0,
                    quiet: bool = True) -> pd.DataFrame:
    """Solve every pattern over a range of stage cuts.

    Returns a tidy DataFrame with one row per (pattern, theta): permeate and
    retentate compositions, recovery of every component in the permeate and
    the membrane area.  Infeasible points are kept as NaN rows.
    """
    rows = []
    for pat in patterns:
        for th in thetas:
            try:
                r = solve_stage(membrane, x_feed, float(th), p_high, p_low, pat, feed_flow)
                rows.append(r.as_dict())
            except (InfeasibleError, RuntimeError) as e:
                if not quiet:
                    warnings.warn(f"{pat} at theta={th:.3f}: {e}")
                rows.append({"pattern": pat, "theta": float(th), "area_m2": np.nan})
    return pd.DataFrame(rows)


def binary_limits(alpha: float, phi: float, x: float) -> dict:
    """Upper bounds on permeate purity for a binary at feed-side fraction x.

    * selectivity limit (infinite pressure ratio):  y = a x / (1 + (a-1) x)
    * pressure-ratio limit (infinite selectivity): y = phi x  (capped at 1)
    * actual local value (both finite): solution of the local flux equation
    The regime is 'pressure-ratio limited' when phi << alpha, 'selectivity
    limited' when phi >> alpha, and intermediate otherwise.
    """
    y_sel = alpha * x / (1.0 + (alpha - 1.0) * x)
    y_pr = min(1.0, phi * x)
    m = Membrane.binary(alpha)
    y_loc = local_permeate(np.array([x, 1 - x]), m.Q, phi * BAR, 1.0 * BAR)[0][0]
    ratio = phi / alpha
    regime = ("pressure-ratio limited" if ratio < 0.5 else
              "selectivity limited" if ratio > 2.0 else "intermediate")
    return {"y_selectivity_limit": y_sel, "y_pressure_ratio_limit": y_pr,
            "y_local": float(y_loc), "phi_over_alpha": ratio, "regime": regime}


def binary_sweep(alpha: float, phi: float, x_feed: float,
                 thetas: Iterable[float] = np.linspace(0.02, 0.95, 32),
                 patterns: Iterable[str] = PATTERNS) -> pd.DataFrame:
    """Dimensionless binary study: p_low = 1 bar, p_high = phi bar, F = 1 mol/s.

    Adds the dimensionless area  A* = A Q_A p_h / F  (independent of the
    absolute permeance), and the fast-gas purity/recovery under short names.
    """
    m = Membrane.binary(alpha)
    df = sweep_stage_cut(m, [x_feed, 1 - x_feed], phi, 1.0, thetas, patterns, 1.0)
    df["area_dimless"] = df["area_m2"] * m.Q[0] * phi * BAR
    df["purity"] = df.get("y_perm_A")
    df["recovery"] = df.get("recovery_perm_A")
    df["retentate_slow_purity"] = df.get("x_ret_B")
    return df
