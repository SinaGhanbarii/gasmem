"""Core objects shared by every model: Membrane, StageResult and the local
(cross-flow) permeate relation.

Notation
--------
F      feed flow (mol/s)
theta  stage cut = permeate flow / feed flow
x      feed-side (retentate) mole fractions
y      permeate-side mole fractions
p_h    feed-side pressure, p_l permeate-side pressure
Q_i    permeance of component i (mol m-2 s-1 Pa-1)
Local flux of i:  J_i = Q_i (p_h x_i - p_l y_i)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from .properties import BAR, GPU


class InfeasibleError(ValueError):
    """Raised when the requested operating point cannot be reached
    (e.g. a stage cut beyond the pressure-ratio limit of a co-current module)."""


class Membrane:
    """A set of components and their permeances.

    Parameters
    ----------
    permeances : mapping name -> permeance
    units : 'GPU' (default) or 'SI' (mol m-2 s-1 Pa-1)
    name : optional label
    """

    def __init__(self, permeances: Mapping[str, float], units: str = "GPU", name: str = ""):
        if len(permeances) < 2:
            raise ValueError("need at least two components")
        factor = GPU if units.upper() == "GPU" else 1.0
        self.components: tuple[str, ...] = tuple(permeances)
        self.Q = np.array([float(permeances[c]) for c in self.components]) * factor
        if np.any(self.Q <= 0):
            raise ValueError("permeances must be positive")
        self.name = name

    # -- constructors -----------------------------------------------------
    @classmethod
    def binary(cls, alpha: float, fast_gpu: float = 100.0,
               names: tuple[str, str] = ("A", "B")) -> "Membrane":
        """Binary membrane with selectivity alpha = Q_fast / Q_slow."""
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        return cls({names[0]: fast_gpu, names[1]: fast_gpu / alpha},
                   name=f"binary a={alpha:g}")

    @classmethod
    def from_permeability(cls, permeability_barrer: Mapping[str, float], thickness_um: float,
                          name: str = "") -> "Membrane":
        """Membrane from intrinsic permeabilities (Barrer) and selective-layer thickness (um).

        Most papers report permeability; the module models need permeance:
        permeance [GPU] = permeability [Barrer] / thickness [um].
        """
        if thickness_um <= 0:
            raise ValueError("thickness must be positive")
        return cls({k: v / thickness_um for k, v in permeability_barrer.items()},
                   name=name or f"{thickness_um:g} um layer")

    # -- helpers ----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.components)

    @property
    def gpu(self) -> dict[str, float]:
        return {c: q / GPU for c, q in zip(self.components, self.Q)}

    def index(self, comp: str | int) -> int:
        return comp if isinstance(comp, (int, np.integer)) else self.components.index(comp)

    def selectivity(self, a: str | int, b: str | int) -> float:
        return float(self.Q[self.index(a)] / self.Q[self.index(b)])

    def vector(self, composition, normalise: bool = True) -> np.ndarray:
        """Composition (mapping or sequence) -> array aligned with components."""
        if isinstance(composition, Mapping):
            unknown = set(composition) - set(self.components)
            if unknown:
                raise KeyError(f"components not in membrane: {sorted(unknown)}")
            v = np.array([float(composition.get(c, 0.0)) for c in self.components])
        else:
            v = np.asarray(composition, float).copy()
            if v.shape != (self.n,):
                raise ValueError(f"composition must have {self.n} entries")
        if np.any(v < 0):
            raise ValueError("mole fractions must be non-negative")
        s = v.sum()
        if s <= 0:
            raise ValueError("composition sums to zero")
        return v / s if normalise else v

    def __repr__(self) -> str:
        g = ", ".join(f"{c}={q:.4g}" for c, q in self.gpu.items())
        return f"Membrane({self.name!r}: {g} GPU)"


def local_permeate(x: np.ndarray, Q: np.ndarray, ph: float, pl: float):
    """Permeate composition leaving a membrane element with no permeate-side mixing.

    Solves y_i = J_i / sum(J) with J_i = Q_i (ph x_i - pl y_i).  Writing
    Jt = sum(J) gives y_i = Q_i ph x_i / (Jt + Q_i pl); sum(y)=1 fixes Jt, which
    has exactly one positive root.  Returns (y, Jt) with Jt in mol m-2 s-1.
    """
    a = Q * ph * x
    if pl <= 0.0:
        Jt = a.sum()
        return a / Jt, Jt
    b = Q * pl
    hi = a.sum()
    if hi <= 0.0:
        raise InfeasibleError("no permeating species")
    f = lambda J: np.sum(a / (J + b)) - 1.0
    if f(0.0) <= 0.0:   # only if sum(x) < pl/ph, i.e. never for normalised x
        raise InfeasibleError("no positive driving force")
    Jt = brentq(f, 0.0, hi, xtol=1e-15 * hi, rtol=4e-15, maxiter=200)
    return a / (Jt + b), Jt


@dataclass
class StageResult:
    """Outcome of one membrane stage.  Pressures stored in bar, flows in mol/s."""

    pattern: str
    components: tuple[str, ...]
    theta: float
    feed_flow: float
    x_feed: np.ndarray
    x_ret: np.ndarray
    y_perm: np.ndarray
    area: float
    p_high: float
    p_low: float
    profile: dict | None = field(default=None, repr=False)
    extra: dict = field(default_factory=dict, repr=False)
    sweep_flow: float = 0.0                       # permeate-side sweep gas, mol/s
    sweep_x: np.ndarray | None = field(default=None, repr=False)

    # -- flows ------------------------------------------------------------
    # With a sweep, the stage cut is still the fraction of the *feed* that
    # crosses the membrane, theta = 1 - L_R/F; the permeate outlet then
    # carries theta*F + sweep_flow.
    @property
    def permeate_flow(self) -> float:
        return self.theta * self.feed_flow + self.sweep_flow

    @property
    def retentate_flow(self) -> float:
        return (1.0 - self.theta) * self.feed_flow

    def _i(self, comp) -> int:
        return comp if isinstance(comp, (int, np.integer)) else self.components.index(comp)

    def purity(self, comp, stream: str = "permeate") -> float:
        """Mole fraction of `comp` in 'permeate' or 'retentate'."""
        i = self._i(comp)
        return float(self.y_perm[i] if stream.startswith("p") else self.x_ret[i])

    def recovery(self, comp, stream: str = "permeate") -> float:
        """Fraction of the feed's `comp` that leaves in the chosen stream."""
        i = self._i(comp)
        if self.x_feed[i] <= 0:
            return float("nan")
        to_ret = (1.0 - self.theta) * self.x_ret[i] / self.x_feed[i]
        if stream.startswith("p"):
            if self.sweep_flow > 0:          # net transfer out of the feed
                return float(1.0 - to_ret)
            return float(self.theta * self.y_perm[i] / self.x_feed[i])
        return float(to_ret)

    def mass_balance_error(self) -> float:
        """Max absolute component balance error, relative to feed flow."""
        sw = (self.sweep_flow / self.feed_flow * self.sweep_x
              if self.sweep_flow > 0 else 0.0)
        err = (self.x_feed + sw - (self.permeate_flow / self.feed_flow) * self.y_perm
               - (1.0 - self.theta) * self.x_ret)
        return float(np.max(np.abs(err)))

    @property
    def pressure_ratio(self) -> float:
        return self.p_high / self.p_low if self.p_low > 0 else np.inf

    def as_dict(self) -> dict:
        d = {"pattern": self.pattern, "theta": self.theta, "area_m2": self.area,
             "feed_flow_mol_s": self.feed_flow, "p_high_bar": self.p_high,
             "p_low_bar": self.p_low}
        for c, xf, xr, yp in zip(self.components, self.x_feed, self.x_ret, self.y_perm):
            d[f"x_feed_{c}"] = xf
            d[f"x_ret_{c}"] = xr
            d[f"y_perm_{c}"] = yp
        for c in self.components:
            d[f"recovery_perm_{c}"] = self.recovery(c, "permeate")
        if self.sweep_flow > 0:
            d["sweep_flow_mol_s"] = self.sweep_flow
        d.update({k: v for k, v in self.extra.items() if np.isscalar(v)})
        return d

    def summary(self) -> str:
        lines = [f"{self.pattern}: stage cut = {self.theta:.4f}, area = {self.area:.4g} m2, "
                 f"p_h/p_l = {self.p_high:g}/{self.p_low:g} bar"]
        lines.append(f"{'comp':>6} {'feed':>8} {'retent.':>8} {'perm.':>8} {'rec->perm':>10}")
        if self.sweep_flow > 0:
            lines[0] += (f", sweep = {self.sweep_flow / self.feed_flow:.3g} x feed "
                         f"({', '.join(f'{c} {v:.3g}' for c, v in zip(self.components, self.sweep_x))})")
        for c, xf, xr, yp in zip(self.components, self.x_feed, self.x_ret, self.y_perm):
            rec = self.recovery(c, "permeate")
            lines.append(f"{c:>6} {xf:8.4f} {xr:8.4f} {yp:8.4f} {rec:10.4f}")
        return "\n".join(lines)


def check_inputs(membrane: Membrane, x_feed, theta, p_high, p_low):
    """Validate and convert common inputs; returns (x, ph_Pa, pl_Pa)."""
    x = membrane.vector(x_feed)
    if not (0.0 < theta < 1.0):
        raise ValueError("stage cut must lie strictly between 0 and 1")
    if p_low < 0 or p_high <= p_low:
        raise ValueError("need p_high > p_low >= 0 (bar)")
    return x, p_high * BAR, p_low * BAR


def as_list(seq: Sequence | np.ndarray | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(seq, float))
