"""Compare the models with your own data, and fit permeances to it.

Your data go in a CSV (or DataFrame) with one row per operating point.  Get a
blank file with the right headers from `template`:

    >>> from gasmem.userdata import template, compare, fit_permeances
    >>> template(["CO2", "CH4"], "my_module.csv")

Columns (mole fractions as fractions, not percent):

    point              optional label
    pattern            optional; 'well_mixed', 'cross_flow', 'cocurrent' or
                       'countercurrent' (otherwise the `pattern` argument)
    p_feed_bar         feed-side pressure, bar(a)
    p_perm_bar         permeate-side pressure, bar(a)
    feed_flow_mol_s    or feed_flow_Nm3h; needed to compare areas or to rate
    x_feed_<comp>      feed composition, one column per component
    stage_cut          measured stage cut  -> design mode: the model is run at
                       this stage cut and predicts compositions (and the area,
                       if a feed flow is given)
    area_m2            membrane area       -> if there is no stage cut, rating
                       mode: the model predicts the stage cut of this area
    y_perm_<comp>      measured permeate composition (any components)
    x_ret_<comp>       measured retentate composition (any components)

Leave a cell empty when it was not measured.  Every measured quantity that
the model can predict is compared.

Please consider sharing your comparison (see CONTRIBUTING.md): data from real
modules are the most useful contribution to the toolkit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .core import InfeasibleError, Membrane, StageResult
from .module import rate_module
from .properties import nm3h_to_mols
from .stagecut import PATTERNS, solve_stage


# ---------------------------------------------------------------------------
def template(components, path: str | Path | None = None, rows: int = 3) -> pd.DataFrame:
    """Blank data sheet for `components`; written to `path` (CSV) if given."""
    cols = (["point", "pattern", "p_feed_bar", "p_perm_bar", "feed_flow_mol_s"]
            + [f"x_feed_{c}" for c in components] + ["stage_cut", "area_m2"]
            + [f"y_perm_{c}" for c in components] + [f"x_ret_{c}" for c in components])
    df = pd.DataFrame({c: [np.nan] * rows for c in cols})
    df["point"] = [f"P{k + 1}" for k in range(rows)]
    df["pattern"] = ""
    if path is not None:
        df.to_csv(path, index=False)
    return df


def read(data) -> pd.DataFrame:
    """Load and check a data sheet (path or DataFrame)."""
    df = pd.read_csv(data) if isinstance(data, (str, Path)) else pd.DataFrame(data).copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ("p_feed_bar", "p_perm_bar") if c not in df]
    if missing:
        raise ValueError(f"data sheet needs columns {missing}")
    if "feed_flow_Nm3h" in df and "feed_flow_mol_s" not in df:
        df["feed_flow_mol_s"] = df["feed_flow_Nm3h"].map(nm3h_to_mols)
    if "point" not in df:
        df["point"] = [f"P{k + 1}" for k in range(len(df))]
    frac = [c for c in df if c.startswith(("x_feed_", "y_perm_", "x_ret_"))] + \
           [c for c in ("stage_cut",) if c in df]
    for c in frac:
        v = pd.to_numeric(df[c], errors="coerce")
        if (v > 1.0 + 1e-9).any():
            raise ValueError(f"column {c!r} has values above 1: give mole fractions and "
                             f"stage cuts as fractions (divide percentages by 100)")
        df[c] = v
    return df


# ---------------------------------------------------------------------------
def _row_inputs(membrane, row, pattern):
    pat = row.get("pattern") if isinstance(row.get("pattern"), str) and row.get("pattern") else pattern
    if pat not in PATTERNS:
        raise ValueError(f"point {row['point']}: unknown pattern {pat!r}")
    xf = {}
    for c in membrane.components:
        v = row.get(f"x_feed_{c}", np.nan)
        xf[c] = 0.0 if pd.isna(v) else float(v)
    if sum(xf.values()) <= 0:
        raise ValueError(f"point {row['point']}: no feed composition (x_feed_<comp> columns)")
    F = row.get("feed_flow_mol_s", np.nan)
    return pat, xf, (None if pd.isna(F) else float(F))


def _predict(membrane, row, pattern, mode="auto"):
    """Model result for one data row, and the mode used.

    design : run at the measured stage cut (predicts compositions and area)
    rating : run the measured area at the measured feed flow (predicts the
             stage cut and compositions)
    auto   : design if a stage cut is given, otherwise rating
    """
    pat, xf, F = _row_inputs(membrane, row, pattern)
    th, A = row.get("stage_cut", np.nan), row.get("area_m2", np.nan)
    ph, pl = float(row["p_feed_bar"]), float(row["p_perm_bar"])
    if mode == "auto":
        mode = "design" if not pd.isna(th) else "rating"
    if mode == "design":
        if pd.isna(th):
            raise ValueError(f"point {row['point']}: design mode needs a stage_cut")
        return solve_stage(membrane, xf, float(th), ph, pl, pat, F or 1.0), "design", F
    if mode == "rating":
        if pd.isna(A) or F is None:
            raise ValueError(f"point {row['point']}: rating needs area_m2 and a feed flow")
        return rate_module(membrane, xf, F, ph, pl, float(A), pat, profile=False), "rating", F
    raise ValueError("mode must be 'auto', 'design' or 'rating'")


def _pairs(membrane, row, res: StageResult, mode, F):
    """(quantity, unit, measured, predicted) for every measured output."""
    out = []
    for c in membrane.components:
        for col, stream in ((f"y_perm_{c}", "permeate"), (f"x_ret_{c}", "retentate")):
            v = row.get(col, np.nan)
            if not pd.isna(v):
                out.append((f"{stream} {c}", "mol%", 100 * float(v),
                            100 * res.purity(c, stream)))
    if mode == "rating":
        if not pd.isna(row.get("stage_cut", np.nan)):
            out.append(("stage cut", "-", float(row["stage_cut"]), res.theta))
    elif F is not None and not pd.isna(row.get("area_m2", np.nan)):
        out.append(("area", "m2", float(row["area_m2"]), res.area))
    return out


@dataclass
class Comparison:
    """Measured versus predicted values, one row per (point, quantity)."""
    table: pd.DataFrame
    membrane: Membrane
    pattern: str
    failed: dict = field(default_factory=dict)   # point -> reason

    def summary(self) -> pd.DataFrame:
        """Bias, RMS and maximum error per quantity (same units as the data;
        compositions in mol%, plus relative RMS for areas)."""
        t = self.table.assign(rel=100 * self.table["error"] / self.table["measured"])
        g = t.groupby(["quantity", "unit"])
        rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
        return pd.DataFrame({"points": g.size(), "bias": g["error"].mean(),
                             "rms": g["error"].agg(rms),
                             "max_abs": g["error"].agg(lambda e: float(np.max(np.abs(e)))),
                             "rms_relative_%": g["rel"].agg(rms)})

    def plot(self, ax=None):
        """Parity plot (predicted against measured), one panel per unit."""
        import matplotlib.pyplot as plt
        units = list(dict.fromkeys(self.table["unit"]))
        if ax is None:
            fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 4), squeeze=False)
            axes = axes[0]
        else:
            axes = np.atleast_1d(ax)
        for a, u in zip(axes, units):
            d = self.table[self.table["unit"] == u]
            for q, dq in d.groupby("quantity"):
                a.plot(dq["measured"], dq["predicted"], "o", ms=5, label=q)
            lo = min(d["measured"].min(), d["predicted"].min())
            hi = max(d["measured"].max(), d["predicted"].max())
            pad = 0.05 * (hi - lo or 1.0)
            a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8)
            a.set_xlabel(f"measured ({u})")
            a.set_ylabel(f"predicted ({u})")
            a.legend(fontsize=8)
            a.set_aspect("equal", adjustable="datalim")
        axes[0].set_title(f"{self.pattern} model vs data")
        return axes


def compare(membrane: Membrane, data, pattern: str = "countercurrent",
            mode: str = "auto") -> Comparison:
    """Run the model at every data point and pair predictions with measurements.

    mode : 'auto' (design at the measured stage cut when there is one, else
           rating of the measured area), 'design' or 'rating'
    """
    df = read(data)
    rows, failed = [], {}
    for _, row in df.iterrows():
        try:
            res, used, F = _predict(membrane, row, pattern, mode)
        except (InfeasibleError, ValueError, RuntimeError) as e:
            failed[row["point"]] = str(e)
            continue
        for q, u, meas, pred in _pairs(membrane, row, res, used, F):
            rows.append({"point": row["point"], "mode": used, "quantity": q, "unit": u,
                         "measured": meas, "predicted": pred, "error": pred - meas})
    if not rows:
        raise ValueError("no point could be compared: " + "; ".join(f"{k}: {v}"
                                                                    for k, v in failed.items()))
    return Comparison(pd.DataFrame(rows), membrane, pattern, failed)


# ---------------------------------------------------------------------------
def fit_permeances(membrane: Membrane, data, pattern: str = "countercurrent",
                   fit: list[str] | None = None, mode: str = "auto", verbose: bool = False):
    """Least-squares permeances that best reproduce the data.

    fit : components whose permeance is adjusted (default: all).  The others
          keep the values in `membrane`, which also provides the starting point.

    Compositions depend only on permeance *ratios*.  If no point carries area
    information (an area with a feed flow), one permeance must stay fixed: the
    fastest component is then held at its starting value and the result fixes
    the selectivities only.

    Returns (fitted Membrane, Comparison at the fitted permeances, info dict).
    """
    from scipy.optimize import least_squares

    df = read(data)
    names = list(membrane.components)
    fit = list(fit) if fit is not None else names[:]
    has_area = (("area_m2" in df and "feed_flow_mol_s" in df)
                and bool((df["area_m2"].notna() & df["feed_flow_mol_s"].notna()).any()))
    note = ""
    if not has_area and len(fit) == len(names):
        fixed = names[int(np.argmax(membrane.Q))]
        fit.remove(fixed)
        note = (f"no area data: {fixed} held at {membrane.gpu[fixed]:.4g} GPU, "
                f"selectivities fitted")
    if not fit:
        raise ValueError("nothing to fit")
    gpu0 = membrane.gpu

    def make(v):
        g = dict(gpu0)
        for c, lv in zip(fit, v):
            g[c] = float(np.exp(lv))
        return Membrane(g, name=(membrane.name or "membrane") + " (fitted)")

    def resid(v):
        m = make(v)
        r = []
        for _, row in df.iterrows():
            try:
                res, used, F = _predict(m, row, pattern, mode)
                for q, u, meas, pred in _pairs(m, row, res, used, F):
                    # mol% errors, relative errors (in %) for area and stage cut
                    r.append(pred - meas if u == "mol%" else 100 * (pred - meas) / meas)
            except (InfeasibleError, ValueError, RuntimeError):
                r.append(1e2)
        if verbose:
            print({c: round(float(np.exp(x)), 4) for c, x in zip(fit, v)},
                  float(np.sqrt(np.mean(np.square(r)))))
        return np.array(r)

    v0 = np.log([gpu0[c] for c in fit])
    sol = least_squares(resid, v0, x_scale=1.0, diff_step=1e-4, max_nfev=200)
    m = make(sol.x)
    info = {"success": bool(sol.success), "message": sol.message, "note": note,
            "rms_residual": float(np.sqrt(np.mean(sol.fun ** 2))),
            "permeances_GPU": m.gpu}
    # approximate 95 % intervals from the Jacobian (log-normal)
    try:
        dof = max(1, sol.fun.size - sol.x.size)
        s2 = float(sol.fun @ sol.fun) / dof
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        se = np.sqrt(np.diag(cov))
        info["permeance_95pct_range_GPU"] = {c: (float(np.exp(x - 1.96 * e)),
                                                 float(np.exp(x + 1.96 * e)))
                                             for c, x, e in zip(fit, sol.x, se)}
    except np.linalg.LinAlgError:
        pass
    return m, compare(m, df, pattern, mode), info
