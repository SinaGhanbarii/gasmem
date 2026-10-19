"""Matplotlib helpers with one consistent, colour-blind-checked style.

Flow patterns always keep the same colour (colour follows the entity), lines
are thin, grids recessive, and every multi-series chart has a legend.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stagecut import LABELS, PATTERNS

# validated categorical slots (light background), fixed order
# (first four slots of the reference palette; passes the adjacent-pair CVD and
#  normal-vision checks; markers give a second, colour-free encoding)
COLORS = {"countercurrent": "#2a78d6", "cross_flow": "#eb6834",
          "cocurrent": "#1baf7a", "well_mixed": "#eda100"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = {"well_mixed": "o", "cross_flow": "s", "cocurrent": "^", "countercurrent": "D"}


def use_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
    })


use_style()


def _mark_every(n, k=6):
    return max(1, n // k)


def _lines(ax, df, ycol, patterns, theta_now=None):
    for p in patterns:
        d = df[df["pattern"] == p].sort_values("theta")
        if d.empty or ycol not in d:
            continue
        ax.plot(d["theta"], d[ycol], color=COLORS[p], marker=MARKERS[p], markersize=4,
                markevery=_mark_every(len(d)), label=LABELS[p])
    if theta_now is not None:
        ax.axvline(theta_now, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Stage cut θ (permeate / feed)")
    ax.set_xlim(0, 1)


def plot_stage_cut_tradeoff(df: pd.DataFrame, comp: str, area_col: str = "area_m2",
                            patterns=PATTERNS, theta_now=None, slow: str | None = None,
                            title: str | None = None, area_label: str = "Membrane area (m²)"):
    """Small multiples: permeate purity, recovery, retentate purity and area vs stage cut."""
    n = 4 if slow else 3
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), constrained_layout=True)
    _lines(axes[0], df, f"y_perm_{comp}", patterns, theta_now)
    axes[0].set_title(f"{comp} purity in permeate")
    axes[0].set_ylabel("Mole fraction")
    _lines(axes[1], df, f"recovery_perm_{comp}", patterns, theta_now)
    axes[1].set_title(f"{comp} recovery in permeate")
    axes[1].set_ylabel("Fraction of feed")
    axes[1].set_ylim(0, 1.02)
    k = 2
    if slow:
        _lines(axes[2], df, f"x_ret_{slow}", patterns, theta_now)
        axes[2].set_title(f"{slow} purity in retentate")
        axes[2].set_ylabel("Mole fraction")
        k = 3
    _lines(axes[k], df, area_col, patterns, theta_now)
    axes[k].set_title("Membrane area")
    axes[k].set_ylabel(area_label)
    axes[k].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l))
    if title:
        fig.suptitle(title, color=INK, fontsize=12, fontweight="bold")
    return fig


def plot_purity_recovery(df: pd.DataFrame, comp: str, patterns=PATTERNS, ax=None,
                         title: str | None = None):
    """The classic trade-off curve: every point is one stage cut."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    for p in patterns:
        d = df[df["pattern"] == p].sort_values("theta")
        ax.plot(d[f"recovery_perm_{comp}"], d[f"y_perm_{comp}"], color=COLORS[p],
                marker=MARKERS[p], markersize=4, markevery=_mark_every(len(d)), label=LABELS[p])
    ax.set_xlabel(f"{comp} recovery in permeate")
    ax.set_ylabel(f"{comp} purity in permeate")
    ax.set_xlim(0, 1.02)
    ax.legend()
    ax.set_title(title or "Purity–recovery trade-off (stage cut varies along each curve)")
    return ax.figure


def plot_profiles(results: dict, comp: str, ax=None, title: str | None = None):
    """Feed-side x and permeate-side y of `comp` along the module.

    results : {pattern: StageResult with profile}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    for p, r in results.items():
        if r.profile is None:
            continue
        i = r.components.index(comp)
        pr = r.profile
        ax.plot(pr["area_fraction"], pr["x"][i], color=COLORS[p], label=f"{LABELS[p]}: feed side")
        ykey = "y" if "y" in pr else "y_local"
        ax.plot(pr["area_fraction"], pr[ykey][i], color=COLORS[p], ls="--",
                label=f"{LABELS[p]}: permeate side")
    ax.set_xlabel("Fraction of membrane area from the feed inlet")
    ax.set_ylabel(f"{comp} mole fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(title or f"{comp} along the module")
    return ax.figure


def bar_breakdown(labels, values, unit: str, title: str, ax=None, color=SERIES[0]):
    """Horizontal bars for an energy or cost breakdown (direct-labelled)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(labels) + 1.2), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.6)
    for yy, v in zip(y, values):
        ax.text(v, yy, f"  {v:,.3g}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(unit)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.25 if len(values) and max(values) > 0 else 1)
    return ax.figure
