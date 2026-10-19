"""Digitised data from published figures (see tools/digitised_data.py for how
each point was read, including its pixel coordinates).

    >>> from gasmem.datasets import load
    >>> pan = load("Pan1986")          # measured points, Figs. 2, 3, 6, 7
    >>> feng = load("Feng1999")        # measured points, Figs. 4, 5, 13, 14
    >>> mak = load("Makaruk2010")      # MODEL curves, Figs. 15, 16
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {"Pan1986": "pan1986_digitised.csv", "Feng1999": "feng1999_digitised.csv",
           "Makaruk2010": "makaruk2010_digitised.csv"}


def load(source: str) -> pd.DataFrame:
    """Digitised points for one paper (columns: figure, quantity, series, x, y, px, py)."""
    try:
        return pd.read_csv(DATA / SOURCES[source])
    except KeyError:
        raise KeyError(f"unknown source {source!r}; available: {sorted(SOURCES)}") from None


def available() -> pd.DataFrame:
    """Summary of every digitised series."""
    frames = [load(s) for s in SOURCES]
    d = pd.concat(frames)
    return d.groupby(["source", "figure", "quantity", "series"]).size().rename("points").reset_index()
