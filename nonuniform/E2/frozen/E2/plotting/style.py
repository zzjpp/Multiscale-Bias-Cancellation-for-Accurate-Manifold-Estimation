from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
N = (2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000)
METHODS = ("STSP", "DR", "YY")
METRICS = ("E_avg", "E_infty")
GEOMETRIES = (
    ("circle_r2", r"$S^1\subset\mathbb{R}^2$"),
    ("clifford_torus_r4", r"Clifford torus $\subset\mathbb{R}^4$"),
    ("sphere2_r10", r"$S^2\subset\mathbb{R}^{10}$"),
    ("sphere2_r3", r"$S^2\subset\mathbb{R}^3$"),
    ("sphere3_r4", r"$S^3\subset\mathbb{R}^4$"),
    ("corrugated_torus_r3", r"Corrugated torus $\subset\mathbb{R}^3$"),
)
COLORS = {"Single": "#0072B2", "Multi": "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}
STYLES = {"Single": "-", "Multi": "--"}

def selected(rows, method, geometry, metric, display):
    estimator = "Multi" if display == "Multi" else (
        "Single" if method == "YY" else "Single_h2")
    data = sorted((r for r in rows if (r["method"], r["geometry"], r["metric"], r["estimator"])
                   == (method, geometry, metric, estimator)), key=lambda r: float(r["scale"]))
    assert tuple(float(r["scale"]) for r in data) == N
    return data


def plot_series(axis, data, display):
    x = np.array([float(r["scale"]) for r in data])
    y = np.array([float(r["median"]) for r in data])
    q1 = np.array([float(r["q1"]) for r in data])
    q3 = np.array([float(r["q3"]) for r in data])
    reliable = np.array([r["reliable"] == "True" for r in data])
    valid = np.isfinite(y) & (y > 0)
    band_valid = valid & np.isfinite(q1) & np.isfinite(q3) & (q1 > 0) & (q3 > 0)
    # Only interquartile uncertainty is shaded. There is no favorable/green span.
    axis.fill_between(x, np.where(band_valid, q1, np.nan),
                      np.where(band_valid, q3, np.nan), color=COLORS[display],
                      alpha=.12, linewidth=0, zorder=1)
    plotted = np.where(valid, y, np.nan)
    line, = axis.plot(x, plotted, color=COLORS[display], marker=MARKERS[display],
                     linestyle=STYLES[display], linewidth=1.4, markersize=3.4,
                     markeredgewidth=.7, zorder=3)
    bad = valid & ~reliable
    axis.scatter(x[bad], y[bad], marker=MARKERS[display], s=25, facecolors="white",
                 edgecolors=COLORS[display], linewidths=.9, zorder=5)
    assert np.array_equal(line.get_xdata(), x)
    assert np.array_equal(line.get_ydata(), plotted, equal_nan=True)
    return {"display": display, "original_estimator": data[0]["estimator"],
            "original_rows": data, "hollow_n": x[bad].tolist(),
            "undefined_or_nonpositive_n": x[~valid].tolist()}

