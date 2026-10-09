from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#3977AD"
ORANGE = "#D55E00"
GEOMETRY_LABELS = {
    "circle_R2": r"$S^1\subset\mathbb{R}^2$",
    "variable_curve_R3": r"$\gamma(t)=(1.5\cos t,\sin t,0.3\sin 2t)$",
}
GEOMETRY_STYLE = {
    "circle_R2": dict(color=BLUE, marker="o"),
    "variable_curve_R3": dict(color=ORANGE, marker="s"),
}


def _style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10, colors="#222222")
    axis.grid(False)


def _save(fig, figures, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(figures/f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _delta(block):
    return 100.0*np.asarray([row["delta"] for row in block["coefficients"]], dtype=float)


def _plot_method_pair(axes, method, payload, *, show_legend=True):
    method_blocks = payload["methods"][method]
    for geometry_key in ("circle_R2", "variable_curve_R3"):
        block = method_blocks[geometry_key]
        delta = _delta(block)
        if method == "YY":
            residual = [row["maximum_directional_residual"] for row in block["coefficients"]]
        else:
            residual = [row["true_cancellation_residual"] for row in block["coefficients"]]
        style = GEOMETRY_STYLE[geometry_key]
        axes[0].plot(
            delta, residual, lw=2.0, ms=5, label=GEOMETRY_LABELS[geometry_key], **style
        )
        axes[1].plot(
            delta, [row["delta_infinity"] for row in block["geometry"]],
            lw=2.0, ms=5, label=GEOMETRY_LABELS[geometry_key], **style
        )
    axes[0].set_xlabel(r"Relative noise error $\delta$ (%)", fontsize=11)
    axes[0].set_ylabel(r"Cancellation residual $R_c(\delta)$", fontsize=11)
    axes[1].set_xlabel(r"Relative noise error $\delta$ (%)", fontsize=11)
    axes[1].set_ylabel(r"Population deviation $\Delta_\infty(\delta)$", fontsize=11)
    if show_legend:
        axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        _style_axis(axis)


def build(root):
    root = Path(root)
    payload = json.loads((root/"results"/"e6_results.json").read_text(encoding="utf-8"))
    figures = root/"figures"
    figures.mkdir(exist_ok=True)

    dr = payload["methods"]["DR"]
    circle = dr["circle_R2"]
    curve = dr["variable_curve_R3"]
    delta = _delta(circle)
    residual = np.asarray(
        [row["true_cancellation_residual"] for row in circle["coefficients"]]
    )
    circle_change = np.asarray([row["delta_infinity"] for row in circle["geometry"]])
    curve_change = np.asarray([row["delta_infinity"] for row in curve["geometry"]])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
    axes[0].plot(delta, residual, color=BLUE, marker="o", lw=2.0, ms=5)
    axes[0].set_xlabel(r"Relative noise error $\delta$ (%)", fontsize=11)
    axes[0].set_ylabel(r"Cancellation residual $R_c(\delta)$", fontsize=11)
    axes[1].plot(
        delta, circle_change, color=BLUE, marker="o", lw=2.0, ms=5,
        label=GEOMETRY_LABELS["circle_R2"],
    )
    axes[1].plot(
        delta, curve_change, color=ORANGE, marker="s", lw=2.0, ms=5,
        label=GEOMETRY_LABELS["variable_curve_R3"],
    )
    axes[1].set_xlabel(r"Relative noise error $\delta$ (%)", fontsize=11)
    axes[1].set_ylabel(r"Population deviation $\Delta_\infty(\delta)$", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        _style_axis(axis)
    _save(fig, figures, "e6_dr_noise_misspecification_main")

    for method, stem in (
        ("STSP", "e6_stsp_noise_misspecification_supplement"),
        ("YY", "e6_yy_noise_misspecification_supplement"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
        _plot_method_pair(axes, method, payload)
        _save(fig, figures, stem)

