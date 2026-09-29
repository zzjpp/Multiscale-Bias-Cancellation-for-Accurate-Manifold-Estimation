from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "E_avg": r"Average error $E_{\mathrm{avg}}$",
    "E_infty": r"Maximum error $E_{\infty}$",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.9,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _draw_panel(
    ax: mpl.axes.Axes,
    rows: list[dict[str, Any]],
    geometry: str,
    method: str,
    metric: str,
    style: dict[str, Any],
) -> None:
    for estimator in style["estimators"]:
        current = sorted(
            (
                row
                for row in rows
                if row["geometry"] == geometry
                and row["method"] == method
                and row["estimator"] == estimator
                and row["metric"] == metric
            ),
            key=lambda row: row["scale"],
        )
        x = np.asarray([row["scale"] for row in current], dtype=float)
        y = np.asarray([row["median"] for row in current], dtype=float)
        q1 = np.asarray([row["q1"] for row in current], dtype=float)
        q3 = np.asarray([row["q3"] for row in current], dtype=float)
        reliable = np.asarray([row["reliable"] for row in current], dtype=bool)
        color = style["colors"][estimator]
        marker = style["markers"][estimator]

        finite_band = np.isfinite(q1) & np.isfinite(q3) & (q1 > 0) & (q3 > 0)
        ax.fill_between(
            x,
            q1,
            q3,
            where=finite_band,
            color=color,
            alpha=style["quartile_band_alpha"],
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=color,
            linestyle="-",
            marker=marker,
            markersize=3.1,
            markeredgewidth=0.75,
            label=estimator,
            zorder=3,
        )
        hollow = (~reliable) & np.isfinite(y) & (y > 0)
        ax.scatter(
            x[hollow],
            y[hollow],
            s=26,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            marker=marker,
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1700, 580000)
    ax.set_xticks(style["sample_sizes"])
    ax.set_xticklabels(
        style["sample_labels"], rotation=32, ha="right", rotation_mode="anchor"
    )
    ax.minorticks_off()
    ax.grid(False)
    ax.tick_params(length=2.5, width=0.6, pad=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(
    rows: list[dict[str, Any]],
    style: dict[str, Any],
    method: str,
    geometries: list[str],
    stem: Path,
) -> list[Path]:
    _configure_style()
    ncols = len(geometries)
    width = 7.15 if ncols == 3 else 4.85
    fig, axes = plt.subplots(2, ncols, figsize=(width, 4.02), squeeze=False)

    for col, geometry in enumerate(geometries):
        axes[0, col].set_title(style["geometry_labels"][geometry], pad=4)
        for row_index, metric in enumerate(style["metrics"]):
            _draw_panel(axes[row_index, col], rows, geometry, method, metric, style)
            if row_index == 0:
                axes[row_index, col].tick_params(labelbottom=False)
            else:
                axes[row_index, col].set_xlabel(r"Sample size $n$", labelpad=2)
            if col == 0:
                axes[row_index, col].set_ylabel(METRIC_LABELS[metric])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unreliable = mpl.lines.Line2D(
        [],
        [],
        color="#555555",
        marker="o",
        markerfacecolor="white",
        linestyle="None",
        markersize=4.2,
        label="Unreliable",
    )
    fig.legend(
        handles + [unreliable],
        labels + ["Unreliable"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.3,
    )
    fig.subplots_adjust(
        left=0.078 if ncols == 3 else 0.115,
        right=0.995,
        bottom=0.15,
        top=0.865,
        wspace=0.27,
        hspace=0.16,
    )

    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return outputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_all(
    rows: list[dict[str, Any]], style: dict[str, Any], output: Path
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    for method in style["methods"]:
        token = method.lower()
        products += _save_figure(
            rows,
            style,
            method,
            style["main_geometries"],
            output / f"figure_e2_{token}_main",
        )
        products += _save_figure(
            rows,
            style,
            method,
            style["appendix_geometries"],
            output / f"appendix_e2_{token}_remaining",
        )

    manifest = {
        "logical_figures": 6,
        "exports": len(products),
        "files": {path.name: _sha256(path) for path in products},
    }
    manifest_path = output / "OUTPUT_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return products

