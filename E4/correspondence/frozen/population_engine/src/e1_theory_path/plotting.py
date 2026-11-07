from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Single": "#3B6EA8", "Multi": "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}
METHOD_LABELS = {"STSP": "STSP", "DR": "Density ridge"}
GEOMETRY_LABELS = {
    "circle_r2": r"$S^1\subset\mathbb{R}^2$",
    "sphere2_r3": r"$S^2\subset\mathbb{R}^3$",
    "clifford_torus_r4": r"Clifford torus $\subset\mathbb{R}^4$",
    "sphere3_r4": r"$S^3\subset\mathbb{R}^4$",
    "sphere2_r10": r"$S^2\subset\mathbb{R}^{10}$",
}
GEOMETRY_ORDER = tuple(GEOMETRY_LABELS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=360, bbox_inches="tight", metadata={"Software": "E1 theory-path experiment"})
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        metadata={"Creator": "E1 theory-path experiment", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return paths


def _five_panel_figure(*, sharey: bool):
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.6), sharex=True, sharey=sharey, squeeze=False)
    flat = list(axes.ravel())
    flat[-1].axis("off")
    return fig, flat[:5]


def plot_scaled_errors(rows: list[dict], output_dir: Path, method: str):
    _style()
    fig, axes = _five_panel_figure(sharey=False)
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRY_ORDER)):
        block = sorted(
            [row for row in rows if row["method"] == method and row["geometry"] == geometry],
            key=lambda row: float(row["sigma"]),
        )
        sigma = np.asarray([float(row["sigma"]) for row in block])
        for estimator, field, label in (
            ("Single", "single_scaled_sigma2", r"Single: $E_\infty/\sigma^2$"),
            ("Multi", "multi_scaled_sigma4", r"Multi: $E_\infty/\sigma^4$"),
        ):
            axis.plot(
                sigma,
                np.asarray([float(row[field]) for row in block]),
                color=COLORS[estimator],
                marker=MARKERS[estimator],
                linewidth=1.6,
                markersize=4.1,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(GEOMETRY_LABELS[geometry], pad=5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
        axis.set_xticks([0.008, 0.01, 0.0125, 0.016, 0.02, 0.025, 0.03])
        axis.set_xticklabels([".008", ".01", ".0125", ".016", ".02", ".025", ".03"], rotation=28)
        if index in (0, 3):
            axis.set_ylabel("Scaled population maximum error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METHOD_LABELS[method]} joint theory path", y=0.945, fontsize=11)
    fig.supxlabel(r"Noise level $\sigma$  ($h_1=3\sigma$, $h_2=6\sigma$)", y=0.01)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.86, hspace=0.38, wspace=0.32)
    return _save(fig, output_dir, f"e1_theory_path_all_manifolds_{method.lower()}_scaled_errors")


def plot_slope_stability(rows: list[dict], output_dir: Path, method: str):
    _style()
    fig, axes = _five_panel_figure(sharey=True)
    window_order = {"smallest_3": 3, "smallest_4": 4, "smallest_5": 5, "all_7": 7}
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRY_ORDER)):
        for estimator in ("Single", "Multi"):
            block = [
                row for row in rows
                if row["method"] == method and row["geometry"] == geometry and row["estimator"] == estimator
            ]
            block.sort(key=lambda row: window_order[row["window"]])
            axis.plot(
                [window_order[row["window"]] for row in block],
                [float(row["slope"]) for row in block],
                color=COLORS[estimator],
                marker=MARKERS[estimator],
                linewidth=1.6,
                markersize=4.2,
                label=estimator,
            )
        axis.axhline(2.0, color=COLORS["Single"], linestyle="--", linewidth=0.8, alpha=0.6)
        axis.axhline(4.0, color=COLORS["Multi"], linestyle="--", linewidth=0.8, alpha=0.6)
        axis.set_title(GEOMETRY_LABELS[geometry], pad=5)
        axis.set_xticks([3, 4, 5, 7])
        axis.set_ylim(1.5, 4.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
        if index in (0, 3):
            axis.set_ylabel("Fitted log–log slope")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METHOD_LABELS[method]} slope stability", y=0.945, fontsize=11)
    fig.supxlabel("Number of smallest-noise points retained in the fixed nested window", y=0.01)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, hspace=0.38, wspace=0.24)
    return _save(fig, output_dir, f"e1_theory_path_all_manifolds_{method.lower()}_slope_stability")


def render(results_dir: Path):
    output_dir = results_dir / "figures"
    error_rows = load_csv(results_dir / "theory_path_errors.csv")
    slope_rows = load_csv(results_dir / "slope_stability.csv")
    outputs = []
    for method in ("STSP", "DR"):
        outputs += plot_scaled_errors(error_rows, output_dir, method)
        outputs += plot_slope_stability(slope_rows, output_dir, method)
    manifest = {
        "schema_version": "E1_THEORY_PATH_FIGURE_MANIFEST_V1",
        "status": "PASS",
        "scaled_error_definition": {
            "Single": "population E_infinity divided by sigma^2",
            "Multi": "population E_infinity divided by sigma^4",
        },
        "slope_windows": "fixed nested windows containing the 3, 4, 5, or all 7 smallest declared sigma values",
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (output_dir / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
