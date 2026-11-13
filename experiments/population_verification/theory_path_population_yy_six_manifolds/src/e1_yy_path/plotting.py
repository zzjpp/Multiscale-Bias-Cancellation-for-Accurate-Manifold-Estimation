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
GEOMETRY_LABELS = {
    "corrugated_torus_r3": r"Corrugated torus $\subset\mathbb{R}^3$",
}
GEOMETRY_ORDER = tuple(GEOMETRY_LABELS)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style():
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


def five_panel(sharey=False):
    figure, axes = plt.subplots(1, 1, figsize=(4.2, 3.25), sharex=True, sharey=sharey, squeeze=False)
    return figure, list(axes.ravel())


def save(figure, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(outputs[0], dpi=360, bbox_inches="tight", metadata={"Software": "E1 YY theory path"})
    figure.savefig(
        outputs[1],
        bbox_inches="tight",
        metadata={"Creator": "E1 YY theory path", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return outputs


def plot_scaled(rows, output_dir):
    style()
    figure, axes = five_panel(False)
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRY_ORDER)):
        block = sorted([row for row in rows if row["geometry"] == geometry], key=lambda row: float(row["sigma"]))
        sigma = np.asarray([float(row["sigma"]) for row in block])
        for estimator, field, label in (
            ("Single", "single_scaled_sigma2", r"Single: $E_\infty/\sigma^2$"),
            ("Multi", "multi_scaled_sigma4", r"Multi: $E_\infty/\sigma^4$"),
        ):
            axis.plot(
                sigma,
                [float(row[field]) for row in block],
                color=COLORS[estimator],
                marker=MARKERS[estimator],
                linewidth=1.6,
                markersize=4.1,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(GEOMETRY_LABELS[geometry], pad=5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks([0.008, 0.01, 0.0125, 0.016, 0.02, 0.025, 0.03])
        axis.set_xticklabels([".008", ".01", ".0125", ".016", ".02", ".025", ".03"], rotation=28)
        if index in (0, 3):
            axis.set_ylabel("Scaled population maximum error")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("YY joint theory path", y=0.945, fontsize=11)
    figure.supxlabel(
        r"Noise level $\sigma$  ($r_0=3\sigma$, $r_1/\sigma\in\{1.25,5,8\}$, "
        r"$r_2=12\sigma\sqrt{\log(1/\sigma)}$)",
        y=0.01,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.86, hspace=0.38, wspace=0.32)
    return save(figure, output_dir, "e1_theory_path_all_manifolds_yy_scaled_errors")


def plot_slopes(rows, output_dir):
    style()
    figure, axes = five_panel(True)
    window_order = {"smallest_3": 3, "smallest_4": 4, "smallest_5": 5, "all_7": 7}
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRY_ORDER)):
        for estimator in ("Single", "Multi"):
            block = [
                row for row in rows if row["geometry"] == geometry and row["estimator"] == estimator
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
        axis.spines[["top", "right"]].set_visible(False)
        if index in (0, 3):
            axis.set_ylabel("Fitted log–log slope")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("YY slope stability", y=0.945, fontsize=11)
    figure.supxlabel("Number of smallest-noise points retained in the fixed nested window", y=0.01)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, hspace=0.38, wspace=0.24)
    return save(figure, output_dir, "e1_theory_path_all_manifolds_yy_slope_stability")


def render(results_dir):
    results_dir = Path(results_dir)
    output_dir = results_dir / "figures"
    outputs = plot_scaled(load_csv(results_dir / "theory_path_errors.csv"), output_dir)
    outputs += plot_slopes(load_csv(results_dir / "slope_stability.csv"), output_dir)
    manifest = {
        "schema_version": "E1_YY_THEORY_PATH_FIGURE_MANIFEST_V1",
        "status": "PASS",
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (output_dir / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
