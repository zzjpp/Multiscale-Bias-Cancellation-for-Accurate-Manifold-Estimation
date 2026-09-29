from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _load_config(root: Path) -> dict:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def _load_rows(path: Path, spec: dict) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    scale_key = spec["scale_column"]
    estimator_key = spec["estimator_column"]
    for row in rows:
        row[scale_key] = float(row[scale_key])
        row["reliable"] = str(row["reliable"]).lower() == "true"
        for source in spec["metric_columns"].values():
            for suffix in ("q25", "median", "q75"):
                row[f"{source}_{suffix}"] = float(row[f"{source}_{suffix}"])

    expected_rows = 2 * len(spec["scale_grid"])
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
    if {str(row[estimator_key]) for row in rows} != {"Single", "Multi"}:
        raise ValueError(f"{path.name}: estimator labels do not match the frozen contract")
    for estimator in ("Single", "Multi"):
        observed = sorted(
            float(row[scale_key]) for row in rows if row[estimator_key] == estimator
        )
        if observed != sorted(float(value) for value in spec["scale_grid"]):
            raise ValueError(f"{path.name}: incomplete or modified scale grid for {estimator}")
    return rows


def _draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    spec: dict,
    metric: str,
    ylabel: str,
    colors: dict[str, str],
    markers: dict[str, str],
    band_alpha: float,
) -> None:
    scale_key = spec["scale_column"]
    estimator_key = spec["estimator_column"]
    source = spec["metric_columns"][metric]
    for estimator in ("Single", "Multi"):
        current = sorted(
            (row for row in rows if row[estimator_key] == estimator),
            key=lambda row: float(row[scale_key]),
        )
        x = np.asarray([row[scale_key] for row in current], dtype=float)
        y = np.asarray([row[f"{source}_median"] for row in current], dtype=float)
        lower = np.asarray([row[f"{source}_q25"] for row in current], dtype=float)
        upper = np.asarray([row[f"{source}_q75"] for row in current], dtype=float)
        reliable = np.asarray([row["reliable"] for row in current], dtype=bool)
        finite = np.isfinite(y)

        ax.plot(x, y, color=colors[estimator], linewidth=1.6)
        ax.fill_between(
            x, lower, upper, color=colors[estimator], alpha=band_alpha, linewidth=0
        )
        ax.scatter(
            x[finite & reliable], y[finite & reliable], s=28,
            marker=markers[estimator], color=colors[estimator], zorder=3,
        )
        ax.scatter(
            x[finite & ~reliable], y[finite & ~reliable], s=28,
            marker=markers[estimator], facecolor="white",
            edgecolor=colors[estimator], linewidth=1.2, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_one(root: Path, output_dir: Path, spec: dict, style: dict) -> list[Path]:
    colors = style["colors"]
    markers = style["markers"]
    rows = _load_rows(root / spec["input_csv"], spec)

    plt.rcParams.update(
        {
            "font.size": style["font_size"],
            "axes.labelsize": style["axes_label_size"],
            "legend.fontsize": style["legend_font_size"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=tuple(style["figure_size"]))
    _draw_panel(
        axes[0], rows, spec, "maximum", r"Maximum error $E_\infty$", colors,
        markers, style["band_alpha"],
    )
    _draw_panel(
        axes[1], rows, spec, "average", r"Average error $E_{\rm avg}$", colors,
        markers, style["band_alpha"],
    )

    handles = [
        Line2D(
            [0], [0], color=colors[name], marker=markers[name], label=name
        )
        for name in ("Single", "Multi")
    ]
    handles.append(
        Line2D(
            [0], [0], marker="o", markerfacecolor="white",
            markeredgecolor="0.4", linestyle="None", label="Unreliable",
        )
    )
    figure.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.99, wspace=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / spec["output_stem"]
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=style["png_dpi"], bbox_inches="tight")
    plt.close(figure)
    return [pdf, png]


def build_all(root: Path, output_dir: Path | None = None) -> list[Path]:
    root = Path(root).resolve()
    output_dir = (root / "figures") if output_dir is None else Path(output_dir).resolve()
    config = _load_config(root)
    outputs: list[Path] = []
    for name in ("stsp", "dr", "yy"):
        outputs.extend(_plot_one(root, output_dir, config["figures"][name], config["style"]))
    return outputs
