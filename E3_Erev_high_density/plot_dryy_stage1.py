from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


GEOMETRIES = (
    ("circle_r2", r"$S^1 \subset \mathbb{R}^2$"),
    ("sphere2_r3", r"$S^2 \subset \mathbb{R}^3$"),
    ("clifford_torus_r4", r"Clifford torus $\subset \mathbb{R}^4$"),
    ("sphere3_r4", r"$S^3 \subset \mathbb{R}^4$"),
    ("sphere2_r10", r"$S^2 \subset \mathbb{R}^{10}$"),
)
COLORS = {"Single": "#3B75AF", "Multi": "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}
XTICKS = (0.06, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
XTICK_LABELS = (".06", ".1", ".2", ".5", "1", "2", "5")


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid reliable value: {value!r}")
    return normalized == "true"


def read_rows(path: Path, circle_path: Path, method: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == method]
    with circle_path.open(newline="", encoding="utf-8") as handle:
        circle_rows = [
            row for row in csv.DictReader(handle)
            if row["method"] == method and row["geometry"] == "circle_r2"
        ]
    for row in circle_rows:
        row["reliable"] = row["reliable_query_gate"]
    rows.extend(circle_rows)
    required = {
        "method", "geometry", "estimator", "scale", "reliable", "E_reverse"
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("input is not an E3 high-density Stage-1 aggregate CSV")
    expected = len(GEOMETRIES) * 2 * 28
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {method} rows, found {len(rows)}")
    identities = {
        (row["geometry"], row["estimator"], float(row["scale"])) for row in rows
    }
    if len(identities) != expected:
        raise ValueError("duplicate or missing geometry/estimator/scale rows")
    return rows


def reliable_multi_lower_intervals(
    rows: list[dict], geometry: str
) -> list[tuple[float, float]]:
    by_estimator = {
        estimator: {
            float(row["scale"]): row
            for row in rows
            if row["geometry"] == geometry and row["estimator"] == estimator
        }
        for estimator in ("Single", "Multi")
    }
    grid = sorted(set(by_estimator["Single"]) & set(by_estimator["Multi"]))
    if len(grid) != 28:
        raise ValueError(f"expected 28 common scales for {geometry}, found {len(grid)}")
    favorable = []
    for scale in grid:
        single = by_estimator["Single"][scale]
        multi = by_estimator["Multi"][scale]
        single_error = float(single["E_reverse"])
        multi_error = float(multi["E_reverse"])
        favorable.append(
            as_bool(single["reliable"])
            and as_bool(multi["reliable"])
            and np.isfinite(single_error)
            and np.isfinite(multi_error)
            and multi_error < single_error
        )
    intervals: list[tuple[float, float]] = []
    start = 0
    while start < len(grid):
        if not favorable[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(grid) and favorable[end + 1]:
            end += 1
        # A shaded interval must contain at least two adjacent tested scales.
        if end > start:
            intervals.append((grid[start], grid[end]))
        start = end + 1
    return intervals


def qualifying_intervals(
    audit_path: Path, method: str, rows: list[dict]
) -> dict[str, list[tuple[float, float]]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("Stage-1 audit did not pass")
    audited_geometries = {
        decision["geometry"]
        for decision in audit["stage2_decisions"]
        if decision["method"] == method
    }
    expected_audited = {geometry for geometry, _ in GEOMETRIES if geometry != "circle_r2"}
    if audited_geometries != expected_audited:
        raise ValueError("Stage-1 audit geometry set mismatch")
    return {
        geometry: reliable_multi_lower_intervals(rows, geometry)
        for geometry, _ in GEOMETRIES
    }


def plot_method(
    csv_path: Path,
    circle_csv_path: Path,
    audit_path: Path,
    output_dir: Path,
    method: str,
) -> Path:
    rows = read_rows(csv_path, circle_csv_path, method)
    intervals = qualifying_intervals(audit_path, method, rows)
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.8), sharex=True)
    for axis, (geometry, title) in zip(axes.flat, GEOMETRIES):
        subset = [row for row in rows if row["geometry"] == geometry]
        for left, right in intervals[geometry]:
            axis.axvspan(left, right, color="#DDEFE8", alpha=0.9, linewidth=0,
                         zorder=0)
        for estimator in ("Single", "Multi"):
            series = sorted(
                (row for row in subset if row["estimator"] == estimator),
                key=lambda row: float(row["scale"]),
            )
            x = np.asarray([float(row["scale"]) for row in series])
            y = np.asarray([float(row["E_reverse"]) for row in series])
            reliable = np.asarray([as_bool(row["reliable"]) for row in series])
            finite = np.isfinite(y)
            color = COLORS[estimator]
            marker = MARKERS[estimator]
            axis.plot(x[finite], y[finite], color=color, linewidth=1.25, zorder=1)
            good = finite & reliable
            bad = finite & ~reliable
            axis.scatter(
                x[good], y[good], s=15, marker=marker, color=color,
                edgecolors=color, linewidths=0.55, zorder=3,
            )
            axis.scatter(
                x[bad], y[bad], s=17, marker=marker, facecolors="white",
                edgecolors=color, linewidths=0.9, zorder=4,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.055, 5.5)
        axis.set_xticks(XTICKS, XTICK_LABELS)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=2.5, width=0.7)

    for axis in axes.flat[len(GEOMETRIES):]:
        axis.set_visible(False)

    scale_symbol = "h" if method == "DR" else r"r_1"
    axes[0, 0].set_ylabel(
        r"Reverse coverage error $E_{\rm rev}$", fontsize=8, labelpad=4
    )
    axes[1, 0].set_ylabel(
        r"Reverse coverage error $E_{\rm rev}$", fontsize=8, labelpad=4
    )
    fig.supxlabel(f"Scale ${scale_symbol}$", y=0.035, fontsize=9)
    legend = [
        Line2D([], [], color=COLORS["Single"], marker="o", markersize=4,
               linewidth=1.25, label="Single"),
        Line2D([], [], color=COLORS["Multi"], marker="s", markersize=4,
               linewidth=1.25, label="Multi"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor="#666666", markeredgewidth=0.9, markersize=4,
               label="unreliable"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.8, columnspacing=1.4)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.86,
                        wspace=0.25, hspace=0.36)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"e3_erev_stage1_{method.lower()}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--circle-csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for method in ("DR", "YY"):
        print(plot_method(
            args.csv, args.circle_csv, args.audit, args.output_dir, method
        ))


if __name__ == "__main__":
    main()
