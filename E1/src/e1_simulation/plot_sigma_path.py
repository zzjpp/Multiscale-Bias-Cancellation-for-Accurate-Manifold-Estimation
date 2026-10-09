from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("STSP", "DR")
ALL_METHODS = ("STSP", "DR", "YY")
GEOMETRIES = (
    "circle_r2",
    "sphere2_r3",
    "clifford_torus_r4",
    "sphere3_r4",
    "sphere2_r10",
    "corrugated_torus_r3",
)
MAIN_GEOMETRIES = GEOMETRIES
APPENDIX_GEOMETRIES: tuple[str, ...] = ()
YY_MAIN = GEOMETRIES
YY_APPENDIX: tuple[str, ...] = ()
SIGMA_GRID = (0.008, 0.01, 0.0125, 0.016, 0.02, 0.025, 0.03)
SLOPE_WINDOWS = {"smallest_3": 3, "smallest_4": 4, "smallest_5": 5, "all_7": 7}
YY_GRID = (0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65)
TITLES = {
    "circle_r2": r"$S^1\subset\mathbb{R}^2$",
    "sphere2_r3": r"$S^2\subset\mathbb{R}^3$",
    "clifford_torus_r4": r"Clifford torus $\subset\mathbb{R}^4$",
    "sphere3_r4": r"$S^3\subset\mathbb{R}^4$",
    "sphere2_r10": r"$S^2\subset\mathbb{R}^{10}$",
    "corrugated_torus_r3": r"Corrugated torus $\subset\mathbb{R}^3$",
}
METHOD_LABELS = {"STSP": "STSP", "DR": "Density ridge", "YY": "Yao--Yau"}
COLORS = {"Single": "#3B6EA8", "Multi": "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_theory_rows(path: Path) -> list[dict]:
    numeric = {
        "sigma",
        "h1",
        "h2",
        "query_count",
        "single_E_infinity",
        "single_E_average",
        "multi_E_infinity",
        "multi_E_average",
        "single_scaled_sigma2",
        "multi_scaled_sigma4",
    }
    return [
        {key: (float(value) if key in numeric else value) for key, value in row.items()}
        for row in _read_csv(path)
    ]


def load_slope_rows(path: Path) -> list[dict]:
    numeric = {"target_order", "point_count", "sigma_min", "sigma_max", "slope", "intercept", "r_squared"}
    return [
        {key: (float(value) if key in numeric else value) for key, value in row.items()}
        for row in _read_csv(path)
    ]


def load_yy_theory_rows(path: Path) -> list[dict]:
    numeric = {
        "sigma",
        "r0",
        "r1_low",
        "r1_single",
        "r1_high",
        "r2",
        "query_count",
        "single_E_infinity",
        "single_E_average",
        "multi_E_infinity",
        "multi_E_average",
        "single_scaled_sigma2",
        "multi_scaled_sigma4",
    }
    return [
        {key: (float(value) if key in numeric else value) for key, value in row.items()}
        for row in _read_csv(path)
    ]


def load_yy_rows(path: Path) -> list[dict]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                **row,
                "scale": float(row["scale"]),
                "sigma": float(row["sigma"]),
                "single_error": float(row["single_error"]),
                "multi_error": float(row["multi_error"]),
            }
        )
    return rows


def validate_theory_rows(rows: list[dict]) -> None:
    if len(rows) != 84:
        raise ValueError(f"Expected 84 STSP/DR sigma-path rows, found {len(rows)}")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["geometry"])].append(row)
    if set(grouped) != {(method, geometry) for method in METHODS for geometry in GEOMETRIES}:
        raise ValueError("Incomplete STSP/DR method-geometry grid")
    for key, block in grouped.items():
        block.sort(key=lambda row: row["sigma"])
        if not np.allclose([row["sigma"] for row in block], SIGMA_GRID, rtol=0.0, atol=1e-14):
            raise ValueError(f"Sigma grid mismatch for {key}")
        for row in block:
            sigma = row["sigma"]
            if not np.isclose(row["h1"], 3.0 * sigma, rtol=0.0, atol=1e-14):
                raise ValueError(f"h1/sigma mismatch for {key} at sigma={sigma}")
            if not np.isclose(row["h2"], 6.0 * sigma, rtol=0.0, atol=1e-14):
                raise ValueError(f"h2/sigma mismatch for {key} at sigma={sigma}")
            expected_queries = 128 if row["geometry"] == "corrugated_torus_r3" else 1
            if int(row["query_count"]) != expected_queries:
                raise ValueError(f"Unexpected homogeneous-query count for {key}")
            if row["single_E_infinity"] <= 0.0 or row["multi_E_infinity"] <= 0.0:
                raise ValueError(f"Nonpositive error for {key} at sigma={sigma}")
            if not np.isclose(
                row["single_scaled_sigma2"], row["single_E_infinity"] / sigma**2, rtol=1e-13, atol=0.0
            ):
                raise ValueError(f"Single scaling mismatch for {key} at sigma={sigma}")
            if not np.isclose(
                row["multi_scaled_sigma4"], row["multi_E_infinity"] / sigma**4, rtol=1e-13, atol=0.0
            ):
                raise ValueError(f"Multi scaling mismatch for {key} at sigma={sigma}")


def validate_slope_rows(rows: list[dict]) -> None:
    if len(rows) != 96:
        raise ValueError(f"Expected 96 fixed-window slope rows, found {len(rows)}")
    expected = {
        (method, geometry, estimator, window)
        for method in METHODS
        for geometry in GEOMETRIES
        for estimator in ("Single", "Multi")
        for window in SLOPE_WINDOWS
    }
    actual = {(row["method"], row["geometry"], row["estimator"], row["window"]) for row in rows}
    if actual != expected:
        raise ValueError("Incomplete fixed nested slope-window grid")
    for row in rows:
        target = 2.0 if row["estimator"] == "Single" else 4.0
        if row["target_order"] != target:
            raise ValueError("Incorrect declared slope target")
        if row["point_count"] != SLOPE_WINDOWS[row["window"]]:
            raise ValueError("Slope-window point count mismatch")
        if not np.all(np.isfinite([row["slope"], row["intercept"], row["r_squared"]])):
            raise ValueError("Nonfinite fixed-window log-log fit")
        if not 0.0 <= row["r_squared"] <= 1.0:
            raise ValueError("Invalid fixed-window R-squared")


def validate_yy_theory_rows(rows: list[dict]) -> None:
    if len(rows) != 42:
        raise ValueError(f"Expected 42 YY sigma-path rows, found {len(rows)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] != "YY":
            raise ValueError("Non-YY row in YY sigma-path input")
        grouped[row["geometry"]].append(row)
    if set(grouped) != set(GEOMETRIES):
        raise ValueError("Incomplete YY sigma-path geometry grid")
    expected_queries = {
        "circle_r2": 1,
        "sphere2_r3": 1,
        "clifford_torus_r4": 32,
        "sphere3_r4": 1,
        "sphere2_r10": 34,
        "corrugated_torus_r3": 128,
    }
    for geometry, block in grouped.items():
        block.sort(key=lambda row: row["sigma"])
        if not np.allclose([row["sigma"] for row in block], SIGMA_GRID, rtol=0.0, atol=1e-14):
            raise ValueError(f"YY sigma grid mismatch for {geometry}")
        for row in block:
            sigma = row["sigma"]
            expected_r2 = 12.0 * sigma * np.sqrt(np.log(1.0 / sigma))
            ratios = [row["r1_low"] / sigma, row["r1_single"] / sigma, row["r1_high"] / sigma]
            if not np.allclose(ratios, [1.25, 5.0, 8.0], rtol=0.0, atol=1e-13):
                raise ValueError(f"YY transverse-radius path mismatch for {geometry} at sigma={sigma}")
            if not np.isclose(row["r0"], 3.0 * sigma, rtol=0.0, atol=1e-14):
                raise ValueError(f"YY r0 path mismatch for {geometry} at sigma={sigma}")
            if not np.isclose(row["r2"], expected_r2, rtol=1e-14, atol=0.0):
                raise ValueError(f"YY r2 path mismatch for {geometry} at sigma={sigma}")
            if int(row["query_count"]) != expected_queries[geometry]:
                raise ValueError(f"YY query count mismatch for {geometry}")
            if row["single_E_infinity"] <= 0.0 or row["multi_E_infinity"] <= 0.0:
                raise ValueError(f"Nonpositive YY error for {geometry} at sigma={sigma}")
            if not np.isclose(
                row["single_scaled_sigma2"], row["single_E_infinity"] / sigma**2, rtol=1e-13, atol=0.0
            ):
                raise ValueError("YY Single scaling mismatch")
            if not np.isclose(
                row["multi_scaled_sigma4"], row["multi_E_infinity"] / sigma**4, rtol=1e-13, atol=0.0
            ):
                raise ValueError("YY Multi scaling mismatch")


def validate_yy_slope_rows(rows: list[dict]) -> None:
    if len(rows) != 48:
        raise ValueError(f"Expected 48 YY fixed-window slope rows, found {len(rows)}")
    expected = {
        (geometry, estimator, window)
        for geometry in GEOMETRIES
        for estimator in ("Single", "Multi")
        for window in SLOPE_WINDOWS
    }
    actual = {(row["geometry"], row["estimator"], row["window"]) for row in rows}
    if actual != expected or any(row["method"] != "YY" for row in rows):
        raise ValueError("Incomplete YY fixed nested slope-window grid")
    for row in rows:
        target = 2.0 if row["estimator"] == "Single" else 4.0
        if row["target_order"] != target:
            raise ValueError("Incorrect YY slope target")
        if row["point_count"] != SLOPE_WINDOWS[row["window"]]:
            raise ValueError("YY slope-window point count mismatch")
        if not np.all(np.isfinite([row["slope"], row["intercept"], row["r_squared"]])):
            raise ValueError("Nonfinite YY fixed-window log-log fit")
        if not 0.0 <= row["r_squared"] <= 1.0:
            raise ValueError("Invalid YY fixed-window R-squared")


def validate_registered_fits(error_rows: list[dict], slope_rows: list[dict]) -> None:
    """Check fitted numbers, not whether the observed order agrees with theory."""
    for row in slope_rows:
        block = sorted(
            (item for item in error_rows if item["method"] == row["method"]
             and item["geometry"] == row["geometry"]),
            key=lambda item: item["sigma"],
        )[:int(row["point_count"])]
        x = np.log([item["sigma"] for item in block])
        y = np.log([item[f"{row['estimator'].lower()}_E_infinity"] for item in block])
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = 1.0 - np.sum((y - (slope*x + intercept))**2) / np.sum((y - y.mean())**2)
        if not np.allclose(
            [row["slope"], row["intercept"], row["r_squared"]],
            [slope, intercept, r_squared], rtol=1e-11, atol=1e-12,
        ):
            raise ValueError(f"Registered fit differs from frozen data: {row}")
        if row["sigma_min"] != block[0]["sigma"] or row["sigma_max"] != block[-1]["sigma"]:
            raise ValueError("Registered fit does not use its declared fixed window")


def validate_yy_rows(rows: list[dict]) -> None:
    if len(rows) != 40:
        raise ValueError(f"Expected 40 Yao--Yau rows, found {len(rows)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] != "Yao--Yau":
            raise ValueError("Non-YY row in YY input")
        grouped[row["geometry"]].append(row)
    if set(grouped) != set(GEOMETRIES):
        raise ValueError("Incomplete YY geometry grid")
    for geometry, block in grouped.items():
        block.sort(key=lambda row: row["scale"])
        if not np.allclose([row["scale"] for row in block], YY_GRID, rtol=0.0, atol=1e-14):
            raise ValueError(f"YY radius grid mismatch for {geometry}")
        if any(row["single_error"] <= 0.0 or row["multi_error"] <= 0.0 for row in block):
            raise ValueError(f"Nonpositive YY error for {geometry}")


def validate_numerical_evidence(
    convergence_path: Path,
    audit_path: Path,
    gpu_gate_path: Path,
) -> tuple[dict, dict]:
    convergence = _read_csv(convergence_path)
    if len(convergence) != 84:
        raise ValueError(f"Expected 84 numerical-convergence rows, found {len(convergence)}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    gate = json.loads(gpu_gate_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise ValueError("Theory-path audit is not PASS")
    if gate.get("status") != "PASS" or not all(gate.get("checks", {}).values()):
        raise ValueError("CPU/GPU equivalence gate is not PASS")
    maxima = audit["numerical_maxima"]
    if maxima["resolution_difference_over_minimum_multi_error"] >= 1e-3:
        raise ValueError("Quadrature error is not sufficiently below the minimum fourth-order error")
    if maxima["root_uncertainty_over_minimum_multi_error"] >= 1e-3:
        raise ValueError("Root uncertainty is not sufficiently below the minimum fourth-order error")
    return audit, gate


def validate_yy_numerical_evidence(
    convergence_path: Path,
    audit_path: Path,
    gpu_gate_path: Path,
) -> tuple[dict, dict]:
    convergence = _read_csv(convergence_path)
    if len(convergence) != 42:
        raise ValueError(f"Expected 42 YY numerical-convergence rows, found {len(convergence)}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    gate = json.loads(gpu_gate_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise ValueError("YY theory-path audit is not PASS")
    if gate.get("status") != "PASS" or not all(gate.get("checks", {}).values()):
        raise ValueError("YY CPU/GPU equivalence gate is not PASS")
    if audit["numerical_maxima"]["metric_difference_over_minimum_multi_error"] >= 0.01:
        raise ValueError("YY production/reference error is not sufficiently below the minimum fourth-order error")
    return audit, gate


def _style() -> None:
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


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=360, bbox_inches="tight", metadata={"Software": "E1 sigma-path reproduction"})
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "E1 sigma-path reproduction", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return [pdf, png]


def _six_panel() -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.6), sharex=True, squeeze=False)
    return fig, list(axes.ravel())


def _log_slope(block: list[dict], field: str) -> float:
    sigma = np.asarray([row["sigma"] for row in block], dtype=float)
    error = np.asarray([row[field] for row in block], dtype=float)
    return float(np.polyfit(np.log(sigma), np.log(error), 1)[0])


def render_actual_errors(
    rows: list[dict],
    method: str,
    metric: str,
    geometries: tuple[str, ...],
    split: str,
    output_dir: Path,
) -> list[Path]:
    """Render unscaled E_avg or E_infinity in the original E1 bar-line style."""
    if metric not in {"average", "infinity"}:
        raise ValueError(f"Unsupported actual-error metric: {metric}")
    _style()
    if split != "main" or len(geometries) != 6:
        raise ValueError("The final E1 release uses one six-manifold 2x3 panel per metric")
    fig, grid = plt.subplots(2, 3, figsize=(9.3, 5.6), sharex=True, sharey=True, squeeze=False)
    axes = list(grid.ravel())
    selected = [
        row
        for row in rows
        if row["method"] == method and row["geometry"] in geometries
    ]
    fields = (f"single_E_{metric}", f"multi_E_{metric}")
    values = [float(row[field]) for row in selected for field in fields]
    floor = 10.0 ** (np.floor(np.log10(min(values))) - 0.25)
    ceiling = 10.0 ** (np.ceil(np.log10(max(values))) + 0.12)
    x = np.arange(len(SIGMA_GRID), dtype=float)
    bar_width = 0.34
    for axis, geometry in zip(axes, geometries):
        block = sorted(
            [row for row in rows if row["method"] == method and row["geometry"] == geometry],
            key=lambda row: row["sigma"],
        )
        single = np.asarray([row[fields[0]] for row in block], dtype=float)
        multi = np.asarray([row[fields[1]] for row in block], dtype=float)
        single_x = x - bar_width / 2
        multi_x = x + bar_width / 2
        axis.bar(
            single_x,
            single,
            bar_width,
            color=COLORS["Single"],
            alpha=0.78,
            edgecolor=COLORS["Single"],
            linewidth=0.55,
            label="Single",
            zorder=2,
        )
        axis.bar(
            multi_x,
            multi,
            bar_width,
            color=COLORS["Multi"],
            alpha=0.78,
            edgecolor=COLORS["Multi"],
            linewidth=0.55,
            label="Multi",
            zorder=2,
        )
        axis.plot(single_x, single, color=COLORS["Single"], marker="o", markersize=3.1, linewidth=1.15, zorder=3)
        axis.plot(multi_x, multi, color=COLORS["Multi"], marker="s", markersize=3.0, linewidth=1.15, zorder=3)
        axis.set_yscale("log")
        axis.set_ylim(floor, ceiling)
        axis.set_title(TITLES[geometry], pad=5)
        axis.set_xticks(x, [".008", ".01", ".0125", ".016", ".02", ".025", ".03"])
        axis.tick_params(axis="x", rotation=32, pad=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        axis.text(
            0.035,
            0.955,
            f"slopes: {_log_slope(block, fields[0]):.2f} / {_log_slope(block, fields[1]):.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "0.78",
                "linewidth": 0.45,
                "alpha": 0.92,
            },
        )
    metric_tex = r"E_{\mathrm{avg}}" if metric == "average" else r"E_{\infty}"
    metric_word = "average" if metric == "average" else "maximum"
    for index in (0, 3):
        axes[index].set_ylabel(rf"Population {metric_word} error ${metric_tex}$")
    if method == "YY":
        xlabel = (
            r"Noise level $\sigma$  ($r_0=3\sigma$, $r_1/\sigma\in\{1.25,5,8\}$, "
            r"$r_2=12\sigma\sqrt{\log(1/\sigma)}$)"
        )
    else:
        xlabel = r"Noise level $\sigma$  ($h_1=3\sigma$, $h_2=6\sigma$)"
    fig.supxlabel(xlabel, y=0.015, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.suptitle(f"{METHOD_LABELS[method]} population {metric_word} error", y=0.935, fontsize=10.5)
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.15,
        top=0.85,
        hspace=0.38,
        wspace=0.30,
    )
    return _save(fig, output_dir, f"e1_{method.lower()}_population_{metric}_{split}")


def render_scaled_errors(rows: list[dict], method: str, output_dir: Path) -> list[Path]:
    _style()
    fig, axes = _six_panel()
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRIES)):
        block = sorted(
            [row for row in rows if row["method"] == method and row["geometry"] == geometry],
            key=lambda row: row["sigma"],
        )
        sigma = np.asarray([row["sigma"] for row in block])
        axis.plot(
            sigma,
            [row["single_scaled_sigma2"] for row in block],
            color=COLORS["Single"],
            marker=MARKERS["Single"],
            linewidth=1.6,
            markersize=4.1,
            label=r"Single: $E_\infty/\sigma^2$",
        )
        axis.plot(
            sigma,
            [row["multi_scaled_sigma4"] for row in block],
            color=COLORS["Multi"],
            marker=MARKERS["Multi"],
            linewidth=1.6,
            markersize=4.1,
            label=r"Multi: $E_\infty/\sigma^4$",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(TITLES[geometry], pad=5)
        axis.set_xticks(SIGMA_GRID)
        axis.set_xticklabels([".008", ".01", ".0125", ".016", ".02", ".025", ".03"], rotation=28)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        if index in (0, 3):
            axis.set_ylabel("Scaled population maximum error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METHOD_LABELS[method]} joint small-noise path", y=0.945, fontsize=11)
    if method == "YY":
        xlabel = (
            r"Noise level $\sigma$  ($r_0=3\sigma$, $r_1/\sigma\in\{1.25,5,8\}$, "
            r"$r_2=12\sigma\sqrt{\log(1/\sigma)}$)"
        )
    else:
        xlabel = r"Noise level $\sigma$  ($h_1=3\sigma$, $h_2=6\sigma$)"
    fig.supxlabel(xlabel, y=0.01)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.86, hspace=0.38, wspace=0.32)
    return _save(fig, output_dir, f"e1_{method.lower()}_population_main")


def render_slope_stability(rows: list[dict], method: str, output_dir: Path) -> list[Path]:
    _style()
    fig, axes = _six_panel()
    for index, (axis, geometry) in enumerate(zip(axes, GEOMETRIES)):
        for estimator in ("Single", "Multi"):
            block = [
                row
                for row in rows
                if row["method"] == method
                and row["geometry"] == geometry
                and row["estimator"] == estimator
            ]
            block.sort(key=lambda row: SLOPE_WINDOWS[row["window"]])
            axis.plot(
                [SLOPE_WINDOWS[row["window"]] for row in block],
                [row["slope"] for row in block],
                color=COLORS[estimator],
                marker=MARKERS[estimator],
                linewidth=1.6,
                markersize=4.2,
                label=estimator,
            )
        axis.axhline(2.0, color=COLORS["Single"], linestyle="--", linewidth=0.8, alpha=0.6)
        axis.axhline(4.0, color=COLORS["Multi"], linestyle="--", linewidth=0.8, alpha=0.6)
        axis.set_title(TITLES[geometry], pad=5)
        axis.set_xticks([3, 4, 5, 7])
        axis.set_ylim(1.5, 4.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        if index in (0, 3):
            axis.set_ylabel("Fitted log--log slope")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METHOD_LABELS[method]} slope stability", y=0.945, fontsize=11)
    fig.supxlabel("Number of smallest-noise points retained in the fixed nested window", y=0.01)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.86, hspace=0.38, wspace=0.24)
    return _save(fig, output_dir, f"e1_{method.lower()}_population_appendix")


def _yy_slope(rows: list[dict], field: str) -> float:
    x = np.log(np.asarray([row["scale"] for row in rows]))
    y = np.log(np.asarray([row[field] for row in rows]))
    return float(np.polyfit(x, y, 1)[0])


def render_yy(rows: list[dict], geometries: tuple[str, ...], split: str, output_dir: Path) -> list[Path]:
    _style()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["geometry"]].append(row)
    for block in grouped.values():
        block.sort(key=lambda row: row["scale"])
    fig, axes = plt.subplots(1, len(geometries), figsize=(8.0 if split == "main" else 5.45, 2.55), sharey=True)
    axes = np.atleast_1d(axes)
    selected = [row for geometry in geometries for row in grouped[geometry]]
    values = [row[field] for row in selected for field in ("single_error", "multi_error")]
    floor = 10.0 ** (np.floor(np.log10(min(values))) - 0.35)
    ceiling = 10.0 ** (np.ceil(np.log10(max(values))) + 0.18)
    x = np.arange(len(YY_GRID))
    width = 0.34
    for axis, geometry in zip(axes, geometries):
        block = grouped[geometry]
        single = np.asarray([row["single_error"] for row in block])
        multi = np.asarray([row["multi_error"] for row in block])
        axis.bar(x - width / 2, single, width, color=COLORS["Single"], alpha=0.78, label="Single")
        axis.bar(x + width / 2, multi, width, color=COLORS["Multi"], alpha=0.78, label="Multi")
        axis.plot(x - width / 2, single, color=COLORS["Single"], marker="o", markersize=3.1, linewidth=1.15)
        axis.plot(x + width / 2, multi, color=COLORS["Multi"], marker="s", markersize=3.0, linewidth=1.15)
        axis.set_yscale("log")
        axis.set_ylim(floor, ceiling)
        axis.set_title(TITLES[geometry], pad=5)
        axis.text(
            0.035,
            0.955,
            f"slopes: {_yy_slope(block, 'single_error'):.2f} / {_yy_slope(block, 'multi_error'):.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.78", "linewidth": 0.45},
        )
        axis.set_xticks(x, [f"{value:g}".lstrip("0") for value in YY_GRID])
        axis.tick_params(axis="x", rotation=32, pad=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    axes[0].set_ylabel(r"Population maximum error $E_\infty$")
    fig.supxlabel(r"$r_1$", y=0.015)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(left=0.095 if split == "main" else 0.13, right=0.995, bottom=0.25, top=0.82, wspace=0.30)
    return _save(fig, output_dir, f"yy_population_{split}_bar_line")


def reproduce(data_dir: Path, provenance_path: Path, output_dir: Path) -> dict:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    inputs = {
        "e1_theory_path_errors.csv": data_dir / "e1_theory_path_errors.csv",
        "e1_theory_path_slopes.csv": data_dir / "e1_theory_path_slopes.csv",
        "e1_theory_path_numerical_convergence.csv": data_dir / "e1_theory_path_numerical_convergence.csv",
        "e1_theory_path_audit.json": data_dir / "e1_theory_path_audit.json",
        "e1_theory_path_gpu_gate.json": data_dir / "e1_theory_path_gpu_gate.json",
        "yy_theory_path_errors.csv": data_dir / "yy_theory_path_errors.csv",
        "yy_theory_path_slopes.csv": data_dir / "yy_theory_path_slopes.csv",
        "yy_theory_path_numerical_convergence.csv": data_dir / "yy_theory_path_numerical_convergence.csv",
        "yy_theory_path_audit.json": data_dir / "yy_theory_path_audit.json",
        "yy_theory_path_gpu_gate.json": data_dir / "yy_theory_path_gpu_gate.json",
    }
    actual_hashes = {name: sha256(path) for name, path in inputs.items()}
    if actual_hashes != provenance["data_sha256"]:
        raise ValueError("One or more E1 data checksums do not match provenance.json")

    theory_rows = load_theory_rows(inputs["e1_theory_path_errors.csv"])
    slope_rows = load_slope_rows(inputs["e1_theory_path_slopes.csv"])
    yy_rows = load_yy_theory_rows(inputs["yy_theory_path_errors.csv"])
    yy_slope_rows = load_slope_rows(inputs["yy_theory_path_slopes.csv"])
    validate_theory_rows(theory_rows)
    validate_slope_rows(slope_rows)
    validate_yy_theory_rows(yy_rows)
    validate_yy_slope_rows(yy_slope_rows)
    validate_registered_fits(theory_rows, slope_rows)
    validate_registered_fits(yy_rows, yy_slope_rows)
    audit, gate = validate_numerical_evidence(
        inputs["e1_theory_path_numerical_convergence.csv"],
        inputs["e1_theory_path_audit.json"],
        inputs["e1_theory_path_gpu_gate.json"],
    )
    yy_audit, yy_gate = validate_yy_numerical_evidence(
        inputs["yy_theory_path_numerical_convergence.csv"],
        inputs["yy_theory_path_audit.json"],
        inputs["yy_theory_path_gpu_gate.json"],
    )

    outputs: list[Path] = []
    for method in METHODS:
        outputs.extend(render_scaled_errors(theory_rows, method, output_dir))
        outputs.extend(render_slope_stability(slope_rows, method, output_dir))
        for metric in ("average", "infinity"):
            outputs.extend(render_actual_errors(theory_rows, method, metric, MAIN_GEOMETRIES, "main", output_dir))
    outputs.extend(render_scaled_errors(yy_rows, "YY", output_dir))
    outputs.extend(render_slope_stability(yy_slope_rows, "YY", output_dir))
    for metric in ("average", "infinity"):
        outputs.extend(render_actual_errors(yy_rows, "YY", metric, MAIN_GEOMETRIES, "main", output_dir))

    manifest = {
        "schema": "E1_SIMULATION_FIGURE_REPRODUCTION_V5",
        "status": "PASS",
        "design": "sigma decreases on frozen proportional-scale paths for STSP, DR, and YY",
        "data_sha256": actual_hashes,
        "theory_path_rows": len(theory_rows),
        "slope_rows": len(slope_rows),
        "yy_rows": len(yy_rows),
        "yy_slope_rows": len(yy_slope_rows),
        "numerical_maxima": {"STSP_DR": audit["numerical_maxima"], "YY": yy_audit["numerical_maxima"]},
        "gpu_gate_max_fitted_point_difference": gate["max_fitted_point_difference"],
        "yy_gpu_gate_max_coordinate_difference": yy_gate["max_coordinate_difference"],
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
