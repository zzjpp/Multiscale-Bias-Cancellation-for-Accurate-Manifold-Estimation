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


METHODS = ("STSP", "DR", "Yao--Yau")
GEOMETRIES = (
    "circle_r2",
    "sphere2_r3",
    "clifford_torus_r4",
    "sphere3_r4",
    "sphere2_r10",
)
MAIN = ("circle_r2", "clifford_torus_r4", "sphere2_r10")
APPENDIX = ("sphere2_r3", "sphere3_r4")
TITLES = {
    "circle_r2": r"$S^1\subset\mathbb{R}^2$",
    "sphere2_r3": r"$S^2\subset\mathbb{R}^3$",
    "clifford_torus_r4": r"Clifford torus $\subset\mathbb{R}^4$",
    "sphere3_r4": r"$S^3\subset\mathbb{R}^4$",
    "sphere2_r10": r"$S^2\subset\mathbb{R}^{10}$",
}
GRIDS = {
    "STSP": (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.125, 1.25),
    "DR": (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45),
    "Yao--Yau": (0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65),
}
COLORS = {"single": "#0072B2", "multi": "#D55E00"}
OUTPUT_STEMS = {
    ("STSP", "main"): "e1_stsp_population_main",
    ("STSP", "appendix"): "e1_stsp_population_appendix",
    ("DR", "main"): "e1_dr_population_main",
    ("DR", "appendix"): "e1_dr_population_appendix",
    ("Yao--Yau", "main"): "yy_population_main_bar_line",
    ("Yao--Yau", "appendix"): "yy_population_appendix_bar_line",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "scale": float(row["scale"]),
                    "sigma": float(row["sigma"]),
                    "single_error": float(row["single_error"]),
                    "multi_error": float(row["multi_error"]),
                    "single_resolved": _parse_bool(row["single_resolved"]),
                    "multi_resolved": _parse_bool(row["multi_resolved"]),
                }
            )
    return rows


def validate_rows(rows: list[dict]) -> None:
    if len(rows) != 115:
        raise ValueError(f"Expected 115 rows, found {len(rows)}")
    grouped = group_rows(rows)
    for method in METHODS:
        for geometry in GEOMETRIES:
            subset = grouped[(method, geometry)]
            grid = tuple(row["scale"] for row in subset)
            if not np.allclose(grid, GRIDS[method], rtol=0.0, atol=1e-12):
                raise ValueError(f"Scale grid mismatch for {method}/{geometry}: {grid}")
            if any(not row["single_resolved"] or not row["multi_resolved"] for row in subset):
                raise ValueError(f"Unresolved formal cell for {method}/{geometry}")
            if any(row["single_error"] <= 0 or row["multi_error"] <= 0 for row in subset):
                raise ValueError(f"Nonpositive error for {method}/{geometry}")
            expected_parameter = "r1" if method == "Yao--Yau" else "h2"
            if any(row["scale_parameter"] != expected_parameter for row in subset):
                raise ValueError(f"Scale label mismatch for {method}/{geometry}")


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["geometry"])].append(row)
    for subset in grouped.values():
        subset.sort(key=lambda row: row["scale"])
    return dict(grouped)


def slope(rows: list[dict], output: str) -> float:
    x = np.log(np.asarray([row["scale"] for row in rows], dtype=float))
    y = np.log(np.asarray([row[f"{output}_error"] for row in rows], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "E1_simulation", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png,
        dpi=360,
        bbox_inches="tight",
        metadata={"Software": "E1_simulation"},
    )
    plt.close(fig)
    return [pdf, png]


def _style_stsp_dr() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_stsp_dr_panel(axis, rows: list[dict], geometry: str, method: str) -> None:
    x = np.arange(len(rows), dtype=float)
    width = 0.34
    single = np.asarray([row["single_error"] for row in rows], dtype=float)
    multi = np.asarray([row["multi_error"] for row in rows], dtype=float)
    axis.bar(
        x - width / 2,
        single,
        width,
        color=COLORS["single"],
        alpha=0.78,
        edgecolor=COLORS["single"],
        linewidth=0.55,
        label="Single",
        zorder=2,
    )
    axis.bar(
        x + width / 2,
        multi,
        width,
        color=COLORS["multi"],
        alpha=0.78,
        edgecolor=COLORS["multi"],
        linewidth=0.55,
        label="Multi",
        zorder=2,
    )
    axis.plot(x - width / 2, single, color=COLORS["single"], marker="o", markersize=3.1, linewidth=1.15, zorder=3)
    axis.plot(x + width / 2, multi, color=COLORS["multi"], marker="s", markersize=3.0, linewidth=1.15, zorder=3)
    axis.set_yscale("log")
    axis.set_ylim((2.5e-5, 2.5e-1) if method == "STSP" else (5e-5, 1.0))
    axis.set_title(TITLES[geometry], pad=5)
    labels = [f"{value:g}".lstrip("0") for value in GRIDS[method]]
    axis.set_xticks(x, labels)
    axis.tick_params(axis="x", rotation=32, pad=1)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.035,
        0.955,
        f"slopes: {slope(rows, 'single'):.2f} / {slope(rows, 'multi'):.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.78", "linewidth": 0.45, "alpha": 0.92},
    )


def render_stsp_dr(grouped: dict, method: str, geometries: tuple[str, ...], split: str, output_dir: Path) -> list[Path]:
    _style_stsp_dr()
    width = 8.0 if split == "main" else 5.45
    fig, axes = plt.subplots(1, len(geometries), figsize=(width, 2.55), squeeze=False, sharey=True)
    axes = axes[0]
    for axis, geometry in zip(axes, geometries):
        _draw_stsp_dr_panel(axis, grouped[(method, geometry)], geometry, method)
    axes[0].set_ylabel(r"Population maximum error $E_{\infty}$")
    fig.supxlabel(r"$h_2$", y=0.015, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015), handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.095 if split == "main" else 0.13, right=0.995, bottom=0.25, top=0.82, wspace=0.30)
    return _save(fig, output_dir, OUTPUT_STEMS[(method, split)])


def _style_yy() -> None:
    _style_stsp_dr()


def render_yy(grouped: dict, geometries: tuple[str, ...], split: str, output_dir: Path) -> list[Path]:
    _style_yy()
    width = 8.0 if split == "main" else 5.45
    fig, axes = plt.subplots(1, len(geometries), figsize=(width, 2.55), sharey=True)
    axes = np.atleast_1d(axes)
    all_rows = [row for geometry in geometries for row in grouped[("Yao--Yau", geometry)]]
    all_values = [row[key] for row in all_rows for key in ("single_error", "multi_error")]
    floor = 10.0 ** (np.floor(np.log10(min(all_values))) - 0.35)
    ceiling = 10.0 ** (np.ceil(np.log10(max(all_values))) + 0.18)
    x = np.arange(len(GRIDS["Yao--Yau"]))
    bar_width = 0.34
    for axis, geometry in zip(axes, geometries):
        rows = grouped[("Yao--Yau", geometry)]
        single = np.asarray([row["single_error"] for row in rows], dtype=float)
        multi = np.asarray([row["multi_error"] for row in rows], dtype=float)
        single_x = x - bar_width / 2
        multi_x = x + bar_width / 2
        axis.bar(single_x, single, bar_width, color=COLORS["single"], alpha=.78,
                 edgecolor=COLORS["single"], linewidth=.55, label="Single", zorder=2)
        axis.bar(multi_x, multi, bar_width, color=COLORS["multi"], alpha=.78,
                 edgecolor=COLORS["multi"], linewidth=.55, label="Multi", zorder=2)
        axis.plot(single_x, single, color=COLORS["single"], marker="o",
                  markersize=3.1, linewidth=1.15, zorder=3)
        axis.plot(multi_x, multi, color=COLORS["multi"], marker="s",
                  markersize=3.0, linewidth=1.15, zorder=3)
        axis.set_yscale("log")
        axis.set_ylim(floor, ceiling)
        axis.set_title(TITLES[geometry], pad=5.0)
        axis.text(
            0.035,
            0.955,
            f"slopes: {slope(rows, 'single'):.2f} / {slope(rows, 'multi'):.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "0.78", "linewidth": 0.45, "alpha": 0.92},
        )
        axis.set_xticks(x, [f"{value:g}".lstrip("0") for value in GRIDS["Yao--Yau"]])
        axis.tick_params(axis="x", rotation=32, pad=1)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Population maximum error $E_\infty$")
    fig.supxlabel(r"$r_1$", y=0.015, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.015),
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.095 if split == "main" else 0.13,
                        right=0.995, bottom=0.25, top=0.82, wspace=0.30)
    return _save(fig, output_dir, OUTPUT_STEMS[("Yao--Yau", split)])


def reproduce(data_path: Path, provenance_path: Path, output_dir: Path) -> dict:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    actual_data_hash = sha256(data_path)
    if actual_data_hash != provenance["data_sha256"]:
        raise ValueError("Data checksum does not match data/provenance.json")
    rows = load_rows(data_path)
    validate_rows(rows)
    grouped = group_rows(rows)
    outputs: list[Path] = []
    for method in ("STSP", "DR"):
        outputs.extend(render_stsp_dr(grouped, method, MAIN, "main", output_dir))
        outputs.extend(render_stsp_dr(grouped, method, APPENDIX, "appendix", output_dir))
    outputs.extend(render_yy(grouped, MAIN, "main", output_dir))
    outputs.extend(render_yy(grouped, APPENDIX, "appendix", output_dir))
    slopes = [
        {
            "method": method,
            "geometry": geometry,
            "single": slope(grouped[(method, geometry)], "single"),
            "multi": slope(grouped[(method, geometry)], "multi"),
        }
        for method in METHODS
        for geometry in GEOMETRIES
    ]
    report = {
        "schema": "E1_SIMULATION_FIGURE_REPRODUCTION_V1",
        "status": "PASS",
        "data_sha256": actual_data_hash,
        "rows": len(rows),
        "slopes": slopes,
        "output_sha256": {path.name: sha256(path) for path in outputs},
    }
    (output_dir / "FIGURE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

