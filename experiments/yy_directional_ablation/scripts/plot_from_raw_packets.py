#!/usr/bin/env python3
"""Diagnostic renderer for newly generated raw packets.

The default paper entry point is ``publish_paper_figures.py``. This script is
kept only for an expensive from-scratch numerical rerun and does not replace
the checked-in paper evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e56v2.e5_ablation import METHODS


COLORS_FINITE = {
    "Single": "#3B73AF", "Scalar-A": "#D55E00",
    "Scalar-C": "#279E68", "Directional": "#A33DA3",
}
COLORS_POPULATION = {
    "Single": "#0072B2", "Scalar-A": "#E69F00",
    "Scalar-C": "#009E73", "Directional": "#CC79A7",
}
MARKERS = {"Single": "o", "Scalar-A": "s", "Scalar-C": "^", "Directional": "D"}


def load_packets(path: Path, pattern: str) -> list[dict]:
    return [
        json.loads(item.read_text(encoding="utf-8"))["result"]
        for item in sorted(path.glob(pattern))
    ]


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(y) & (y > 0)
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def population_figure(results: Path, output: Path, config: dict) -> dict:
    rows = sorted(
        load_packets(results / "population" / "packets", "population_sigma_*.json"),
        key=lambda row: row["sigma"],
    )
    if len(rows) != len(config["sigma_grid"]):
        raise RuntimeError("population packet set is incomplete")
    metrics = (
        ("geometric_max", "Total maximum error"),
        ("axial_normal_max", "Axial normal error"),
        ("complementary_normal_max", "Complementary normal error"),
    )
    slope_window = set(config["slope_window"])
    slopes = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.85), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        for index, method in enumerate(METHODS):
            x = np.asarray([row["sigma"] for row in rows])
            y = np.asarray([row["summaries"][method][metric] for row in rows])
            select = np.asarray([value in slope_window for value in x])
            fitted = slope(x[select], y[select])
            slopes.setdefault(method, {})[metric] = fitted
            axis.loglog(
                x, y, color=COLORS_POPULATION[method], marker=MARKERS[method],
                lw=1.5, ms=4, label=method,
            )
            axis.text(
                0.055, 0.54 - 0.06 * index,
                rf"$p_{{\mathrm{{{method}}}}}={fitted:.2f}$",
                transform=axis.transAxes, ha="left", va="top",
                fontsize=7.5, color="#333333",
            )
        axis.set_title(title, fontsize=10, color="#1A1A1A", fontweight="normal")
        axis.set_xlabel(r"Noise level $\sigma$", fontsize=9.5, color="#1A1A1A")
        axis.tick_params(axis="both", which="both", labelsize=8.5, colors="#1A1A1A")
        axis.grid(False)
    axes[0].set_ylabel("Population maximum error", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               frameon=False, fontsize=8, ncol=4)
    base = output / "figure_e5_population_directional_ablation"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return slopes


def finite_rows(results: Path, config: dict) -> list[dict]:
    packets = load_packets(results / "finite_sample" / "packets", "finite_n*_r*.json")
    expected = len(config["sample_sizes"]) * int(config["repeats"])
    if len(packets) != expected:
        raise RuntimeError(f"expected {expected} finite packets, found {len(packets)}")
    output = []
    for result in packets:
        if result["failure_count"] != 0 or len(result["query_rows"]) != config["query_count"]:
            raise RuntimeError("finite source figure requires all declared queries")
        for method in METHODS:
            values = np.asarray([
                query["metrics"][method]["geometric"] for query in result["query_rows"]
            ])
            output.append({
                "sample_size": int(result["sample_size"]),
                "repeat": int(result["repeat"]), "method": method,
                "E_inf": float(np.max(values)), "E_avg": float(np.mean(values)),
            })
    return output


def finite_summary(rows: list[dict], config: dict) -> list[dict]:
    output = []
    for metric in ("E_inf", "E_avg"):
        for sample_size in config["sample_sizes"]:
            for method in METHODS:
                values = np.asarray([
                    row[metric] for row in rows
                    if row["sample_size"] == sample_size and row["method"] == method
                ])
                if values.size != config["repeats"]:
                    raise RuntimeError("finite repeat grid is incomplete")
                output.append({
                    "metric": metric, "sample_size": sample_size, "method": method,
                    "count": int(values.size), "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                })
    return output


def finite_figure(rows: list[dict], output: Path, config: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 12, "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.linewidth": 0.9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75), constrained_layout=True)
    specs = (("E_inf", r"Maximum error $E_{\infty}$"), ("E_avg", r"Average error $E_{\mathrm{avg}}$"))
    sample_sizes = config["sample_sizes"]
    x = np.arange(len(sample_sizes), dtype=float)
    for axis, (metric, ylabel) in zip(axes, specs):
        for method in METHODS:
            data = [
                next(row for row in rows if row["metric"] == metric
                     and row["sample_size"] == sample_size and row["method"] == method)
                for sample_size in sample_sizes
            ]
            median = np.asarray([row["median"] for row in data])
            q1 = np.asarray([row["q1"] for row in data])
            q3 = np.asarray([row["q3"] for row in data])
            axis.fill_between(x, q1, q3, color=COLORS_FINITE[method], alpha=0.14, linewidth=0)
            axis.plot(x, median, "-", lw=2.0, marker=MARKERS[method], ms=5.5,
                      color=COLORS_FINITE[method], label=method)
        axis.set_yscale("log")
        axis.set_xticks(x, [f"{sample_size // 1000}k" for sample_size in sample_sizes])
        axis.set_xlabel(r"Sample size $n$")
        axis.set_ylabel(ylabel)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=4, width=0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.055))
    base = output / "figure_e5_finite_sample_dual_metrics"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    results = args.results.resolve()
    audit = json.loads((results / "AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("figures require a passing packet audit")
    population_config = json.loads((ROOT / "config" / "population.json").read_text(encoding="utf-8"))
    finite_config = json.loads((ROOT / "config" / "finite_sample.json").read_text(encoding="utf-8"))
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    slopes = population_figure(results, output, population_config)
    repeated = finite_rows(results, finite_config)
    summarized = finite_summary(repeated, finite_config)
    write_csv(output / "finite_repeat_values.csv", repeated)
    write_csv(output / "finite_summary.csv", summarized)
    finite_figure(summarized, output, finite_config)
    (output / "population_slopes.json").write_text(
        json.dumps(slopes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote exactly two figures to {output}")


if __name__ == "__main__":
    main()
