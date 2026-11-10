"""Build the held-out E5 E_inf/E_avg figure from frozen result packets.

This script is post-processing only.  For each repeat and method it computes
E_inf as the maximum and E_avg as the arithmetic mean of the 256 recorded
query-wise geometric errors.  Across the 20 repeats it plots the median and
the interquartile range.  No experimental cell is rerun or filtered.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Single", "Scalar-A", "Scalar-C", "Directional")
COLORS = {
    "Single": "#3B73AF",
    "Scalar-A": "#D55E00",
    "Scalar-C": "#279E68",
    "Directional": "#A33DA3",
}
MARKERS = {"Single": "o", "Scalar-A": "s", "Scalar-C": "^", "Directional": "D"}


def read_packets(packet_dir: Path) -> list[dict]:
    rows: list[dict] = []
    packet_paths = sorted(packet_dir.glob("e5_directional_n*_r*.json"))
    if len(packet_paths) != 60:
        raise RuntimeError(f"Expected 60 frozen packets, found {len(packet_paths)}")

    for path in packet_paths:
        packet = json.loads(path.read_text(encoding="utf-8"))
        result = packet["result"]
        if result.get("failure_count") != 0:
            raise RuntimeError(f"Packet contains failures: {path.name}")
        sample_size = int(result["sample_size"])
        repeat = int(result["repeat"])
        query_rows = result["query_rows"]
        if len(query_rows) != 256 or any(q.get("status") != "success" for q in query_rows):
            raise RuntimeError(f"Incomplete query grid: {path.name}")

        for method in METHODS:
            values = np.asarray(
                [float(q["metrics"][method]["geometric"]) for q in query_rows],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Non-finite metric: {path.name}, {method}")
            rows.append(
                {
                    "sample_size": sample_size,
                    "repeat": repeat,
                    "method": method,
                    "E_inf": float(np.max(values)),
                    "E_avg": float(np.mean(values)),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for metric in ("E_inf", "E_avg"):
        for sample_size in sorted({r["sample_size"] for r in rows}):
            for method in METHODS:
                values = np.asarray(
                    [r[metric] for r in rows if r["sample_size"] == sample_size and r["method"] == method],
                    dtype=float,
                )
                if values.size != 20:
                    raise RuntimeError(f"Expected 20 repeats for {sample_size}, {method}, {metric}")
                out.append(
                    {
                        "metric": metric,
                        "sample_size": sample_size,
                        "method": method,
                        "count": int(values.size),
                        "median": float(np.median(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict], output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75), constrained_layout=True)
    specs = (
        ("E_inf", r"Maximum error $E_{\infty}$"),
        ("E_avg", r"Average error $E_{\mathrm{avg}}$"),
    )
    sample_sizes = sorted({r["sample_size"] for r in summary})
    x = np.arange(len(sample_sizes), dtype=float)

    for ax, (metric, ylabel) in zip(axes, specs):
        for method in METHODS:
            data = [
                next(
                    r
                    for r in summary
                    if r["metric"] == metric
                    and r["sample_size"] == n
                    and r["method"] == method
                )
                for n in sample_sizes
            ]
            median = np.asarray([r["median"] for r in data])
            q1 = np.asarray([r["q1"] for r in data])
            q3 = np.asarray([r["q3"] for r in data])
            ax.fill_between(x, q1, q3, color=COLORS[method], alpha=0.14, linewidth=0)
            ax.plot(
                x,
                median,
                linestyle="-",
                linewidth=2.0,
                marker=MARKERS[method],
                markersize=5.5,
                color=COLORS[method],
                label=method,
            )

        ax.set_yscale("log")
        ax.set_xticks(x, [f"{n // 1000}k" for n in sample_sizes])
        ax.set_xlabel(r"Sample size $n$")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.055))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    repeat_rows = read_packets(result_root / "packets")
    summary_rows = summarize(repeat_rows)
    write_csv(result_root / "audit" / "repeat_values_dual_metrics.csv", repeat_rows)
    write_csv(result_root / "audit" / "summary_dual_metrics.csv", summary_rows)
    plot(summary_rows, result_root / "figures" / "e5_directional_ordering_heldout_dual")


if __name__ == "__main__":
    main()
