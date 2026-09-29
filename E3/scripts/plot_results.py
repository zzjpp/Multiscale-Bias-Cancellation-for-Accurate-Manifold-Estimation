#!/usr/bin/env python3
"""Build matched E3 figures for STSP, DR, and YY from formal packets."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / "config" / "e3_n12000.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "results" / "figures"
GEOMETRIES = DESIGN["geometries"]
LABELS = {
    "circle_r2": r"$S^1 \subset \mathbb{R}^2$",
    "sphere2_r3": r"$S^2 \subset \mathbb{R}^3$",
    "clifford_torus_r4": r"Clifford torus $\subset \mathbb{R}^4$",
    "sphere3_r4": r"$S^3 \subset \mathbb{R}^4$",
    "sphere2_r10": r"$S^2 \subset \mathbb{R}^{10}$",
}


def summary(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3)}


def stsp_rows() -> list[dict]:
    source = ROOT / "methods" / "stsp"
    sys.path.insert(0, str(source / "src"))
    from e3.config import CONFIG, CONFIG_HASH, code_hash
    from e3.packets import read_packet

    packets = []
    packet_root = ROOT / "results" / "formal" / "stsp" / "packets"
    for path in sorted(packet_root.glob("*.production.json.gz")):
        packets.append(read_packet(path, CONFIG_HASH, code_hash()))
    if len(packets) != 100:
        raise RuntimeError(f"expected 100 STSP packets, found {len(packets)}")
    rows = []
    for geometry in GEOMETRIES:
        selected = sorted((p for p in packets if p["geometry"] == geometry), key=lambda p: p["repeat"])
        for index, h in enumerate(CONFIG["bandwidths_h2"]):
            for estimator, field in (("Single", "Single_h2_metrics"), ("Multi", "ENF_Multi_metrics")):
                metrics = [packet["bandwidths"][index][field] for packet in selected]
                failures = sum(row["query_failure_count"] for row in metrics)
                queries = sum(row["query_count"] for row in metrics)
                complete = sum(bool(row["complete"]) for row in metrics)
                rows.append({
                    "geometry": geometry, "x": float(h), "estimator": estimator,
                    "reliable": failures / queries <= CONFIG["validation"]["query_failure_rate_max"]
                                and complete >= CONFIG["validation"]["complete_repeats_min"],
                    "E_avg": summary([row["E_avg"] for row in metrics]),
                    "E_infty": summary([row["E_infty"] for row in metrics]),
                    "failure_taxonomy": dict(Counter(
                        reason for row in metrics for reason, count in row["failure_taxonomy"].items()
                        for _ in range(count)
                    )),
                })
    return rows


def dryy_rows(method: str) -> list[dict]:
    path = ROOT / "results" / "formal" / "dr_yy" / "e3" / "audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise RuntimeError("DR/YY formal audit is not PASS")
    rows = []
    for row in report["summaries"]:
        if row["method"].lower() != method:
            continue
        x = float(row["scale"])
        if method == "yy":
            x *= DESIGN["sigma"]
        rows.append({**row, "x": x})
    return rows


def intervals(rows: list[dict], geometry: str, metric: str) -> list[tuple[float, float]]:
    table = {
        name: {row["x"]: row for row in rows if row["geometry"] == geometry and row["estimator"] == name}
        for name in ("Single", "Multi")
    }
    grid = sorted(set(table["Single"]) & set(table["Multi"]))
    flags = [
        table["Single"][x]["reliable"] and table["Multi"][x]["reliable"]
        and table["Multi"][x][metric]["median"] < table["Single"][x][metric]["median"]
        for x in grid
    ]
    output, start = [], None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            output.append((grid[start], grid[index - 1]))
            start = None
    return output


def build(method: str, rows: list[dict], metric: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.15), squeeze=False)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    colors = {"Single": "#3B6EA8", "Multi": "#D55E00"}
    markers = {"Single": "o", "Multi": "s"}
    for geometry, position in zip(GEOMETRIES, positions):
        ax = axes[position]
        for left, right in intervals(rows, geometry, metric):
            ax.axvspan(left / 1.01, right * 1.01, color="#DCEFE8", alpha=0.82, linewidth=0, zorder=0)
        for estimator in ("Single", "Multi"):
            current = sorted(
                (row for row in rows if row["geometry"] == geometry and row["estimator"] == estimator),
                key=lambda row: row["x"],
            )
            x = np.asarray([row["x"] for row in current])
            y = np.asarray([row[metric]["median"] for row in current])
            reliable = np.asarray([row["reliable"] for row in current], dtype=bool)
            finite = np.isfinite(y) & (y > 0)
            ax.plot(x[finite], y[finite], color=colors[estimator], marker=markers[estimator],
                    linewidth=1.2, markersize=2.7, label="Single" if estimator == "Single" else "Multi")
            failed = finite & ~reliable
            ax.scatter(x[failed], y[failed], marker=markers[estimator], s=19,
                       facecolors="white", edgecolors=colors[estimator], linewidths=0.9, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.055, 5.35)
        ax.set_xticks([0.06, 0.1, 0.2, 0.5, 1, 2, 5])
        ax.set_xticklabels([".06", ".1", ".2", ".5", "1", "2", "5"])
        ax.minorticks_off()
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(LABELS[geometry], pad=3.5)
    axes[1, 2].axis("off")
    ylabel = r"Average geometric error $E_{\mathrm{avg}}$" if metric == "E_avg" else r"Maximum geometric error $E_{\infty}$"
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axis = r"Bandwidth $h$" if method != "yy" else r"Transverse radius $r_1$"
    fig.supxlabel(axis + r" (fixed $\sigma=0.015$, $n=12000$)", y=0.035, fontsize=8.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    open_marker = mpl.lines.Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
                                  linestyle="None", markersize=4.0, label="unreliable")
    fig.legend(handles + [open_marker], labels + ["unreliable"], loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.088, right=0.995, bottom=0.135, top=0.87, wspace=0.27, hspace=0.48)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = "average_error" if metric == "E_avg" else "max_error"
    fig.savefig(OUTPUT / f"e3_{method}_{tag}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUTPUT / f"e3_{method}_{tag}.png", dpi=400, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6, "axes.titlesize": 8.7,
        "axes.labelsize": 8.0, "xtick.labelsize": 6.7, "ytick.labelsize": 6.9,
        "legend.fontsize": 7.0, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    })
    method_rows = {"stsp": stsp_rows(), "dr": dryy_rows("dr"), "yy": dryy_rows("yy")}
    for method, rows in method_rows.items():
        for metric in ("E_avg", "E_infty"):
            build(method, rows, metric)
    print(f"figures written to {OUTPUT}")


if __name__ == "__main__":
    main()
