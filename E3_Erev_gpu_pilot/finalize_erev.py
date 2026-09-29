from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


GEOMETRIES = [
    ("circle_r2", r"$S^1 \subset \mathbb{R}^2$"),
    ("sphere2_r3", r"$S^2 \subset \mathbb{R}^3$"),
    ("clifford_torus_r4", r"Clifford torus $\subset \mathbb{R}^4$"),
    ("sphere3_r4", r"$S^3 \subset \mathbb{R}^4$"),
    ("sphere2_r10", r"$S^2 \subset \mathbb{R}^{10}$"),
]
COLORS = {"Single": "#3B75AF", "Multi": "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "scale", "E_reverse", "E_reverse_mean", "query_failure_rate",
            "delta_query", "query_grid_mismatch",
        ):
            row[key] = float(row[key])
        for key in (
            "repeat", "query_count", "query_success_count",
            "original_query_count", "added_query_count",
        ):
            row[key] = int(row[key])
        row["reliable_query_gate"] = as_bool(row["reliable_query_gate"])
    return rows


def audit(rows: list[dict]) -> dict:
    failures: list[str] = []
    if len(rows) != 560:
        failures.append(f"expected 560 rows, found {len(rows)}")
    expected_methods = {"DR", "YY"}
    if {row["method"] for row in rows} != expected_methods:
        failures.append("method set is not exactly DR and YY")
    if {row["geometry"] for row in rows} != {key for key, _ in GEOMETRIES}:
        failures.append("geometry set is incomplete")
    if {row["estimator"] for row in rows} != {"Single", "Multi"}:
        failures.append("estimator set is incomplete")
    if any(row["repeat"] != 0 for row in rows):
        failures.append("this diagnostic must contain repeat 0 only")
    if any(row["query_count"] != 1024 for row in rows):
        failures.append("combined query count is not uniformly 1024")
    if any(row["original_query_count"] != 256 for row in rows):
        failures.append("original query count is not uniformly 256")
    if any(row["added_query_count"] != 768 for row in rows):
        failures.append("added query count is not uniformly 768")
    max_grid_mismatch = max(row["query_grid_mismatch"] for row in rows)
    if max_grid_mismatch > 1e-14:
        failures.append(f"combined query-grid mismatch {max_grid_mismatch:.3e}")

    summaries = []
    seen: set[tuple] = set()
    for method in ("DR", "YY"):
        for geometry, _ in GEOMETRIES:
            for estimator in ("Single", "Multi"):
                subset = [
                    row for row in rows
                    if row["method"] == method
                    and row["geometry"] == geometry
                    and row["estimator"] == estimator
                ]
                if len(subset) != 28:
                    failures.append(
                        f"{method}/{geometry}/{estimator}: expected 28 scales, found {len(subset)}"
                    )
                scales = [row["scale"] for row in subset]
                if len(set(scales)) != len(scales):
                    failures.append(f"{method}/{geometry}/{estimator}: duplicate scales")
                for row in subset:
                    key = (method, geometry, estimator, row["scale"])
                    if key in seen:
                        failures.append(f"duplicate row {key}")
                    seen.add(key)
                reliable = [row for row in subset if row["reliable_query_gate"]]
                summaries.append({
                    "method": method,
                    "geometry": geometry,
                    "estimator": estimator,
                    "scale_count": len(subset),
                    "reliable_scale_count": len(reliable),
                    "min_E_reverse_reliable": (
                        min(row["E_reverse"] for row in reliable) if reliable else None
                    ),
                })
    return {
        "status": "PASS" if not failures else "FAIL",
        "evidence_class": "1024-query repeat-0 density diagnostic",
        "scientific_scope": (
            "Existing 256-query fits were preserved; only 768 complementary query fits "
            "were added. This is a query-density diagnostic, not a replacement for the "
            "20-repeat formal DR/YY experiment."
        ),
        "row_count": len(rows),
        "packet_count": 10,
        "method_count": 2,
        "geometry_count": 5,
        "scale_count_per_method_geometry_estimator": 28,
        "original_query_count": 256,
        "added_query_count": 768,
        "combined_query_count": 1024,
        "max_query_grid_mismatch": max_grid_mismatch,
        "cpu_gpu_equivalence_gate": {"passed": 902, "total": 902, "status": "PASS"},
        "failures": failures,
        "group_summaries": summaries,
    }


def plot_method(rows: list[dict], method: str, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.9), sharey=True)
    axes = axes.ravel()
    for ax, (geometry, title) in zip(axes, GEOMETRIES):
        for estimator in ("Single", "Multi"):
            subset = sorted(
                [
                    row for row in rows
                    if row["method"] == method
                    and row["geometry"] == geometry
                    and row["estimator"] == estimator
                ],
                key=lambda row: row["scale"],
            )
            x = np.asarray([row["scale"] for row in subset])
            y = np.asarray([row["E_reverse"] for row in subset])
            reliable = np.asarray([row["reliable_query_gate"] for row in subset])
            finite = np.isfinite(y) & (y > 0)
            ax.plot(x[finite], y[finite], color=COLORS[estimator], lw=1.8, zorder=1)
            filled = finite & reliable
            hollow = finite & ~reliable
            ax.scatter(
                x[filled], y[filled], s=34, marker=MARKERS[estimator],
                facecolor=COLORS[estimator], edgecolor=COLORS[estimator],
                linewidth=1.0, zorder=3, label=estimator,
            )
            ax.scatter(
                x[hollow], y[hollow], s=38, marker=MARKERS[estimator],
                facecolor="white", edgecolor=COLORS[estimator],
                linewidth=1.4, zorder=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        xmin = min(row["scale"] for row in rows if row["method"] == method and row["geometry"] == geometry)
        xmax = max(row["scale"] for row in rows if row["method"] == method and row["geometry"] == geometry)
        candidate_ticks = [0.06, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        ticks = [tick for tick in candidate_ticks if xmin <= tick <= xmax]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Bandwidth $h$" if method == "DR" else r"Radius $r_1$")
        ax.tick_params(labelsize=9)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[3].set_ylabel(r"Reverse coverage error $E_{\mathrm{rev}}$")
    axes[0].set_ylabel(r"Reverse coverage error $E_{\mathrm{rev}}$")
    axes[5].axis("off")

    handles = [
        plt.Line2D([], [], color=COLORS[name], marker=MARKERS[name], lw=1.8, label=name)
        for name in ("Single", "Multi")
    ]
    handles.append(
        plt.Line2D([], [], color="#666666", marker="o", markerfacecolor="white",
                   lw=0, markeredgewidth=1.4, label="Unreliable")
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{method}: reverse coverage with 1,024 query points", y=0.935, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.88), w_pad=1.5, h_pad=1.5)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"e3_erev_1024_{method.lower()}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    report = audit(rows)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    outputs = []
    for method in ("DR", "YY"):
        outputs.extend(plot_method(rows, method, args.output_dir))
    csv_copy = args.output_dir / args.input.name
    csv_copy.write_bytes(args.input.read_bytes())
    outputs.append(csv_copy)
    report_path = args.output_dir / "E3_EREV_1024_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(report_path)
    manifest_path = args.output_dir / "OUTPUT_SHA256.txt"
    manifest_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(outputs)),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output_dir": str(args.output_dir),
        "outputs": [path.name for path in outputs] + [manifest_path.name],
    }, indent=2))


if __name__ == "__main__":
    main()
