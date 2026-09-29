from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
E7_ROOT = ROOT.parent / "E7"
sys.path.insert(0, str(E7_ROOT / "src"))

from e7.geometry import tensor_query_set  # noqa: E402
from e7.runner import read_packet, run_parent  # noqa: E402

COLORS = {512: "#7A7A7A", 1024: "#3B6EA8", 2048: "#D55E00"}
MARKERS = {"Single": "o", "Multi": "s"}


def load_config() -> dict:
    return json.loads((ROOT / "density_config.json").read_text(encoding="utf-8"))


def select_rows(packet: dict, h2_grid: list[float], query_count: int) -> list[dict]:
    requested = {float(h) for h in h2_grid}
    rows = []
    for row in packet["rows"]:
        if float(row["h2"]) not in requested:
            continue
        rows.append(
            {
                "N_query": query_count,
                "method": row["method"],
                "h2": float(row["h2"]),
                "E_reverse": float(row["E_reverse"]),
                "E_reverse_mean": float(row["E_reverse_mean"]),
                "delta_query": float(row["delta_query"]),
                "query_failure_rate": float(row["query_failure_rate"]),
                "n_query_valid": int(row["n_query_valid"]),
            }
        )
    return rows


def run() -> list[dict]:
    cfg = load_config()
    h2_grid = [float(x) for x in cfg["h2_grid"]]
    baseline = read_packet((ROOT / cfg["baseline_packet"]).resolve())
    rows = select_rows(baseline, h2_grid, int(cfg["baseline_query_count"]))

    for count_text, shape in cfg["query_shapes"].items():
        count = int(count_text)
        queries = tensor_query_set(int(shape[0]), int(shape[1]))[1]
        if len(queries) != count:
            raise RuntimeError(f"query shape {shape} produced {len(queries)}, expected {count}")
        packet_path = run_parent(
            int(cfg["repeat"]),
            ROOT / "results",
            bandwidths=h2_grid,
            queries=queries,
            mode=f"density_curve_{count}",
            resume=True,
        )
        rows.extend(select_rows(read_packet(packet_path), h2_grid, count))

    rows.sort(key=lambda r: (r["N_query"], r["method"], r["h2"]))
    out = ROOT / "e7_query_density_curve.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, method in zip(axes, ("Single", "Multi")):
        for count in (512, 1024, 2048):
            current = sorted(
                (r for r in rows if r["method"] == method and r["N_query"] == count),
                key=lambda r: r["h2"],
            )
            x = np.asarray([r["h2"] for r in current])
            y = np.asarray([r["E_reverse"] for r in current])
            delta = float(current[0]["delta_query"])
            ax.plot(x, y, color=COLORS[count], marker=MARKERS[method], ms=4,
                    lw=1.4, label=rf"$N_Q={count}$")
            ax.axhline(delta, color=COLORS[count], lw=0.9, alpha=0.55)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Bandwidth $h_2$")
        ax.set_title(method)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Reverse coverage error $E_{\rm rev}$")
    density_handles = [Line2D([0], [0], color=COLORS[n], lw=1.5, label=rf"$N_Q={n}$")
                       for n in (512, 1024, 2048)]
    floor_handle = Line2D([0], [0], color="0.4", lw=0.9, label=r"ideal floor $\delta_Q$")
    fig.legend(handles=density_handles + [floor_handle], loc="upper center", ncol=4,
               frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.09, right=0.98, wspace=0.15)
    fig.savefig(ROOT / "e7_query_density_Erev.png", dpi=400, bbox_inches="tight")
    fig.savefig(ROOT / "e7_query_density_Erev.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    result_rows = run()
    plot(result_rows)
    print(ROOT / "e7_query_density_Erev.png")
