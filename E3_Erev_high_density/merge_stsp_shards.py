from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
STSP = ROOT.parent / "E3" / "methods" / "stsp"
sys.path.insert(0, str(STSP / "src"))

from e3.geometry import BY_KEY  # noqa: E402
from e3.metrics import method_metrics  # noqa: E402
from e3.packets import atomic_write, read_packet  # noqa: E402
from e3.runner import _single_cache, validate_parent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--stage", choices=(1, 2), type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    specification = design["geometries"][args.geometry]
    targets = [int(value) for value in specification["query_count_stages"]]
    target_count = targets[args.stage - 1]
    base_count = 0 if args.stage == 1 else targets[0]
    total = target_count - base_count
    mode = "full_parent" if args.stage == 1 else "benchmark"
    geometry = BY_KEY[args.geometry]
    packets = []
    for shard_index in range(args.shard_count):
        path = (
            args.result_root / f"stage{args.stage}" / "stsp_shards" /
            f"s{shard_index:02d}of{args.shard_count:02d}" / "audits" /
            f"e3_{args.geometry}_r00.{mode}.json.gz"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        packets.append(read_packet(path))
    first = packets[0]
    code_hash = first["code_hash"]
    observed_hash = first["data_hashes"]["observed"]
    cell_count = len(first["bandwidths"])
    merged_queries = np.empty((total, geometry.D), dtype=float)
    written = np.zeros(total, dtype=bool)
    merged_cells = []
    for cell in first["bandwidths"]:
        template = dict(cell)
        template["single"] = dict(cell["single"])
        template["multi"] = dict(cell["multi"])
        template["single"]["queries"] = [None] * total
        template["multi"]["queries"] = [None] * total
        template["single"]["pipeline_id"] = "merged-single"
        template["multi"]["pipeline_id"] = "merged-multi"
        merged_cells.append(template)

    provenance = []
    for shard_index, packet in enumerate(packets):
        indices = np.arange(total, dtype=int)[shard_index::args.shard_count]
        if packet["geometry"] != args.geometry or packet["repeat"] != 0:
            raise ValueError("shard identity mismatch")
        if packet["code_hash"] != code_hash:
            raise ValueError("scientific code hash mismatch")
        if packet["data_hashes"]["observed"] != observed_hash:
            raise ValueError("observation hash mismatch")
        if len(packet["query_coordinates"]) != len(indices):
            raise ValueError("shard query count mismatch")
        merged_queries[indices] = np.asarray(packet["query_coordinates"], dtype=float)
        written[indices] = True
        provenance.append({
            "shard_index": shard_index,
            "packet_total_seconds": packet["timing"]["total_before_serialization"],
        })
        if len(packet["bandwidths"]) != cell_count:
            raise ValueError("shard cell count mismatch")
        for cell_index, cell in enumerate(packet["bandwidths"]):
            target = merged_cells[cell_index]
            if not np.isclose(float(cell["h2"]), float(target["h2"]), rtol=0.0, atol=1e-14):
                raise ValueError("bandwidth mismatch")
            if cell["weights"] != target["weights"]:
                raise ValueError("weight mismatch")
            for local_index, global_index in enumerate(indices):
                target["single"]["queries"][global_index] = cell["single"]["queries"][local_index]
                target["multi"]["queries"][global_index] = cell["multi"]["queries"][local_index]
    if not written.all():
        raise RuntimeError("incomplete query merge")

    truth_count = int(design["reference_count"])
    for cell_index, cell in enumerate(merged_cells):
        single_rows = cell["single"]["queries"]
        multi_rows = cell["multi"]["queries"]
        if any(row is None for row in single_rows + multi_rows):
            raise RuntimeError("incomplete fitted-row merge")
        cell["Single_h2_metrics"] = method_metrics(geometry, single_rows, "Single_h2", truth_count)
        cell["Single_h1_metrics"] = method_metrics(geometry, single_rows, "Single_h1", truth_count)
        cell["ENF_Multi_metrics"] = method_metrics(geometry, multi_rows, "ENF_Multi", truth_count)
        cell["query_success_counts"] = {
            name: cell[name]["query_success_count"]
            for name in ("Single_h2_metrics", "Single_h1_metrics", "ENF_Multi_metrics")
        }
        cell["query_failure_counts"] = {
            name: cell[name]["query_failure_count"]
            for name in ("Single_h2_metrics", "Single_h1_metrics", "ENF_Multi_metrics")
        }
        successes = list(cell["query_success_counts"].values())
        cell["status"] = (
            "success" if all(value == total for value in successes)
            else "failed" if all(value == 0 for value in successes)
            else "partial"
        )
        for pipeline in ("single", "multi"):
            cell[pipeline]["timing"] = {
                key: float(sum(packet["bandwidths"][cell_index][pipeline]["timing"][key] for packet in packets))
                for key in cell[pipeline]["timing"]
            }
        cell["runtime_single"] = cell["single"]["timing"]["total"]
        cell["runtime_multi"] = cell["multi"]["timing"]["total"]
        cell["runtime_ratio"] = (
            cell["runtime_multi"] / cell["runtime_single"]
            if cell["runtime_single"] > 0 else float("nan")
        )

    merged = dict(first)
    merged["N_q"] = total
    merged["query_coordinates"] = merged_queries
    merged["bandwidths"] = merged_cells
    merged["single_cache"] = _single_cache(merged_cells)
    merged["timing"] = {
        "sample_generation": float(sum(packet["timing"]["sample_generation"] for packet in packets)),
        "total_before_serialization": float(sum(packet["timing"]["total_before_serialization"] for packet in packets)),
    }
    merged["shard_provenance"] = {
        "schema_version": "E3_EREV_STSP_SHARD_MERGE_V1",
        "stage": args.stage,
        "base_query_count": base_count,
        "target_query_count": target_count,
        "evaluated_query_count": total,
        "shard_count": args.shard_count,
        "partition": "zero-based strided query indices",
        "records": provenance,
    }
    validate_parent(merged, args.geometry, 0)
    destination = (
        args.result_root / f"stage{args.stage}" / "merged" / "stsp" / "packets" /
        f"e3_stsp_{args.geometry}_r00.json.gz"
    )
    if destination.exists():
        raise FileExistsError(destination)
    atomic_write(destination, merged)
    print(destination)


if __name__ == "__main__":
    main()
