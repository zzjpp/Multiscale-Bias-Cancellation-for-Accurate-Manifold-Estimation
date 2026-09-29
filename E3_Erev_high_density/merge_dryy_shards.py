from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DRYY = ROOT.parent / "E3" / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))

from dryy.geometry import geometries_from_config  # noqa: E402
from dryy.config import load_config  # noqa: E402
from dryy.io import read_packet, write_packet  # noqa: E402
from dryy.metrics import summarize_points  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("DR", "YY"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--stage", choices=(1, 2), type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    design_path = ROOT / "config.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    specification = design["geometries"][args.geometry]
    targets = [int(value) for value in specification["query_count_stages"]]
    target_count = targets[args.stage - 1]
    base_count = 1024 if args.stage == 1 else targets[0]
    total = target_count - base_count
    base_config, _ = load_config("e3")
    geometry = next(g for g in geometries_from_config(base_config) if g.key == args.geometry)

    packets = []
    for shard_index in range(args.shard_count):
        path = (
            args.result_root / f"stage{args.stage}" / "shards" /
            f"s{shard_index:02d}of{args.shard_count:02d}" / "packets" /
            f"e3_{args.method.lower()}_{args.geometry}_r00_"
            f"s{shard_index:02d}of{args.shard_count:02d}.json.gz"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        packets.append(read_packet(path))

    first = packets[0]
    observed_hash = first["seed"]["observed_sha256"]
    code_hash = first["code_hash"]
    gpu_code_hash = first["gpu_backend_code_hash"]
    cell_count = len(first["cells"])
    merged_queries = np.empty((total, int(first["D"])), dtype=float)
    query_written = np.zeros(total, dtype=bool)
    merged_cells = []
    for cell_index in range(cell_count):
        template = dict(first["cells"][cell_index])
        template["single_rows"] = [None] * total
        template["multi_rows"] = [None] * total
        if "common_start_digests" in template:
            template["common_start_digests"] = [None] * total
        merged_cells.append(template)

    shard_hashes = []
    for shard_index, packet in enumerate(packets):
        expected_indices = np.arange(total, dtype=int)[shard_index::args.shard_count]
        if packet["method"] != args.method or packet["geometry"] != args.geometry:
            raise ValueError("shard identity mismatch")
        if packet["seed"]["observed_sha256"] != observed_hash:
            raise ValueError("observation hash mismatch across shards")
        if packet["code_hash"] != code_hash or packet["gpu_backend_code_hash"] != gpu_code_hash:
            raise ValueError("scientific code hash mismatch across shards")
        if len(packet["query_coordinates"]) != len(expected_indices):
            raise ValueError("shard query count mismatch")
        if len(packet["cells"]) != cell_count:
            raise ValueError("shard cell count mismatch")
        merged_queries[expected_indices] = np.asarray(packet["query_coordinates"], dtype=float)
        query_written[expected_indices] = True
        shard_hashes.append({
            "shard_index": shard_index,
            "config_hash": packet["config_hash"],
            "runtime_seconds": packet["runtime_seconds"],
        })
        for cell_index, cell in enumerate(packet["cells"]):
            template = merged_cells[cell_index]
            scale_key = "h2" if args.method == "DR" else "r1"
            if not np.isclose(float(cell[scale_key]), float(template[scale_key]), rtol=0.0, atol=1e-14):
                raise ValueError("cell scale mismatch across shards")
            if cell["weights"] != template["weights"]:
                raise ValueError("cell weights mismatch across shards")
            for local_index, global_index in enumerate(expected_indices):
                template["single_rows"][global_index] = cell["single_rows"][local_index]
                template["multi_rows"][global_index] = cell["multi_rows"][local_index]
                if "common_start_digests" in template:
                    template["common_start_digests"][global_index] = cell["common_start_digests"][local_index]
    if not query_written.all():
        raise RuntimeError("incomplete query merge")
    for cell in merged_cells:
        if any(row is None for row in cell["single_rows"] + cell["multi_rows"]):
            raise RuntimeError("incomplete fitted-row merge")
        cell["Single"] = summarize_points(geometry, cell["single_rows"], "point")
        cell["Multi"] = summarize_points(geometry, cell["multi_rows"], "point")

    merged = dict(first)
    merged["query_count"] = total
    merged["query_coordinates"] = merged_queries
    merged["cells"] = merged_cells
    merged["config_hash"] = hashlib.sha256(design_path.read_bytes()).hexdigest()
    merged["runtime_seconds"] = float(sum(row["runtime_seconds"] for row in packets))
    merged["shard_provenance"] = {
        "schema_version": "E3_EREV_SHARD_MERGE_V1",
        "stage": args.stage,
        "base_query_count": base_count,
        "target_query_count": target_count,
        "added_query_count": total,
        "shard_count": args.shard_count,
        "partition": "zero-based strided complement indices",
        "records": shard_hashes,
    }
    destination = (
        args.result_root / f"stage{args.stage}" / "merged" / "packets" /
        f"e3_{args.method.lower()}_{args.geometry}_r00.json.gz"
    )
    if destination.exists():
        raise FileExistsError(destination)
    write_packet(destination, merged)
    print(destination)


if __name__ == "__main__":
    main()
