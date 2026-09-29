from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
E3 = ROOT.parent / "E3"
DRYY = E3 / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))

from dryy.geometry import geometries_from_config  # noqa: E402
from dryy.config import canonical_json, load_config as load_base_config  # noqa: E402
from dryy.io import read_packet  # noqa: E402


def load_overlay() -> tuple[dict, str, dict]:
    """Load the frozen pilot overlay without importing the CuPy runner."""
    pilot = json.loads((ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    base, _ = load_base_config("e3")
    overlay = json.loads(json.dumps(base))
    overlay["N_q"] = int(pilot["added_query_count"])
    overlay["erev_pilot"] = {
        "schema_version": pilot["schema_version"],
        "reference_count": int(pilot["reference_count"]),
        "original_query_count": int(pilot["original_query_count"]),
        "added_query_count": int(pilot["added_query_count"]),
        "combined_query_count": int(pilot["combined_query_count"]),
        "repeat_indices": list(pilot["repeat_indices"]),
    }
    digest = hashlib.sha256(canonical_json(overlay)).hexdigest()
    return overlay, digest, pilot


def fitted(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [row["point"] for row in rows if row.get("status") == "success"],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=ROOT / "results" / "gpu_packets" / "packets")
    parser.add_argument("--base-packet-root", type=Path, required=True)
    parser.add_argument("--yy-packet-root", type=Path)
    parser.add_argument("--yy-base-packet-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "e3_erev_gpu_pilot.csv")
    args = parser.parse_args()
    config, config_hash, pilot = load_overlay()
    by_key = {geometry.key: geometry for geometry in geometries_from_config(config)}
    output_rows = []
    paths = sorted(args.packet_root.glob("e3_dr_*.json.gz"))
    yy_packet_root = args.yy_packet_root or args.packet_root
    paths += sorted(yy_packet_root.glob("e3_yy_*.json.gz"))
    for path in paths:
        packet = read_packet(path)
        if packet["config_hash"] != config_hash:
            raise ValueError(f"overlay config hash mismatch: {path}")
        base_root = args.yy_base_packet_root if packet["method"] == "YY" and args.yy_base_packet_root else args.base_packet_root
        base_path = base_root / path.name
        if not base_path.exists():
            raise FileNotFoundError(f"missing immutable 256-query packet: {base_path}")
        base_packet = read_packet(base_path)
        if base_packet["method"] != packet["method"] or base_packet["geometry"] != packet["geometry"] or base_packet["repeat"] != packet["repeat"]:
            raise ValueError("base and incremental packet identity mismatch")
        if base_packet.get("code_hash") != packet.get("code_hash"):
            raise ValueError("base and incremental scientific code hashes differ")
        if base_packet["seed"]["observed_sha256"] != packet["seed"]["observed_sha256"]:
            raise ValueError("base and incremental observations differ")
        geometry = by_key[packet["geometry"]]
        reference = geometry.query_grid(int(pilot["reference_count"]), int(config["query_seed"]) + 7919)
        queries = np.vstack((np.asarray(base_packet["query_coordinates"], dtype=float), np.asarray(packet["query_coordinates"], dtype=float)))
        if len(queries) != int(pilot["combined_query_count"]):
            raise ValueError("combined query count mismatch")
        expected_queries = geometry.query_grid(
            int(pilot["combined_query_count"]), int(config["query_seed"])
        )
        query_to_expected = cKDTree(expected_queries).query(queries, k=1, workers=-1)[0]
        expected_to_query = cKDTree(queries).query(expected_queries, k=1, workers=-1)[0]
        query_grid_mismatch = float(max(query_to_expected.max(), expected_to_query.max()))
        if query_grid_mismatch > 1e-14:
            raise ValueError(
                f"combined query set differs from the frozen 1024-point grid: "
                f"{packet['method']} {packet['geometry']} mismatch={query_grid_mismatch:.3e}"
            )
        delta = float(cKDTree(queries).query(reference, k=1, workers=-1)[0].max())
        for cell in packet["cells"]:
            scale = float(cell.get("h2", cell.get("r1")))
            matching = [base_cell for base_cell in base_packet["cells"] if np.isclose(float(base_cell.get("h2", base_cell.get("r1"))), scale, rtol=0.0, atol=1e-14)]
            if len(matching) != 1:
                raise ValueError(f"base scale is missing or duplicated: {packet['method']} {packet['geometry']} {scale}")
            base_cell = matching[0]
            for estimator, key in (("Single", "single_rows"), ("Multi", "multi_rows")):
                combined_rows = list(base_cell[key]) + list(cell[key])
                points = fitted(combined_rows)
                total = len(combined_rows)
                if len(points):
                    reverse = cKDTree(points).query(reference, k=1, workers=-1)[0]
                    e_reverse = float(reverse.max())
                    e_reverse_mean = float(reverse.mean())
                else:
                    e_reverse = float("nan")
                    e_reverse_mean = float("nan")
                failure_rate = (total - len(points)) / total
                output_rows.append({
                    "method": packet["method"],
                    "geometry": packet["geometry"],
                    "repeat": packet["repeat"],
                    "estimator": estimator,
                    "scale": scale,
                    "query_count": total,
                    "query_success_count": len(points),
                    "original_query_count": len(base_cell[key]),
                    "added_query_count": len(cell[key]),
                    "query_failure_rate": failure_rate,
                    "reliable_query_gate": failure_rate <= float(config["validation"]["query_failure_rate_max"]),
                    "E_reverse": e_reverse,
                    "E_reverse_mean": e_reverse_mean,
                    "delta_query": delta,
                    "query_grid_mismatch": query_grid_mismatch,
                    "reverse_to_query_ratio": e_reverse / delta if np.isfinite(e_reverse) else float("nan"),
                    "base_config_hash": base_packet["config_hash"],
                    "incremental_config_hash": packet["config_hash"],
                    "scientific_code_hash": packet.get("code_hash"),
                    "observed_sha256": packet["seed"]["observed_sha256"],
                })
    if not output_rows:
        raise RuntimeError("no GPU packets found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)


if __name__ == "__main__":
    main()
