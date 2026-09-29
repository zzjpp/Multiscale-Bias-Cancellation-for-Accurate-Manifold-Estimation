from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
E3 = ROOT.parent / "E3"
DRYY = E3 / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))
sys.path.insert(0, str(DRYY / "gpu_backend"))

from dryy.config import canonical_json, load_config as load_base_config  # noqa: E402
from dryy.geometry import array_hash, geometries_from_config  # noqa: E402
import run_parent_gpu as gpu_parent  # noqa: E402


def load_overlay() -> tuple[dict, str, dict]:
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


class IncrementalGeometry:
    def __init__(self, base, added_queries: np.ndarray):
        self._base = base
        self._added_queries = np.asarray(added_queries, dtype=float)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def query_grid(self, count: int, query_seed: int) -> np.ndarray:
        if int(count) != len(self._added_queries):
            raise ValueError("incremental query count mismatch")
        return self._added_queries.copy()


def added_queries(base_geometry, pilot: dict, query_seed: int) -> np.ndarray:
    old_count = int(pilot["original_query_count"])
    total_count = int(pilot["combined_query_count"])
    old = base_geometry.query_grid(old_count, query_seed)
    dense = base_geometry.query_grid(total_count, query_seed)
    nearest = cKDTree(old).query(dense, k=1, workers=-1)[0]
    added = dense[nearest > 1e-12]
    expected = int(pilot["added_query_count"])
    if len(added) != expected:
        raise RuntimeError(f"expected {expected} new queries, obtained {len(added)}")
    merged = np.vstack((old, added))
    if len(merged) != total_count or cKDTree(merged).query_pairs(1e-12):
        raise RuntimeError("combined query set is not a unique 1024-point set")
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--method", choices=("DR", "YY"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()

    overlay, overlay_hash, pilot = load_overlay()
    if args.geometry not in pilot["geometries"]:
        raise ValueError("geometry is outside the frozen pilot set")
    if args.repeat not in pilot["repeat_indices"]:
        raise ValueError("repeat is outside the frozen pilot set")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(pilot["gpu"]["visible_device"]))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    original_loader = gpu_parent.load_config
    original_geometry_loader = gpu_parent.geometry_from_config
    original_sampler = gpu_parent.noisy_sample
    base_geometry = next(
        geometry for geometry in geometries_from_config(overlay)
        if geometry.key == args.geometry
    )
    new_queries = added_queries(base_geometry, pilot, int(overlay["query_seed"]))
    incremental_geometry = IncrementalGeometry(base_geometry, new_queries)
    observation_path = ROOT / "observations_numpy244" / f"{args.geometry}_r{args.repeat:02d}_observed.npy"
    manifest_path = ROOT / "observations_numpy244" / "manifest.json"
    if not observation_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("matched NumPy-2.4.4 observation assets are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row["geometry"] == args.geometry and row["repeat"] == args.repeat]
    if len(records) != 1:
        raise RuntimeError("matched observation manifest identity failure")
    observation = np.load(observation_path, allow_pickle=False)
    if array_hash(observation) != records[0]["observed_sha256"]:
        raise RuntimeError("matched observation hash failure")

    def matched_sampler(config, geometry, repeat, n):
        if geometry.key != args.geometry or int(repeat) != args.repeat or int(n) != len(observation):
            raise RuntimeError("matched observation request mismatch")
        lineage = {
            "algorithm": "PCG64DXSM",
            "clean_seed": records[0]["clean_seed"],
            "noise_seed": records[0]["noise_seed"],
            "observed_sha256": records[0]["observed_sha256"],
            "observation_source": "uploaded NumPy-2.4.4 array matched to immutable formal packet",
        }
        return observation.copy(), lineage
    destination = (
        args.output_root / "gpu_packets" / "packets" /
        f"e3_{args.method.lower()}_{args.geometry}_r{args.repeat:02d}.json.gz"
    )
    if destination.exists():
        print(f"existing={destination}")
        return
    gpu_parent.load_config = lambda experiment: (overlay, overlay_hash)
    gpu_parent.geometry_from_config = lambda config, key: incremental_geometry
    gpu_parent.noisy_sample = matched_sampler
    try:
        path = gpu_parent.run_parent_gpu(
            "e3", args.method, args.geometry, args.repeat,
            args.output_root / "gpu_packets", overwrite=False,
        )
    finally:
        gpu_parent.load_config = original_loader
        gpu_parent.geometry_from_config = original_geometry_loader
        gpu_parent.noisy_sample = original_sampler
    print(path)


if __name__ == "__main__":
    main()
