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
PILOT = ROOT.parent / "E3_Erev_gpu_pilot"
DRYY = E3 / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))
sys.path.insert(0, str(DRYY / "gpu_backend"))

from dryy.config import canonical_json, load_config as load_base_config  # noqa: E402
from dryy.geometry import array_hash, geometries_from_config, noisy_sample  # noqa: E402
from dryy.io import read_packet  # noqa: E402
import run_parent_gpu as gpu_parent  # noqa: E402


class IncrementalGeometry:
    def __init__(self, base, queries: np.ndarray):
        self._base = base
        self._queries = np.asarray(queries, dtype=float)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def query_grid(self, count: int, query_seed: int) -> np.ndarray:
        if int(count) != len(self._queries):
            raise ValueError("incremental query count mismatch")
        return self._queries.copy()


def complement(base_geometry, base_count: int, target_count: int, query_seed: int) -> np.ndarray:
    base = base_geometry.query_grid(base_count, query_seed)
    target = base_geometry.query_grid(target_count, query_seed)
    distances = cKDTree(base).query(target, k=1, workers=-1)[0]
    added = target[distances > 1e-12]
    if len(added) != target_count - base_count:
        raise RuntimeError(
            f"expected {target_count - base_count} complementary queries, got {len(added)}"
        )
    merged = np.vstack((base, added))
    mismatch = max(
        cKDTree(target).query(merged, k=1, workers=-1)[0].max(),
        cKDTree(merged).query(target, k=1, workers=-1)[0].max(),
    )
    if mismatch > 1e-14:
        raise RuntimeError(f"nested query-grid mismatch {mismatch:.3e}")
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("DR", "YY"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--stage", choices=(1, 2), type=int, required=True)
    parser.add_argument("--base-packet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    design = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    specification = design["geometries"].get(args.geometry)
    if specification is None:
        raise ValueError("geometry is outside the frozen high-density design")
    targets = [int(value) for value in specification["query_count_stages"]]
    target_count = targets[args.stage - 1]
    base_count = 1024 if args.stage == 1 else targets[0]
    added_count = target_count - base_count
    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise ValueError("invalid shard identity")

    base_config, _ = load_base_config("e3")
    overlay = json.loads(json.dumps(base_config))
    shard_indices = np.arange(added_count, dtype=int)[args.shard_index::args.shard_count]
    overlay["N_q"] = int(len(shard_indices))
    overlay["erev_high_density"] = {
        "schema_version": design["schema_version"],
        "stage": args.stage,
        "base_query_count": base_count,
        "target_query_count": target_count,
        "added_query_count": added_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_query_count": int(len(shard_indices)),
        "shard_indexing": "zero-based strided partition of the deterministic complement",
        "reference_count": int(design["reference_count"]),
        "design_sha256": hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest(),
    }
    overlay_hash = hashlib.sha256(canonical_json(overlay)).hexdigest()
    base_geometry = next(
        geometry for geometry in geometries_from_config(overlay)
        if geometry.key == args.geometry
    )
    new_queries = complement(
        base_geometry, base_count, target_count, int(overlay["query_seed"])
    )
    shard_queries = new_queries[shard_indices]
    incremental_geometry = IncrementalGeometry(base_geometry, shard_queries)

    base_packet_path = (
        args.base_packet_root /
        f"e3_{args.method.lower()}_{args.geometry}_r00.json.gz"
    )
    base_packet = read_packet(base_packet_path)
    expected_observed_hash = base_packet["seed"]["observed_sha256"]
    generated_observation, generated_lineage = noisy_sample(
        overlay, base_geometry, 0, int(overlay["n"])
    )
    observation = generated_observation
    lineage = generated_lineage
    observation_source = "regenerated with server environment"
    if array_hash(observation) != expected_observed_hash:
        asset = PILOT / "observations_numpy244" / f"{args.geometry}_r00_observed.npy"
        manifest_path = PILOT / "observations_numpy244" / "manifest.json"
        if not asset.exists() or not manifest_path.exists():
            raise RuntimeError("observation mismatch and no immutable matching asset exists")
        candidate = np.load(asset, allow_pickle=False)
        if array_hash(candidate) != expected_observed_hash:
            raise RuntimeError("neither regenerated nor immutable observation matches base packet")
        observation = candidate
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches = [
            row for row in manifest["records"]
            if row["geometry"] == args.geometry and int(row["repeat"]) == 0
        ]
        if len(matches) != 1:
            raise RuntimeError("immutable observation manifest identity failure")
        lineage = {
            "algorithm": "PCG64DXSM",
            "clean_seed": matches[0]["clean_seed"],
            "noise_seed": matches[0]["noise_seed"],
            "observed_sha256": expected_observed_hash,
        }
        observation_source = "immutable NumPy-matched observation asset"
    lineage = dict(lineage)
    lineage["observed_sha256"] = expected_observed_hash
    lineage["observation_source"] = observation_source

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    shard_root = (
        args.output_root / f"stage{args.stage}" / "shards" /
        f"s{args.shard_index:02d}of{args.shard_count:02d}"
    )
    destination = (
        shard_root / "packets" /
        f"e3_{args.method.lower()}_{args.geometry}_r00_"
        f"s{args.shard_index:02d}of{args.shard_count:02d}.json.gz"
    )
    if destination.exists():
        print(f"existing={destination}")
        return

    original_loader = gpu_parent.load_config
    original_geometry_loader = gpu_parent.geometry_from_config
    original_sampler = gpu_parent.noisy_sample

    def matched_sampler(config, geometry, repeat, n):
        if geometry.key != args.geometry or int(repeat) != 0 or int(n) != len(observation):
            raise RuntimeError("matched observation request mismatch")
        return observation.copy(), dict(lineage)

    gpu_parent.load_config = lambda experiment: (overlay, overlay_hash)
    gpu_parent.geometry_from_config = lambda config, key: incremental_geometry
    gpu_parent.noisy_sample = matched_sampler
    try:
        path = gpu_parent.run_parent_gpu(
            "e3", args.method, args.geometry, 0,
            shard_root, overwrite=False,
        )
    finally:
        gpu_parent.load_config = original_loader
        gpu_parent.geometry_from_config = original_geometry_loader
        gpu_parent.noisy_sample = original_sampler
    raw_path = Path(path)
    if not raw_path.exists():
        raise RuntimeError(f"GPU runner did not create packet: {raw_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
