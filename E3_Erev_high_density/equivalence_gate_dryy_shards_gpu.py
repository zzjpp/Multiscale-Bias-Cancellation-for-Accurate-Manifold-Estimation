from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
DRYY = ROOT.parent / "E3" / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))
sys.path.insert(0, str(DRYY / "gpu_backend"))

from dryy.config import load_config  # noqa: E402
from dryy.geometry import geometries_from_config, noisy_sample  # noqa: E402
from dryy_gpu import dr_cell_gpu, yy_cell_gpu  # noqa: E402


def merge_rows(shards: list[dict], key: str, total: int) -> list[dict]:
    merged = [None] * total
    count = len(shards)
    for shard_index, shard in enumerate(shards):
        indices = np.arange(total)[shard_index::count]
        for local, global_index in enumerate(indices):
            merged[global_index] = shard[key][local]
    if any(row is None for row in merged):
        raise RuntimeError("incomplete equivalence merge")
    return merged


def compare(reference: list[dict], candidate: list[dict], label: str, failures: list[str]) -> None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left["status"] != right["status"]:
            failures.append(f"{label}/query={index}: status mismatch")
            continue
        if left.get("reason") != right.get("reason"):
            failures.append(f"{label}/query={index}: failure reason mismatch")
        if left["status"] == "success":
            error = float(np.max(np.abs(np.asarray(left["point"]) - np.asarray(right["point"]))))
            if error > 1e-12:
                failures.append(f"{label}/query={index}: point mismatch {error:.3e}")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    config, _ = load_config("e3")
    geometry = next(g for g in geometries_from_config(config) if g.key == "sphere2_r3")
    observed, lineage = noisy_sample(config, geometry, 0, int(config["n"]))
    queries = geometry.query_grid(8, int(config["query_seed"]))
    shard_count = 4
    query_shards = [queries[index::shard_count] for index in range(shard_count)]
    failures: list[str] = []

    h2 = 0.45
    h1 = h2 * float(config["dr"]["h1_ratio"])
    full_dr = dr_cell_gpu(observed, geometry, queries, float(config["sigma"]), h1, h2, config)
    shard_dr = [
        dr_cell_gpu(observed, geometry, shard, float(config["sigma"]), h1, h2, config)
        for shard in query_shards
    ]
    compare(full_dr["single_rows"], merge_rows(shard_dr, "single_rows", len(queries)), "DR/Single", failures)
    compare(full_dr["multi_rows"], merge_rows(shard_dr, "multi_rows", len(queries)), "DR/Multi", failures)

    central_kappa = float(config["yy"]["central_kappas"][10])
    tree = cKDTree(observed)
    full_yy = yy_cell_gpu(
        observed, tree, geometry, queries, float(config["sigma"]), central_kappa, config
    )
    shard_yy = [
        yy_cell_gpu(observed, tree, geometry, shard, float(config["sigma"]), central_kappa, config)
        for shard in query_shards
    ]
    compare(full_yy["single_rows"], merge_rows(shard_yy, "single_rows", len(queries)), "YY/Single", failures)
    compare(full_yy["multi_rows"], merge_rows(shard_yy, "multi_rows", len(queries)), "YY/Multi", failures)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "geometry": geometry.key,
        "observed_sha256": lineage["observed_sha256"],
        "query_count": len(queries),
        "shard_count": shard_count,
        "dr_h2": h2,
        "yy_central_kappa": central_kappa,
        "point_absolute_tolerance": 1e-12,
        "failures": failures,
    }
    output = ROOT / "DRYY_GPU_SHARD_EQUIVALENCE.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
