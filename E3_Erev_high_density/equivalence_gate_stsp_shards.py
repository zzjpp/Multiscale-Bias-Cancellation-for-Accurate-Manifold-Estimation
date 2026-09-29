from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
STSP = ROOT.parent / "E3" / "methods" / "stsp"
sys.path.insert(0, str(STSP / "src"))

from e3.coefficients import two_scale_weights  # noqa: E402
from e3.config import CONFIG  # noqa: E402
from e3.geometry import BY_KEY, noisy_sample  # noqa: E402
from e3.pipelines import run_cell  # noqa: E402


def compare_row(reference: dict, candidate: dict, method: str, failures: list[str], label: str) -> None:
    left = reference[method]
    right = candidate[method]
    if left["status"] != right["status"]:
        failures.append(f"{label}/{method}: status mismatch")
        return
    if left.get("failure_reason") != right.get("failure_reason"):
        failures.append(f"{label}/{method}: failure reason mismatch")
    left_point = np.asarray(left["point"], dtype=float)
    right_point = np.asarray(right["point"], dtype=float)
    if left["status"] == "success":
        error = float(np.max(np.abs(left_point - right_point)))
        if error > 1e-12:
            failures.append(f"{label}/{method}: point mismatch {error:.3e}")
    elif not (np.isnan(left_point).all() and np.isnan(right_point).all()):
        failures.append(f"{label}/{method}: failed points are not NaN")


def main() -> None:
    geometry = BY_KEY["sphere2_r3"]
    _, _, observed, lineage = noisy_sample(geometry, 0)
    queries = geometry.query_grid(16)
    h2 = 0.45
    weights = two_scale_weights(geometry.d, geometry.D, h2 / 2, h2, CONFIG["sigma"])
    full = run_cell(observed, geometry, h2, queries, weights)
    shard_count = 4
    shards = [run_cell(observed, geometry, h2, queries[index::shard_count], weights) for index in range(shard_count)]
    merged_single = [None] * len(queries)
    merged_multi = [None] * len(queries)
    for index, shard in enumerate(shards):
        global_indices = np.arange(len(queries))[index::shard_count]
        for local, global_index in enumerate(global_indices):
            merged_single[global_index] = shard["single"]["queries"][local]
            merged_multi[global_index] = shard["multi"]["queries"][local]
    failures: list[str] = []
    for index in range(len(queries)):
        for method in ("Single_h1", "Single_h2"):
            compare_row(
                full["single"]["queries"][index], merged_single[index], method,
                failures, f"query={index}",
            )
        compare_row(
            full["multi"]["queries"][index], merged_multi[index], "ENF_Multi",
            failures, f"query={index}",
        )
    report = {
        "status": "PASS" if not failures else "FAIL",
        "geometry": geometry.key,
        "observed_sha256": lineage["observed_sha256"],
        "query_count": len(queries),
        "shard_count": shard_count,
        "h2": h2,
        "point_absolute_tolerance": 1e-12,
        "failures": failures,
    }
    (ROOT / "STSP_SHARD_EQUIVALENCE.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
