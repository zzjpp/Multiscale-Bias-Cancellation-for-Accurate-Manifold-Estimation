from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_theory_path.experiment import config_hash, run_cell, source_hash, write_json  # noqa: E402
from e1_theory_path.gpu_experiment import gpu_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU/GPU numerical-equivalence gate for E1 theory-path population operators.")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    sigma = min(float(value) for value in config["sigma_grid"])
    cases = []
    for method in config["methods"]:
        for geometry in config["geometries"]:
            full_count = int(config["geometries"][geometry]["query_count"])
            selected = [0] if full_count == 1 else list(range(0, full_count, 8))
            for resolution in config["resolutions"]:
                cases.append((method, geometry, sigma, resolution, selected))
    rows = []
    classification_match = True
    max_point_difference = 0.0
    max_metric_difference = 0.0
    for method, geometry, value, resolution, selected in cases:
        cpu = run_cell(config, method, geometry, value, resolution, backend="cpu", query_ids=selected)
        gpu = run_cell(
            config,
            method,
            geometry,
            value,
            resolution,
            backend="gpu_cupy",
            device=args.device,
            query_ids=selected,
        )
        classifications_equal = [row["failure"] for row in cpu["query_rows"]] == [row["failure"] for row in gpu["query_rows"]]
        classification_match = classification_match and classifications_equal
        cpu_points = np.column_stack((np.asarray(cpu["single_points"]), np.asarray(cpu["multi_points"])))
        gpu_points = np.column_stack((np.asarray(gpu["single_points"]), np.asarray(gpu["multi_points"])))
        point_difference = float(np.nanmax(np.linalg.norm(cpu_points - gpu_points, axis=1)))
        metric_difference = max(
            abs(float(cpu["metrics"][key]) - float(gpu["metrics"][key]))
            for key in ("single_E_infinity", "single_E_average", "multi_E_infinity", "multi_E_average")
        )
        max_point_difference = max(max_point_difference, point_difference)
        max_metric_difference = max(max_metric_difference, metric_difference)
        rows.append(
            {
                "method": method,
                "geometry": geometry,
                "sigma": value,
                "resolution": resolution,
                "query_ids": selected,
                "classification_match": classifications_equal,
                "max_fitted_point_difference": point_difference,
                "max_metric_difference": metric_difference,
            }
        )
        print(f"gate {method} {geometry} {resolution}: point={point_difference:.3e} metric={metric_difference:.3e}", flush=True)
    threshold = float(config["audit_gates"]["cpu_gpu_fitted_point_max"])
    checks = {
        "success_failure_classification_identical": classification_match,
        "fitted_points_within_tolerance": max_point_difference <= threshold,
        "metrics_within_tolerance": max_metric_difference <= threshold,
    }
    payload = {
        "schema_version": "E1_THEORY_PATH_GPU_GATE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "threshold": threshold,
        "max_fitted_point_difference": max_point_difference,
        "max_metric_difference": max_metric_difference,
        "cases": rows,
        "gpu": gpu_metadata(args.device),
        "config_sha256": config_hash(config),
        "source_sha256": source_hash(ROOT),
    }
    write_json(ROOT / "results" / "gpu_equivalence_gate.json", payload)
    print(json.dumps({"status": payload["status"], "max_point_difference": max_point_difference}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
