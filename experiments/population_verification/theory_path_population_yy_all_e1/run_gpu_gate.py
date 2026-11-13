from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_yy_path.experiment import config_hash, run_cell, source_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    gate_config = copy.deepcopy(config)
    gate_config["resolutions"]["production"] = {"radial_order": 64, "longitudinal_order": 64}
    for geometry, design in gate_config["geometries"].items():
        design["query_resolution"]["production"] = 1
        design["integration_resolution"]["production"] = (
            1024 if geometry == "circle_r2" else 96 if geometry == "clifford_torus_r4" else 128
        )
    sigma = 0.02
    rows = []
    max_point = 0.0
    max_metric = 0.0
    classifications = True
    for geometry in gate_config["geometries"]:
        cpu = run_cell(gate_config, geometry, sigma, "production", backend="cpu", device=args.device)
        gpu = run_cell(gate_config, geometry, sigma, "production", backend="gpu_cupy", device=args.device)
        equal = [row["failure"] for row in cpu["query_rows"]] == [
            row["failure"] for row in gpu["query_rows"]
        ]
        classifications = classifications and equal
        cpu_points = np.column_stack((cpu["single_points"], cpu["multi_points"]))
        gpu_points = np.column_stack((gpu["single_points"], gpu["multi_points"]))
        point_difference = float(np.nanmax(np.abs(cpu_points - gpu_points)))
        metric_difference = max(
            abs(float(cpu["metrics"][key]) - float(gpu["metrics"][key]))
            for key in (
                "single_E_infinity",
                "single_E_average",
                "multi_E_infinity",
                "multi_E_average",
            )
        )
        max_point = max(max_point, point_difference)
        max_metric = max(max_metric, metric_difference)
        rows.append(
            {
                "geometry": geometry,
                "sigma": sigma,
                "classification_match": equal,
                "max_coordinate_difference": point_difference,
                "max_metric_difference": metric_difference,
                "gate_latent_resolution": gate_config["geometries"][geometry]["integration_resolution"]["production"],
            }
        )
        print(f"{geometry}: point={point_difference:.3e} metric={metric_difference:.3e}", flush=True)
    threshold = float(config["audit_gates"]["cpu_gpu_point_max"])
    checks = {
        "success_failure_classification_identical": classifications,
        "fitted_points_within_tolerance": max_point <= threshold,
        "metrics_within_tolerance": max_metric <= threshold,
    }
    payload = {
        "schema_version": "E1_YY_THEORY_PATH_GPU_GATE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "threshold": threshold,
        "max_coordinate_difference": max_point,
        "max_metric_difference": max_metric,
        "cases": rows,
        "config_sha256": config_hash(config),
        "source_sha256": source_hash(ROOT),
    }
    write_json(ROOT / "results" / "gpu_equivalence_gate.json", payload)
    print(json.dumps({"status": payload["status"], "max_coordinate_difference": max_point}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
