from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .experiment import config_hash, source_hash, write_json


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slope_rows(config, summary):
    rows = []
    for geometry in config["geometries"]:
        block = sorted(
            [row for row in summary if row["geometry"] == geometry], key=lambda row: row["sigma"]
        )
        for window, count in config["slope_windows"].items():
            selected = block[: int(count)]
            for estimator, field, target in (
                ("Single", "single_E_infinity", 2),
                ("Multi", "multi_E_infinity", 4),
            ):
                x = np.log([row["sigma"] for row in selected])
                y = np.log([row[field] for row in selected])
                slope, intercept = np.polyfit(x, y, 1)
                predicted = slope * x + intercept
                ss_res = float(np.sum((y - predicted) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                rows.append(
                    {
                        "method": "YY",
                        "geometry": geometry,
                        "estimator": estimator,
                        "target_order": target,
                        "window": window,
                        "point_count": len(selected),
                        "sigma_min": min(row["sigma"] for row in selected),
                        "sigma_max": max(row["sigma"] for row in selected),
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
                    }
                )
    return rows


def aggregate(root: Path):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    packet_paths = sorted((root / "results" / "packets").glob("*.json"))
    packets = [json.loads(path.read_text(encoding="utf-8")) for path in packet_paths]
    expected = len(config["geometries"]) * len(config["sigma_grid"]) * len(config["resolutions"])
    expected_config = config_hash(config)
    expected_source = source_hash(root)
    errors = []
    if len(packets) != expected:
        errors.append(f"packet_count:{len(packets)}!={expected}")
    for packet in packets:
        identity = f"{packet.get('geometry')}:{packet.get('sigma')}:{packet.get('resolution')}"
        if packet.get("schema_version") != "E1_YY_THEORY_PATH_CELL_V1":
            errors.append(f"schema:{identity}")
        if packet.get("config_sha256") != expected_config:
            errors.append(f"config_hash:{identity}")
        if packet.get("source_sha256") != expected_source:
            errors.append(f"source_hash:{identity}")
        if packet.get("backend") != config["audit_gates"]["required_backend"]:
            errors.append(f"backend:{identity}")
        gpu = packet.get("gpu", {})
        if gpu.get("device_name") != config["audit_gates"]["required_gpu"] or gpu.get("float_precision") != "float64":
            errors.append(f"gpu_metadata:{identity}")
        if packet.get("failure_count"):
            errors.append(f"failures:{identity}:{packet.get('failure_count')}")
    by_key = {
        (packet["geometry"], round(float(packet["sigma"]), 12), packet["resolution"]): packet
        for packet in packets
    }
    summary, convergence = [], []
    weight_residuals = []
    for geometry in config["geometries"]:
        for sigma in config["sigma_grid"]:
            production = by_key.get((geometry, round(float(sigma), 12), "production"))
            reference = by_key.get((geometry, round(float(sigma), 12), "reference"))
            if production is None or reference is None:
                errors.append(f"missing:{geometry}:{sigma}")
                continue
            for packet in (production, reference):
                expected_queries = int(
                    config["geometries"][geometry]["query_resolution"][packet["resolution"]]
                )
                if geometry == "sphere2_r10":
                    expected_queries += 2
                if packet["query_count"] != expected_queries:
                    errors.append(f"query_count:{geometry}:{sigma}:{packet['resolution']}")
                weight_residuals.append(float(packet["weights"]["max_abs_residual"]))
            metric_differences = {
                key: abs(float(production["metrics"][key]) - float(reference["metrics"][key]))
                for key in (
                    "single_E_infinity",
                    "single_E_average",
                    "multi_E_infinity",
                    "multi_E_average",
                )
            }
            convergence.append(
                {
                    "method": "YY",
                    "geometry": geometry,
                    "sigma": float(sigma),
                    **{f"{key}_production_reference_difference": value for key, value in metric_differences.items()},
                    "maximum_metric_difference": max(metric_differences.values()),
                }
            )
            summary.append(
                {
                    "method": "YY",
                    "geometry": geometry,
                    "sigma": float(sigma),
                    "r0": production["r0"],
                    "r1_low": production["r1"][0],
                    "r1_single": production["r1"][1],
                    "r1_high": production["r1"][2],
                    "r2": production["r2"],
                    "query_count": production["query_count"],
                    "single_E_infinity": production["metrics"]["single_E_infinity"],
                    "single_E_average": production["metrics"]["single_E_average"],
                    "multi_E_infinity": production["metrics"]["multi_E_infinity"],
                    "multi_E_average": production["metrics"]["multi_E_average"],
                    "single_scaled_sigma2": production["metrics"]["single_E_infinity"] / float(sigma) ** 2,
                    "multi_scaled_sigma4": production["metrics"]["multi_E_infinity"] / float(sigma) ** 4,
                }
            )
    slopes = slope_rows(config, summary) if len(summary) == len(config["geometries"]) * len(config["sigma_grid"]) else []
    gate_path = root / "results" / "gpu_equivalence_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
    gate_valid = bool(
        gate
        and gate.get("status") == "PASS"
        and gate.get("config_sha256") == expected_config
        and gate.get("source_sha256") == expected_source
    )
    min_multi = min((row["multi_E_infinity"] for row in summary), default=float("nan"))
    max_difference = max((row["maximum_metric_difference"] for row in convergence), default=float("inf"))
    max_weight_residual = max(weight_residuals, default=float("inf"))
    checks = {
        "all_declared_packets_present": len(packets) == expected and not any(item.startswith("missing:") for item in errors),
        "packet_schema_config_and_source_hashes": not any(item.startswith(("schema:", "config_hash:", "source_hash:")) for item in errors),
        "formal_backend_is_float64_NVIDIA_L40": not any(item.startswith(("backend:", "gpu_metadata:")) for item in errors),
        "cpu_gpu_equivalence_gate": gate_valid,
        "all_declared_cells_and_queries_resolved": not any(item.startswith("failures:") for item in errors),
        "weight_constraints": max_weight_residual <= config["audit_gates"]["weight_constraint_max"],
        "production_reference_metric_difference_below_fraction_of_minimum_multi_error": max_difference <= config["audit_gates"]["reference_metric_difference_fraction_of_min_multi_error"] * min_multi,
    }
    audit = {
        "schema_version": "E1_YY_THEORY_PATH_AUDIT_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "packet_count": len(packets),
        "expected_packet_count": expected,
        "config_sha256": expected_config,
        "source_sha256": expected_source,
        "gpu_equivalence_gate": gate,
        "numerical_maxima": {
            "weight_constraint_residual": max_weight_residual,
            "production_reference_metric_difference": max_difference,
            "minimum_production_multi_E_infinity": min_multi,
            "metric_difference_over_minimum_multi_error": max_difference / min_multi,
        },
        "population_integration": {
            "description": config["population_integration"]["noise_reduction"],
            "manifold_quadrature": "periodic trapezoid for circle/torus and Gauss-Jacobi latitude with symmetry-exact signed-axis angular moments for spheres",
            "query_quadrature": "canonical query for homogeneous q=1 cases; periodic phase quadrature for the Clifford torus; Gauss-Legendre marginal quadrature plus endpoints for S^2 in R^10",
            "monte_carlo": False,
        },
    }
    results = root / "results"
    write_csv(results / "theory_path_errors.csv", summary)
    write_csv(results / "slope_stability.csv", slopes)
    write_csv(results / "numerical_convergence.csv", convergence)
    write_json(results / "AUDIT.json", audit)
    return audit
