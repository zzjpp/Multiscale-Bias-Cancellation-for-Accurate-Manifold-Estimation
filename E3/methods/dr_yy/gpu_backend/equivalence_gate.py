from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu_backend"))

from dryy.config import code_hash, load_config
from dryy.geometry import geometries_from_config, noisy_sample
from dryy.runner import _dr_cell
from dryy.yy import YYFailure
from dryy.yy_common import evaluate_common_start, prepare_common_start
from dryy_gpu import (
    dr_cell_gpu,
    evaluate_common_start_gpu,
    make_candidate_cache_gpu,
    prepare_common_start_gpu,
)
from run_parent_gpu import backend_code_hash


ATOL = 2e-8
RTOL = 2e-7


def _compare_point(cpu, gpu, label: str, checks: list[dict], errors: list[str]) -> None:
    if cpu is None or gpu is None:
        passed = cpu is None and gpu is None
        maximum = None
    else:
        cpu_array = np.asarray(cpu, dtype=float)
        gpu_array = np.asarray(gpu, dtype=float)
        maximum = float(np.max(np.abs(cpu_array - gpu_array)))
        passed = bool(np.allclose(cpu_array, gpu_array, rtol=RTOL, atol=ATOL))
    checks.append({"name": label, "passed": passed, "max_abs_difference": maximum})
    if not passed:
        errors.append(label)


def _compare_rows(cpu_rows, gpu_rows, label: str, checks: list[dict], errors: list[str]) -> None:
    for index, (cpu_row, gpu_row) in enumerate(zip(cpu_rows, gpu_rows)):
        stem = f"{label}:q{index}"
        status_ok = cpu_row["status"] == gpu_row["status"]
        reason_ok = cpu_row.get("reason") == gpu_row.get("reason")
        checks.append({"name": stem + ":status", "passed": status_ok, "cpu": cpu_row["status"], "gpu": gpu_row["status"]})
        checks.append({"name": stem + ":reason", "passed": reason_ok, "cpu": cpu_row.get("reason"), "gpu": gpu_row.get("reason")})
        if not status_ok:
            errors.append(stem + ":status")
        if not reason_ok:
            errors.append(stem + ":reason")
        if status_ok and cpu_row["status"] == "success":
            _compare_point(cpu_row["point"], gpu_row["point"], stem + ":point", checks, errors)


def run_gate() -> dict:
    config, config_hash = load_config("e3")
    checks: list[dict] = []
    errors: list[str] = []
    dr_scales = [min(config["dr"]["bandwidths_h2"]), 0.2, max(config["dr"]["bandwidths_h2"])]
    yy_scales = list(config["yy"]["central_kappas"])

    for geometry in geometries_from_config(config):
        observations, _ = noisy_sample(config, geometry, 0, int(config["n"]))
        truth = geometry.query_grid(int(config["N_q"]), int(config["query_seed"]))[:1]
        for h2 in dr_scales:
            h1 = float(h2) * float(config["dr"]["h1_ratio"])
            cpu = _dr_cell(observations, geometry, truth, float(config["sigma"]), h1, float(h2), config)
            gpu = dr_cell_gpu(observations, geometry, truth, float(config["sigma"]), h1, float(h2), config)
            label = f"DR:{geometry.key}:h2={h2:g}"
            _compare_rows(cpu["single_rows"], gpu["single_rows"], label + ":Single", checks, errors)
            _compare_rows(cpu["multi_rows"], gpu["multi_rows"], label + ":Multi", checks, errors)
            weights_ok = np.array_equal(cpu["weights"]["weights"], gpu["weights"]["weights"])
            checks.append({"name": label + ":weights", "passed": bool(weights_ok)})
            if not weights_ok:
                errors.append(label + ":weights")
            cp.get_default_memory_pool().free_all_blocks()

        tree = cKDTree(observations)
        controls = dict(config["yy"])
        controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
        truth_point = truth[0]
        try:
            cpu_start = prepare_common_start(observations, tree, geometry, truth_point, float(config["sigma"]), controls)
        except YYFailure as error:
            cpu_start = error
        try:
            gpu_start = prepare_common_start_gpu(observations, tree, geometry, truth_point, float(config["sigma"]), controls)
        except YYFailure as error:
            gpu_start = error
        start_class_ok = isinstance(cpu_start, YYFailure) == isinstance(gpu_start, YYFailure)
        checks.append({"name": f"YY:{geometry.key}:common_start_class", "passed": start_class_ok})
        if not start_class_ok:
            errors.append(f"YY:{geometry.key}:common_start_class")
            continue
        if isinstance(cpu_start, YYFailure):
            reason_ok = cpu_start.reason == gpu_start.reason
            checks.append({"name": f"YY:{geometry.key}:common_start_reason", "passed": reason_ok})
            if not reason_ok:
                errors.append(f"YY:{geometry.key}:common_start_reason")
            continue
        _compare_point(cpu_start["center"], gpu_start["center"], f"YY:{geometry.key}:center", checks, errors)
        _compare_point(cpu_start["axis"], gpu_start["axis"], f"YY:{geometry.key}:axis", checks, errors)
        cached = make_candidate_cache_gpu(
            observations, tree, gpu_start, float(config["sigma"]), max(yy_scales), controls
        )
        gpu_digests = []
        for central in yy_scales:
            cpu = evaluate_common_start(
                observations, tree, geometry, cpu_start, float(config["sigma"]), float(central), controls
            )
            gpu = evaluate_common_start_gpu(
                observations, tree, geometry, gpu_start, float(config["sigma"]), float(central), controls,
                cached_vectors=cached,
            )
            label = f"YY:{geometry.key}:kappa={central:g}"
            gpu_digests.append(gpu["common_start_digest"])
            for estimator in ("single", "multi"):
                status_key = estimator + "_status"
                status_ok = cpu[status_key] == gpu[status_key]
                checks.append({"name": label + ":" + status_key, "passed": status_ok, "cpu": cpu[status_key], "gpu": gpu[status_key]})
                if not status_ok:
                    errors.append(label + ":" + status_key)
                if status_ok and cpu[status_key] == "success":
                    _compare_point(cpu[estimator], gpu[estimator], label + ":" + estimator, checks, errors)
            cpu_stage = [(row["status"], row["reason"]) for row in cpu["stages"]]
            gpu_stage = [(row["status"], row["reason"]) for row in gpu["stages"]]
            stage_ok = cpu_stage == gpu_stage
            checks.append({"name": label + ":stage_classification", "passed": stage_ok, "cpu": cpu_stage, "gpu": gpu_stage})
            if not stage_ok:
                errors.append(label + ":stage_classification")
            weights_ok = (
                np.array_equal(cpu["weights"]["axial"], gpu["weights"]["axial"])
                and np.array_equal(cpu["weights"]["complementary"], gpu["weights"]["complementary"])
            )
            checks.append({"name": label + ":directional_weights", "passed": bool(weights_ok)})
            if not weights_ok:
                errors.append(label + ":directional_weights")
        digest_ok = len(set(gpu_digests)) == 1 and gpu_digests[0] == gpu_start["digest"]
        checks.append({"name": f"YY:{geometry.key}:common_start_all_kappa", "passed": digest_ok})
        if not digest_ok:
            errors.append(f"YY:{geometry.key}:common_start_all_kappa")
        cp.get_default_memory_pool().free_all_blocks()

    return {
        "schema_version": "DR_YY_CPU_GPU_EQUIVALENCE_V1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "config_hash": config_hash,
        "cpu_scientific_code_hash": code_hash(),
        "gpu_backend_code_hash": backend_code_hash(),
        "gpu": {
            "cupy": cp.__version__,
            "device_count_visible": cp.cuda.runtime.getDeviceCount(),
            "pci_bus_id": cp.cuda.Device().pci_bus_id,
        },
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "dr_scales": dr_scales,
        "yy_central_kappas": yy_scales,
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "check_count": report["check_count"],
        "errors": report["errors"][:30], "gpu_backend_code_hash": report["gpu_backend_code_hash"],
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
