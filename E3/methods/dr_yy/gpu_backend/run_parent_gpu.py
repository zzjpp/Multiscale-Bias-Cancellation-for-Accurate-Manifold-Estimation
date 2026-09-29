from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import platform
from pathlib import Path
import socket
import sys
import time

import cupy as cp
import numpy as np
import scipy
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu_backend"))

from dryy.coefficients import yy_weights
from dryy.config import code_hash, load_config
from dryy.geometry import geometries_from_config, noisy_sample
from dryy.io import write_packet
from dryy.metrics import summarize_points
from dryy.yy import YYFailure
from dryy_gpu import (
    _failure,
    _success,
    dr_cell_gpu,
    evaluate_common_start_gpu,
    make_candidate_cache_gpu,
    prepare_common_start_gpu,
    yy_cell_gpu,
)


def backend_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "gpu_backend").glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def geometry_from_config(config: dict, key: str):
    matches = [geometry for geometry in geometries_from_config(config) if geometry.key == key]
    if len(matches) != 1:
        raise ValueError("unknown geometry")
    return matches[0]


def run_parent_gpu(experiment: str, method: str, geometry_key: str, repeat: int,
                   output_root: Path, overwrite: bool = False) -> Path:
    config, config_hash = load_config(experiment)
    method = method.upper()
    if method not in {"DR", "YY"}:
        raise ValueError("method must be DR or YY")
    if repeat not in range(int(config["repeats"])):
        raise ValueError("repeat out of range")
    geometry = geometry_from_config(config, geometry_key)
    queries = geometry.query_grid(int(config["N_q"]), int(config["query_seed"]))
    destination = Path(output_root) / "packets" / f"{experiment}_{method.lower()}_{geometry.key}_r{repeat:02d}.json.gz"
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    started = time.perf_counter()

    if experiment == "e2":
        observed, lineage = noisy_sample(config, geometry, repeat, int(config["maximum_n"]))
        cells = []
        for n in config["n_values"]:
            data = observed[: int(n)]
            if method == "DR":
                cell = dr_cell_gpu(
                    data, geometry, queries, float(config["sigma"]),
                    float(config["dr"]["h1"]), float(config["dr"]["h2"]), config,
                )
            else:
                cell = yy_cell_gpu(
                    data, cKDTree(data), geometry, queries, float(config["sigma"]),
                    float(config["yy"]["central_kappa"]), config,
                )
            cell["n"] = int(n)
            cells.append(cell)
            cp.get_default_memory_pool().free_all_blocks()
        scale_name = "n"
    else:
        observed, lineage = noisy_sample(config, geometry, repeat, int(config["n"]))
        cells = []
        if method == "DR":
            for h2 in config["dr"]["bandwidths_h2"]:
                cells.append(dr_cell_gpu(
                    observed, geometry, queries, float(config["sigma"]),
                    float(h2) * float(config["dr"]["h1_ratio"]), float(h2), config,
                ))
                cp.get_default_memory_pool().free_all_blocks()
            scale_name = "h2"
        else:
            scales = list(config["yy"]["central_kappas"])
            tree = cKDTree(observed)
            controls = dict(config["yy"])
            controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
            per_scale = {float(value): {"single": [], "multi": [], "digests": []} for value in scales}
            for truth in queries:
                try:
                    common = prepare_common_start_gpu(observed, tree, geometry, truth, float(config["sigma"]), controls)
                    cached = make_candidate_cache_gpu(
                        observed, tree, common, float(config["sigma"]), max(scales), controls
                    )
                except YYFailure as error:
                    for value in scales:
                        per_scale[float(value)]["single"].append(_failure(error.reason, **error.diagnostics))
                        per_scale[float(value)]["multi"].append(_failure(error.reason, **error.diagnostics))
                        per_scale[float(value)]["digests"].append(None)
                    continue
                for value in scales:
                    fitted = evaluate_common_start_gpu(
                        observed, tree, geometry, common, float(config["sigma"]),
                        float(value), controls, cached_vectors=cached,
                    )
                    per_scale[float(value)]["digests"].append(common["digest"])
                    if fitted["single_status"] == "success":
                        per_scale[float(value)]["single"].append(
                            _success(fitted["single"], common_start_digest=common["digest"])
                        )
                    else:
                        per_scale[float(value)]["single"].append(
                            _failure("second_stage_mass_failure", common_start_digest=common["digest"])
                        )
                    if fitted["multi_status"] == "success":
                        per_scale[float(value)]["multi"].append(
                            _success(fitted["multi"], common_start_digest=common["digest"])
                        )
                    else:
                        failed = [stage for stage in fitted["stages"] if stage["status"] != "success"]
                        per_scale[float(value)]["multi"].append(
                            _failure("second_stage_incomplete", common_start_digest=common["digest"], failed_stages=failed)
                        )
            for value in scales:
                rows = per_scale[float(value)]
                kappas = float(value) * np.asarray(controls["relative_radii"], dtype=float)
                cells.append({
                    "method": "YY", "central_kappa": float(value),
                    "r1": float(value) * float(config["sigma"]),
                    "kappas": kappas, "radii": float(config["sigma"]) * kappas,
                    "weights": yy_weights(geometry.d, geometry.q, kappas, controls["kernel_power"]),
                    "Single": summarize_points(geometry, rows["single"], "point"),
                    "Multi": summarize_points(geometry, rows["multi"], "point"),
                    "single_rows": rows["single"], "multi_rows": rows["multi"],
                    "common_start_digests": rows["digests"],
                })
            scale_name = "central_kappa"

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"].decode() if isinstance(properties["name"], bytes) else str(properties["name"])
    packet = {
        "schema_version": f"{experiment.upper()}_{method}_PARENT_V1",
        "experiment": experiment, "method": method, "geometry": geometry.key,
        "d": geometry.d, "D": geometry.D, "q": geometry.q, "repeat": int(repeat),
        "config_hash": config_hash,
        "code_hash": code_hash(),
        "gpu_backend_code_hash": backend_code_hash(),
        "sigma": float(config["sigma"]), "query_count": int(config["N_q"]),
        "query_coordinates": queries, "seed": lineage, "scale_name": scale_name,
        "cells": cells, "execution_complete": True,
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "backend": "cupy", "cupy": cp.__version__, "cuda_visible_device": 0,
            "gpu_name": name, "gpu_pci_bus_id": device.pci_bus_id,
        },
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_packet(destination, packet)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--method", choices=("DR", "YY"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_parent_gpu(
        args.experiment, args.method, args.geometry, args.repeat, args.output_root,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
