from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import GEOMETRIES
from .population import PopulationYY


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hash(root: Path):
    digest = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py")) + sorted(root.glob("*.py")) + [root / "config.json"]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gpu_metadata(device):
    import cupy as cp

    cp.cuda.Device(int(device)).use()
    properties = cp.cuda.runtime.getDeviceProperties(int(device))
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "device_index": int(device),
        "device_name": str(name),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "float_precision": "float64",
    }


def make_operator(config, geometry, sigma, resolution_name, backend, device):
    design = config["geometries"][geometry.key]
    resolution = config["resolutions"][resolution_name]
    arguments = dict(
        geometry=geometry,
        sigma=float(sigma),
        latent_resolution=int(design["integration_resolution"][resolution_name]),
        radial_order=int(resolution["radial_order"]),
        longitudinal_order=int(resolution["longitudinal_order"]),
        kappas=config["yy_path"]["kappas"],
        c0=config["yy_path"]["c0"],
        c2=config["yy_path"]["c2"],
        power=config["kernel_power"],
        tail_padding=config["population_integration"]["gaussian_tail_padding_sigma"],
    )
    if backend == "cpu":
        return PopulationYY(**arguments)
    if backend == "gpu_cupy":
        from .gpu_population import GpuPopulationYY

        return GpuPopulationYY(**arguments, device=device)
    raise ValueError(f"unknown backend: {backend}")


def run_cell(config, geometry_key, sigma, resolution_name, backend="cpu", device=0):
    geometry = GEOMETRIES[geometry_key]
    design = config["geometries"][geometry_key]
    query_resolution = int(design["query_resolution"][resolution_name])
    amplitudes, average_weights, query_coordinates = geometry.query_rule(query_resolution)
    operator = make_operator(config, geometry, sigma, resolution_name, backend, device)
    truth = geometry.canonical_truth()
    query_rows = []
    single_points, multi_points, failures = [], [], []
    for query_id, (amplitude, average_weight, coordinate) in enumerate(
        zip(amplitudes, average_weights, query_coordinates)
    ):
        try:
            result = operator.evaluate(float(amplitude))
            single = result["single"]
            multi = result["multi"]
            failure = ""
        except Exception as error:
            result = {}
            single = np.full(geometry.D, np.nan)
            multi = np.full(geometry.D, np.nan)
            failure = f"{type(error).__name__}: {error}"
            failures.append({"query_id": query_id, "reason": failure})
        single_points.append(single)
        multi_points.append(multi)
        query_rows.append(
            {
                "query_id": query_id,
                "coordinate": float(coordinate),
                "preliminary_amplitude": float(amplitude),
                "average_weight": float(average_weight),
                "truth_canonical": truth,
                "single_point": single,
                "multi_point": multi,
                "diagnostics": result,
                "failure": failure,
            }
        )
    single_points = np.asarray(single_points)
    multi_points = np.asarray(multi_points)
    success = np.isfinite(single_points).all(axis=1) & np.isfinite(multi_points).all(axis=1)
    if np.all(success):
        single_distance = geometry.distances(single_points)
        multi_distance = geometry.distances(multi_points)
        normalized_weights = average_weights / np.sum(average_weights)
        metrics = {
            "single_E_infinity": float(np.max(single_distance)),
            "single_E_average": float(np.sum(normalized_weights * single_distance)),
            "multi_E_infinity": float(np.max(multi_distance)),
            "multi_E_average": float(np.sum(normalized_weights * multi_distance)),
            "single_distances": single_distance,
            "multi_distances": multi_distance,
        }
    else:
        metrics = {
            "single_E_infinity": float("nan"),
            "single_E_average": float("nan"),
            "multi_E_infinity": float("nan"),
            "multi_E_average": float("nan"),
            "single_distances": [],
            "multi_distances": [],
        }
    packet = {
        "schema_version": "E1_YY_THEORY_PATH_CELL_V1",
        "backend": backend,
        "method": "YY",
        "geometry": geometry_key,
        "sigma": float(sigma),
        "r0": float(config["yy_path"]["c0"]) * float(sigma),
        "r1": [float(value) * float(sigma) for value in config["yy_path"]["kappas"]],
        "r2": float(config["yy_path"]["c2"]) * float(sigma) * np.sqrt(np.log(1.0 / float(sigma))),
        "resolution": resolution_name,
        "resolution_settings": {
            **config["resolutions"][resolution_name],
            "latent_resolution": int(design["integration_resolution"][resolution_name]),
            "actual_latent_node_count": geometry.integration_node_count(
                int(design["integration_resolution"][resolution_name])
            ),
            "query_resolution": query_resolution,
            "query_rule": design["query_rule"],
            "latent_rule": design["latent_rule"],
        },
        "query_count": int(len(amplitudes)),
        "success_count": int(np.sum(success)),
        "failure_count": int(np.sum(~success)),
        "failures": failures,
        "weights": operator.weights,
        "query_rows": query_rows,
        "single_points": single_points,
        "multi_points": multi_points,
        "metrics": metrics,
    }
    if backend == "gpu_cupy":
        packet["gpu"] = gpu_metadata(device)
    return packet
