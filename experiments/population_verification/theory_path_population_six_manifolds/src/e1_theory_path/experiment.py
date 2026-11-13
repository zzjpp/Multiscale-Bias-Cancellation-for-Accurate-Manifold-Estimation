from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .coefficients import two_scale_weights
from .geometry import GEOMETRIES
from .population import PopulationDR, PopulationSTSP, estimate_pca_normal, solve_normal_fiber


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py")) + sorted(root.glob("*.py")) + [root / "config.json"]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _operator(
    method: str,
    geometry,
    h: float,
    sigma: float,
    resolution: dict,
    power: int,
    noncentral_radius_padding: float,
    backend: str = "cpu",
    device: int | None = None,
):
    if backend == "gpu_cupy":
        from .gpu_population import GpuPopulationDR, GpuPopulationSTSP

        if method == "DR":
            return GpuPopulationDR(geometry, h, sigma, resolution["dr_latent_count"], device=device,
                                   pca_radial_order=resolution["stsp_radial_order"], pca_padding=noncentral_radius_padding)
        if method == "STSP":
            return GpuPopulationSTSP(
                geometry,
                h,
                sigma,
                resolution["stsp_latent_count"],
                resolution["stsp_radial_order"],
                power,
                noncentral_radius_padding,
                device=device,
            )
        raise ValueError(method)
    if backend != "cpu":
        raise ValueError(f"unknown backend: {backend}")
    if method == "DR":
        return PopulationDR(geometry, h, sigma, resolution["dr_latent_count"],
                            pca_radial_order=resolution["stsp_radial_order"], pca_padding=noncentral_radius_padding)
    if method == "STSP":
        return PopulationSTSP(
            geometry,
            h,
            sigma,
            resolution["stsp_latent_count"],
            resolution["stsp_radial_order"],
            power,
            noncentral_radius_padding,
        )
    raise ValueError(method)


def run_cell(
    config: dict,
    method: str,
    geometry_key: str,
    sigma: float,
    resolution_name: str,
    backend: str = "cpu",
    device: int | None = None,
    query_ids: list[int] | None = None,
):
    geometry = GEOMETRIES[geometry_key]
    design = config["geometries"][geometry_key]
    all_points, all_parameters = geometry.query_grid(design["query_count"], design.get("query_phase", 0.0))
    selected_ids = list(range(len(all_points))) if query_ids is None else [int(index) for index in query_ids]
    points = all_points[selected_ids]
    parameters = all_parameters[selected_ids]
    h1 = float(config["bandwidth_path"]["a1"]) * float(sigma)
    h2 = float(config["bandwidth_path"]["a2"]) * float(sigma)
    resolution = dict(config["resolutions"][resolution_name])
    effective_latent_resolution = int(design["integration_resolution"][resolution_name])
    resolution["dr_latent_count"] = effective_latent_resolution
    resolution["stsp_latent_count"] = effective_latent_resolution
    resolution["latent_rule"] = design["latent_rule"]
    resolution["actual_latent_node_count"] = geometry.integration_node_count(effective_latent_resolution)
    operators = [
        _operator(
            method,
            geometry,
            h,
            sigma,
            resolution,
            config["kernel_power"],
            config["population_integration"]["stsp_noncentral_radius_padding"],
            backend,
            device,
        )
        for h in (h1, h2)
    ]
    weights = two_scale_weights(
        method,
        geometry,
        h1,
        h2,
        sigma,
        config["kernel_power"],
        config["coefficient_quadrature"],
    )
    query_rows = []
    single_points = []
    multi_points = []
    failures = []
    for query_id, truth, parameter in zip(selected_ids, points, parameters):
        basis = None
        single_fit = None
        pca_diagnostics = {}
        fits = []
        multi_failure = ""
        single_failure = ""
        try:
            initial_basis, initial_diag = estimate_pca_normal(operators[0], truth, config["solver"])
            fine_fit = solve_normal_fiber(operators[0], truth, initial_basis, h1, config["solver"])
            basis, reference_diag = estimate_pca_normal(operators[0], fine_fit["point"], config["solver"])
            coarse_fit = solve_normal_fiber(operators[1], fine_fit["point"], basis, h2, config["solver"])
            fits = [fine_fit, coarse_fit]
            pca_diagnostics.update(initial=initial_diag, reference=reference_diag)
            # The successful query set depends only on this algorithm's roots.
            multi = fine_fit["point"] + basis @ (weights["weights"][1] * coarse_fit["coordinate"])
        except Exception as error:
            multi = np.full(geometry.D, np.nan)
            multi_failure = f"{type(error).__name__}: {error}"
        # Single(h2) is an independent baseline and cannot invalidate Multi's S.
        try:
            single_basis, single_diag = estimate_pca_normal(operators[1], truth, config["solver"])
            single_fit = solve_normal_fiber(operators[1], truth, single_basis, h2, config["solver"])
            pca_diagnostics["single"] = single_diag
            single = single_fit["point"]
        except Exception as error:
            single = np.full(geometry.D, np.nan)
            single_failure = f"{type(error).__name__}: {error}"
        failure = multi_failure or single_failure
        if failure:
            failures.append({"query_id": query_id, "reason": failure,
                             "single_failure": single_failure, "multi_failure": multi_failure})
        single_points.append(single)
        multi_points.append(multi)
        query_rows.append({
            "query_id": query_id, "truth": truth, "parameter": parameter,
            "normal_basis": basis, "normal_space_source": "PCA",
            "normal_guided": None if not fits else {
                "reference_point": fine_fit["point"], "reference_basis": basis, "pca_radius": h1,
                "t1": np.zeros(geometry.q), "t2": coarse_fit["coordinate"],
                "target_projected_residual": coarse_fit["final"]["projected_residual"]},
            "pca_diagnostics": pca_diagnostics, "single_fit": single_fit,
            "single_point": single, "multi_point": multi, "fits": fits,
            "failure": failure, "single_failure": single_failure, "multi_failure": multi_failure})
    single_points = np.asarray(single_points)
    multi_points = np.asarray(multi_points)
    single_success = np.isfinite(single_points).all(axis=1)
    success = np.isfinite(multi_points).all(axis=1)
    single_distances = geometry.distances(single_points[single_success]) if np.any(single_success) else np.array([])
    multi_distances = geometry.distances(multi_points[success]) if np.any(success) else np.array([])
    metrics = {
        "single_E_infinity": float(np.max(single_distances)) if len(single_distances) else float("nan"),
        "single_E_average": float(np.mean(single_distances)) if len(single_distances) else float("nan"),
        "multi_E_infinity": float(np.max(multi_distances)) if len(multi_distances) else float("nan"),
        "multi_E_average": float(np.mean(multi_distances)) if len(multi_distances) else float("nan"),
        "single_distances": single_distances, "multi_distances": multi_distances,
    }
    return {
        "schema_version": "E1_THEORY_PATH_CELL_V2_PCA_FIBERS",
        "normal_space_policy": "PCA_initial_and_fine_reference_fibers; unchanged_estimator_operator",
        "backend": backend,
        "method": method,
        "geometry": geometry_key,
        "sigma": float(sigma),
        "h1": h1,
        "h2": h2,
        "resolution": resolution_name,
        "resolution_settings": resolution,
        "query_count": int(len(points)),
        "query_ids": selected_ids,
        "success_count": int(np.sum(success)),
        "single_success_count": int(np.sum(single_success)),
        "single_failure_count": int(np.sum(~single_success)),
        "successful_query_ids": [selected_ids[i] for i in np.flatnonzero(success)],
        "failure_count": int(np.sum(~success)),
        "failures": failures,
        "weights": weights,
        "query_rows": query_rows,
        "single_points": single_points,
        "multi_points": multi_points,
        "metrics": metrics,
    }


def _task(config: dict, task: tuple[str, str, float, str]):
    return run_cell(config, *task)


def run_all(root: Path, workers: int):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    frozen_config_hash = config_hash(config)
    frozen_source_hash = source_hash(root)
    packet_dir = root / "results" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (method, geometry, float(sigma), resolution)
        for method in config["methods"]
        for geometry in config["geometries"]
        for sigma in config["sigma_grid"]
        for resolution in config["resolutions"]
    ]
    pending = []
    completed = []
    for task in tasks:
        method, geometry, sigma, resolution = task
        path = packet_dir / f"{method.lower()}_{geometry}_sigma_{sigma:.6f}_{resolution}.json"
        if path.exists():
            completed.append(path)
        else:
            pending.append((task, path))
    if pending:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(_task, config, task): (task, path) for task, path in pending}
            for future in as_completed(futures):
                task, path = futures[future]
                packet = future.result()
                packet["config_sha256"] = frozen_config_hash
                packet["source_sha256"] = frozen_source_hash
                write_json(path, packet)
                completed.append(path)
                print(f"[{len(completed)}/{len(tasks)}] {task}", flush=True)
    return config, sorted(completed)


def load_packets(paths: list[Path]):
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _key(packet):
    return packet["method"], packet["geometry"], round(float(packet["sigma"]), 12)


def slope_rows(config: dict, summary_rows: list[dict]):
    output = []
    for method in config["methods"]:
        for geometry in config["geometries"]:
            block = sorted(
                [row for row in summary_rows if row["method"] == method and row["geometry"] == geometry],
                key=lambda row: row["sigma"],
            )
            for window_name, count in config["slope_windows"].items():
                selected = block[: int(count)]
                for estimator, exponent in (("Single", 2), ("Multi", 4)):
                    error_key = "single_E_infinity" if estimator == "Single" else "multi_E_infinity"
                    x = np.log([row["sigma"] for row in selected])
                    y = np.log([row[error_key] for row in selected])
                    slope, intercept = np.polyfit(x, y, 1)
                    predicted = slope * x + intercept
                    ss_res = float(np.sum((y - predicted) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    output.append(
                        {
                            "method": method,
                            "geometry": geometry,
                            "estimator": estimator,
                            "target_order": exponent,
                            "window": window_name,
                            "point_count": len(selected),
                            "sigma_min": min(row["sigma"] for row in selected),
                            "sigma_max": max(row["sigma"] for row in selected),
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
                        }
                    )
    return output


def aggregate(root: Path, config: dict, packet_paths: list[Path]):
    packets = load_packets(packet_paths)
    expected = len(config["methods"]) * len(config["geometries"]) * len(config["sigma_grid"]) * len(config["resolutions"])
    errors = []
    expected_config_hash = config_hash(config)
    expected_source_hash = source_hash(root)
    if len(packets) != expected:
        errors.append(f"packet_count:{len(packets)}!={expected}")
    for packet in packets:
        if packet.get("config_sha256") != expected_config_hash:
            errors.append(f"config_hash:{packet.get('method')}:{packet.get('geometry')}:{packet.get('sigma')}:{packet.get('resolution')}")
        if packet.get("source_sha256") != expected_source_hash:
            errors.append(f"source_hash:{packet.get('method')}:{packet.get('geometry')}:{packet.get('sigma')}:{packet.get('resolution')}")
        if packet.get("backend") != config["audit_gates"]["required_backend"]:
            errors.append(f"backend:{packet.get('method')}:{packet.get('geometry')}:{packet.get('sigma')}:{packet.get('resolution')}:{packet.get('backend')}")
        gpu = packet.get("gpu", {})
        if gpu.get("device_name") != "NVIDIA L40" or gpu.get("float_precision") != "float64":
            errors.append(f"gpu_metadata:{packet.get('method')}:{packet.get('geometry')}:{packet.get('sigma')}:{packet.get('resolution')}")
    gate_path = root / "results" / "gpu_equivalence_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_valid = (
            gate.get("status") == "PASS"
            and gate.get("config_sha256") == expected_config_hash
            and gate.get("source_sha256") == expected_source_hash
        )
    else:
        gate = None
        gate_valid = False
        errors.append("gpu_equivalence_gate:missing")
    production = {_key(packet): packet for packet in packets if packet["resolution"] == "production"}
    reference = {_key(packet): packet for packet in packets if packet["resolution"] == "reference"}
    summary = []
    convergence = []
    weight_residuals = []
    root_uncertainties = []
    for method in config["methods"]:
        for geometry in config["geometries"]:
            for sigma in config["sigma_grid"]:
                key = (method, geometry, round(float(sigma), 12))
                prod = production.get(key)
                ref = reference.get(key)
                if prod is None or ref is None:
                    errors.append(f"missing:{key}")
                    continue
                expected_queries = int(config["geometries"][geometry]["query_count"])
                for label, packet in (("production", prod), ("reference", ref)):
                    if packet["query_count"] != expected_queries:
                        errors.append(f"query_count:{label}:{key}")
                    if packet["failure_count"]:
                        errors.append(f"failures:{label}:{key}:{packet['failure_count']}")
                weight_residuals.append(max(abs(value) for value in prod["weights"]["constraint_residual"]))
                for query in prod["query_rows"]:
                    for fit in query["fits"] + ([query["single_fit"]] if query.get("single_fit") else []):
                        root_uncertainties.append(float(fit["final"]["coordinate_correction"]))
                prod_single = np.asarray(prod["single_points"], dtype=float)
                prod_multi = np.asarray(prod["multi_points"], dtype=float)
                ref_single = np.asarray(ref["single_points"], dtype=float)
                ref_multi = np.asarray(ref["multi_points"], dtype=float)
                single_diff = np.linalg.norm(prod_single - ref_single, axis=1)
                multi_diff = np.linalg.norm(prod_multi - ref_multi, axis=1)
                convergence.append(
                    {
                        "method": method,
                        "geometry": geometry,
                        "sigma": float(sigma),
                        "single_point_difference_max": float(np.max(single_diff)),
                        "multi_point_difference_max": float(np.max(multi_diff)),
                        "production_single_E_infinity": prod["metrics"]["single_E_infinity"],
                        "reference_single_E_infinity": ref["metrics"]["single_E_infinity"],
                        "production_multi_E_infinity": prod["metrics"]["multi_E_infinity"],
                        "reference_multi_E_infinity": ref["metrics"]["multi_E_infinity"],
                    }
                )
                summary.append(
                    {
                        "method": method,
                        "geometry": geometry,
                        "sigma": float(sigma),
                        "h1": prod["h1"],
                        "h2": prod["h2"],
                        "query_count": prod["query_count"],
                        "single_E_infinity": prod["metrics"]["single_E_infinity"],
                        "single_E_average": prod["metrics"]["single_E_average"],
                        "multi_E_infinity": prod["metrics"]["multi_E_infinity"],
                        "multi_E_average": prod["metrics"]["multi_E_average"],
                        "single_scaled_sigma2": prod["metrics"]["single_E_infinity"] / float(sigma) ** 2,
                        "multi_scaled_sigma4": prod["metrics"]["multi_E_infinity"] / float(sigma) ** 4,
                    }
                )
    slopes = slope_rows(config, summary) if not errors else []
    min_multi = min(row["multi_E_infinity"] for row in summary) if summary else float("nan")
    max_resolution_difference = max(
        max(row["single_point_difference_max"], row["multi_point_difference_max"])
        for row in convergence
    ) if convergence else float("inf")
    max_root_uncertainty = max(root_uncertainties, default=float("inf"))
    gates = config["audit_gates"]
    checks = {
        "all_declared_packets_present": len(packets) == expected and not any(item.startswith("missing:") for item in errors),
        "packet_config_and_source_hashes": not any(item.startswith(("config_hash:", "source_hash:")) for item in errors),
        "formal_backend_is_float64_NVIDIA_L40": not any(item.startswith(("backend:", "gpu_metadata:")) for item in errors),
        "cpu_gpu_equivalence_gate": gate_valid,
        "all_declared_cells_and_queries_resolved": not any(item.startswith("failures:") for item in errors),
        "weight_constraints": max(weight_residuals, default=float("inf")) <= gates["weight_constraint_max"],
        "resolution_error_below_fraction_of_minimum_multi_error": max_resolution_difference <= gates["reference_point_difference_fraction_of_min_multi_error"] * min_multi,
        "root_uncertainty_below_fraction_of_minimum_multi_error": max_root_uncertainty <= gates["root_position_uncertainty_fraction_of_min_multi_error"] * min_multi,
    }
    audit = {
        "schema_version": "E1_THEORY_PATH_AUDIT_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "packet_count": len(packets),
        "expected_packet_count": expected,
        "config_sha256": expected_config_hash,
        "source_sha256": expected_source_hash,
        "gpu_equivalence_gate": gate,
        "numerical_maxima": {
            "weight_constraint_residual": max(weight_residuals, default=float("nan")),
            "root_position_uncertainty": max_root_uncertainty,
            "production_reference_point_difference": max_resolution_difference,
            "minimum_production_multi_E_infinity": min_multi,
            "resolution_difference_over_minimum_multi_error": max_resolution_difference / min_multi,
            "root_uncertainty_over_minimum_multi_error": max_root_uncertainty / min_multi,
        },
        "population_integration": {
            "STSP": "periodic trapezoid on circle/torus and Gauss-Jacobi latitude with exact signed-axis first/second angular moments on spheres; Gaussian noise reduced analytically to noncentral-chi radial moments and integrated by Gauss-Legendre quadrature",
            "DR": "the same deterministic manifold quadratures; Gaussian measurement noise and Gaussian kernel convolution integrated analytically for the ridge residual; matching PCA uses the STSP compact-polynomial noncentral-chi moment integration at h1",
            "monte_carlo": False,
        },
    }
    results = root / "results"
    _write_csv(results / "theory_path_errors.csv", summary)
    _write_csv(results / "slope_stability.csv", slopes)
    _write_csv(results / "numerical_convergence.csv", convergence)
    write_json(results / "AUDIT.json", audit)
    return audit, summary, slopes, convergence


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
