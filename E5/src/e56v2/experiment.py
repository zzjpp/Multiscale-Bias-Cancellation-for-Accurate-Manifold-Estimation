"""E5-only cells and packet production for the two released figures."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import directional_weights
from .e5_ablation import METHODS, combine_common_start, component_metrics
from .e5_empirical import (
    EmpiricalQueryFailure,
    empirical_common_start_query,
    retained_fill_distance,
    sample_noisy_curve,
)
from .e5_population import common_start_population_query
from .geometry_e5 import arclength_theta, gamma4, preliminary_point
from .moments import directional_moments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def weights_for(config: dict) -> dict:
    moments = directional_moments(config["kappas"], d=1, p=6, m=2)
    result = directional_weights(
        [row.alpha for row in moments], [row.beta for row in moments]
    )
    return {
        "a": np.asarray(result["a"], dtype=float),
        "b": np.asarray(result["b"], dtype=float),
        "W": float(result["W"]),
        "max_abs_residual": float(result["max_abs_residual"]),
        "axial_condition": float(result["axial_condition"]),
        "complementary_condition": float(result["complementary_condition"]),
        "moments": [row.__dict__ for row in moments],
    }


def finite_conditioning_gate(config: dict, weights: dict) -> dict:
    specification = config["conditioning_gate"]
    observed = float(weights["complementary_condition"])
    residual = float(weights["max_abs_residual"])
    threshold = float(specification["threshold"])
    residual_tolerance = float(specification["weight_residual_tolerance"])
    return {
        "name": specification["name"],
        "observed": observed,
        "threshold": threshold,
        "weight_residual": residual,
        "weight_residual_tolerance": residual_tolerance,
        "pass": bool(
            np.isfinite(observed) and observed <= threshold
            and np.isfinite(residual) and residual <= residual_tolerance
        ),
    }


def summarize(rows: dict[str, list[dict]], fill_distance: float | None) -> dict:
    output = {}
    for method in METHODS:
        values = rows[method]
        item = {"retained_count": len(values)}
        for metric in ("geometric", "axial_normal", "complementary_normal", "tangential"):
            series = np.asarray([row[metric] for row in values], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(series)) if len(series) else None
            item[f"{metric}_max"] = float(np.max(series)) if len(series) else None
        item["finite_cloud_hausdorff_upper"] = (
            max(float(fill_distance), item["geometric_max"])
            if values and fill_distance is not None
            else None
        )
        output[method] = item
    return output


def population_cell(config: dict, sigma: float, query_count: int | None = None,
                    numerical: dict | None = None) -> dict:
    query_count = int(config["query_count"] if query_count is None else query_count)
    numerical = dict(config["numerical"] if numerical is None else numerical)
    weights = weights_for(config)
    theta_grid = arclength_theta((np.arange(query_count) + 0.37) / query_count)
    metric_rows = {name: [] for name in METHODS}
    query_rows, retained_truth, failures = [], [], []
    for index, theta in enumerate(theta_grid):
        try:
            query = common_start_population_query(
                float(theta), float(sigma), config["kappas"],
                c0=float(config["c0"]), c2=float(config["c2"]),
                p=int(config["kernel"]["p"]), **numerical,
            )
            combined = combine_common_start(
                query, weights["a"], weights["b"],
                single_kappa=float(config["single_kappa"]),
            )
            per_method = {}
            for method in METHODS:
                metrics = component_metrics(float(theta), combined[method], query["axis"])
                metric_rows[method].append(metrics)
                per_method[method] = metrics
            retained_truth.append(gamma4(float(theta)))
            query_rows.append({
                "query_index": index, "theta": float(theta), "status": "success",
                "axis_digest": query["axis_digest"], "first_mass": query["first_mass"],
                "first_signal_over_sigma": query["first_signal_over_sigma"],
                "second_masses": query["second_masses"].tolist(), "metrics": per_method,
            })
        except (ArithmeticError, ValueError) as error:
            failure = {
                "query_index": index, "theta": float(theta),
                "status": "failure", "reason": str(error),
            }
            failures.append(failure)
            query_rows.append(failure)
    fill = retained_fill_distance(np.asarray(retained_truth), 8192) if retained_truth else None
    return {
        "schema": "E5_RELEASE_POPULATION_CELL_V1",
        "formal_evidence": bool(config["formal_evidence"]),
        "sigma": float(sigma), "query_count": query_count, "numerical": numerical,
        "kappas": list(config["kappas"]), "single_kappa": float(config["single_kappa"]),
        "c0": float(config["c0"]), "c2": float(config["c2"]),
        "weight_l1": weights["W"],
        "weight_max_abs_residual": weights["max_abs_residual"],
        "retained_count": len(retained_truth), "failure_count": len(failures),
        "failures": failures, "retained_fill_distance": fill,
        "summaries": summarize(metric_rows, fill), "query_rows": query_rows,
    }


def finite_cell(config: dict, sample_size: int, repeat: int, seed: int,
                query_count: int | None = None) -> dict:
    query_count = int(config["query_count"] if query_count is None else query_count)
    sigma = float(config["sigma"])
    weights = weights_for(config)
    conditioning_gate = finite_conditioning_gate(config, weights)
    if not conditioning_gate["pass"]:
        raise ArithmeticError("finite-sample directional weights fail the conditioning gate")
    sample = sample_noisy_curve(
        int(sample_size), sigma, np.random.default_rng(int(seed))
    )
    observations = sample["observations"]
    observation_digest = hashlib.sha256(
        np.ascontiguousarray(observations).tobytes()
    ).hexdigest()
    theta_grid = arclength_theta((np.arange(query_count) + 0.37) / query_count)
    preliminary_digest = hashlib.sha256(
        np.ascontiguousarray(preliminary_point(theta_grid, sigma)).tobytes()
    ).hexdigest()
    metric_rows = {name: [] for name in METHODS}
    retained_truth, query_rows, failures = [], [], []
    minimum_first_ess = np.inf
    minimum_second_ess = np.inf
    for index, theta in enumerate(theta_grid):
        try:
            query = empirical_common_start_query(
                observations, float(theta), sigma, config["kappas"],
                c0=float(config["c0"]), c2=float(config["c2"]),
                p=int(config["kernel"]["p"]),
                minimum_effective_count=float(config["minimum_effective_count"]),
                minimum_signal_over_sigma=float(config["minimum_signal_over_sigma"]),
            )
            combined = combine_common_start(
                query, weights["a"], weights["b"],
                single_kappa=float(config["single_kappa"]),
            )
            per_method = {}
            for method in METHODS:
                metrics = component_metrics(float(theta), combined[method], query["axis"])
                metric_rows[method].append(metrics)
                per_method[method] = metrics
            retained_truth.append(gamma4(float(theta)))
            minimum_first_ess = min(minimum_first_ess, query["first_effective_count"])
            minimum_second_ess = min(
                minimum_second_ess, float(np.min(query["second_effective_counts"]))
            )
            query_rows.append({
                "query_index": index, "status": "success",
                "axis_digest": query["axis_digest"],
                "first_mass": float(query["first_mass"]),
                "first_effective_count": float(query["first_effective_count"]),
                "second_masses": query["second_masses"].tolist(),
                "second_effective_counts": query["second_effective_counts"].tolist(),
                "metrics": per_method,
            })
        except (ArithmeticError, EmpiricalQueryFailure, ValueError) as error:
            failure = {"query_index": index, "status": "failure", "reason": str(error)}
            failures.append(failure)
            query_rows.append(failure)
    fill = retained_fill_distance(np.asarray(retained_truth), 8192) if retained_truth else None
    return {
        "schema": "E5_RELEASE_FINITE_PAPER_PROTOCOL_CELL_V2",
        "formal_evidence": bool(config["formal_evidence"]),
        "sigma": sigma, "sample_size": int(sample_size), "repeat": int(repeat),
        "seed": int(seed), "query_count": query_count,
        "kappas": list(config["kappas"]),
        "single_kappa": float(config["single_kappa"]),
        "c0": float(config["c0"]), "c2": float(config["c2"]),
        "observation_digest": observation_digest,
        "fixed_analytic_preliminary_manifold": True,
        "preliminary_digest": preliminary_digest,
        "common_start_for_all_scales": True,
        "weight_l1": weights["W"],
        "weight_max_abs_residual": weights["max_abs_residual"],
        "minimum_first_effective_count": (
            None if not np.isfinite(minimum_first_ess) else float(minimum_first_ess)
        ),
        "minimum_second_effective_count": (
            None if not np.isfinite(minimum_second_ess) else float(minimum_second_ess)
        ),
        "retained_count": len(retained_truth), "failure_count": len(failures),
        "failures": failures, "retained_fill_distance": fill,
        "summaries": summarize(metric_rows, fill), "query_rows": query_rows,
        "provenance": config["provenance"],
        "legacy_failed_gate": config["legacy_failed_gate"],
        "conditioning_gate": conditioning_gate,
    }


def finite_seed(config: dict, sample_size: int, repeat: int) -> int:
    index = config["sample_sizes"].index(int(sample_size))
    return int(config["master_seed"] + 1_000_003 * index + 10_009 * int(repeat))


def _write_packet(path: Path, result: dict, config: dict, source_hash: str) -> str:
    payload = {
        "metadata": {
            "config_sha256": canonical_hash(config),
            "source_sha256": source_hash,
            "formal_evidence": bool(result["formal_evidence"]),
        },
        "result": result,
    }
    atomic_json(path, payload)
    return path.stem


def _finite_task(args: tuple[dict, int, int, str, str]) -> str:
    config, sample_size, repeat, output_text, source_hash = args
    output = Path(output_text)
    path = output / "packets" / f"finite_n{sample_size:07d}_r{repeat:03d}.json"
    if path.exists():
        return path.stem
    result = finite_cell(
        config, sample_size, repeat, finite_seed(config, sample_size, repeat)
    )
    return _write_packet(path, result, config, source_hash)


def run_formal(root: Path, population_config: dict, finite_config: dict,
               source_hash: str, workers: int) -> dict:
    population_output = root / "population"
    finite_output = root / "finite_sample"
    population_ids = []
    for sigma in population_config["sigma_grid"]:
        path = population_output / "packets" / f"population_sigma_{sigma:.6f}.json"
        if path.exists():
            population_ids.append(path.stem)
            continue
        population_ids.append(_write_packet(
            path, population_cell(population_config, sigma), population_config, source_hash
        ))
    tasks = [
        (finite_config, int(sample_size), repeat, str(finite_output), source_hash)
        for sample_size in finite_config["sample_sizes"]
        for repeat in range(int(finite_config["repeats"]))
    ]
    finite_ids, failures = [], []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(_finite_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                finite_ids.append(future.result())
            except Exception as error:
                failures.append({
                    "sample_size": task[1], "repeat": task[2], "error": repr(error)
                })
    report = {
        "schema": "E5_TWO_FIGURE_RUN_V1",
        "status": "PASS" if not failures and len(finite_ids) == len(tasks) else "FAIL",
        "population_completed": len(population_ids),
        "finite_completed": len(finite_ids),
        "failures": failures,
    }
    atomic_json(root / "RUN_SUMMARY.json", report)
    return report


def audit(root: Path, population_config: dict, finite_config: dict,
          source_hash: str) -> dict:
    errors, hashes = [], {}
    expected_finite_gate = finite_conditioning_gate(
        finite_config, weights_for(finite_config)
    )
    specifications = (
        (
            "population", population_config,
            sorted((root / "population" / "packets").glob("population_sigma_*.json")),
            len(population_config["sigma_grid"]), True,
        ),
        (
            "finite_sample", finite_config,
            sorted((root / "finite_sample" / "packets").glob("finite_n*_r*.json")),
            len(finite_config["sample_sizes"]) * int(finite_config["repeats"]),
            bool(finite_config["formal_evidence"]),
        ),
    )
    failure_taxonomy = Counter()
    for family, config, paths, expected, formal in specifications:
        if len(paths) != expected:
            errors.append(f"{family}: expected {expected} packets, found {len(paths)}")
        for path in paths:
            hashes[f"{family}/{path.name}"] = sha256_file(path)
            packet = json.loads(path.read_text(encoding="utf-8"))
            metadata, result = packet.get("metadata", {}), packet.get("result", {})
            if metadata.get("config_sha256") != canonical_hash(config):
                errors.append(f"{path.name}: config hash mismatch")
            if metadata.get("source_sha256") != source_hash:
                errors.append(f"{path.name}: source hash mismatch")
            if bool(result.get("formal_evidence")) != formal:
                errors.append(f"{path.name}: evidence class mismatch")
            if bool(metadata.get("formal_evidence")) != formal:
                errors.append(f"{path.name}: metadata evidence class mismatch")
            if result.get("retained_count", 0) + result.get("failure_count", 0) != result.get("query_count"):
                errors.append(f"{path.name}: query accounting mismatch")
            if set(result.get("summaries", {})) != set(METHODS):
                errors.append(f"{path.name}: method set mismatch")
            for failure in result.get("failures", []):
                failure_taxonomy[failure.get("reason", "unknown")] += 1
            if family == "finite_sample" and result.get("failure_count") != 0:
                errors.append(f"{path.name}: source figure requires a complete 256-query grid")
            if family == "finite_sample":
                if (not expected_finite_gate["pass"]
                        or result.get("conditioning_gate") != expected_finite_gate):
                    errors.append(f"{path.name}: conditioning gate mismatch")
                if result.get("legacy_failed_gate") != config["legacy_failed_gate"]:
                    errors.append(f"{path.name}: legacy gate provenance mismatch")
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    report = {
        "schema": "E5_TWO_FIGURE_AUDIT_V1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "packet_count": len(hashes),
        "packet_set_sha256": digest.hexdigest(),
        "failure_taxonomy": dict(failure_taxonomy),
        "finite_figure_formal_evidence": bool(finite_config["formal_evidence"]),
        "finite_gate_revision": finite_config["conditioning_gate"]["revision"],
        "population_figure_formal_evidence": True,
    }
    atomic_json(root / "AUDIT.json", report)
    return report
