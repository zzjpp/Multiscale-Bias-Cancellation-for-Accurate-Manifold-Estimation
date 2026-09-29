from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .coefficients import dr_coefficient, stsp_coefficient, two_scale_weights, yy_weights
from .geometry import GEOMETRIES
from .population import fit_pair
from .yy_population import frozen_outputs as yy_frozen_outputs, geometric_sensitivity as yy_geometric_sensitivity


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default)+"\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coefficient_rows(method, geometry, config):
    sigma = config["true_sigma"]
    h1, h2 = config[method.lower()]["h1"], config[method.lower()]["h2"]
    if method == "DR":
        coefficient = lambda h, s: dr_coefficient(geometry.d, h, s)
    else:
        p = config["stsp"]["kernel_power"]
        coefficient = lambda h, s: stsp_coefficient(geometry.d, geometry.D, h, s, p)
    true_c = np.array([coefficient(h1, sigma), coefficient(h2, sigma)])
    true_w = two_scale_weights(true_c)
    rows = []
    for delta in config["deltas"]:
        sigma_hat = (1.0+delta)*sigma
        assumed_c = np.array([coefficient(h1, sigma_hat), coefficient(h2, sigma_hat)])
        weights = two_scale_weights(assumed_c)
        rows.append({
            "delta": delta,
            "sigma_hat": sigma_hat,
            "true_coefficients": true_c,
            "assumed_coefficients": assumed_c,
            "weights": weights,
            "weight_sum_residual": float(abs(weights.sum()-1.0)),
            "assumed_cancellation_residual": float(abs(weights@assumed_c)),
            "true_cancellation_residual": float(abs(weights@true_c)),
            "true_weights": true_w,
        })
    return rows


def _geometric_rows(method, geometry, coefficient_rows, config, resolution_name):
    sigma = config["true_sigma"]
    section = config[method.lower()]
    queries = geometry.queries(config["query_count"], config["query_phase"])
    fits = fit_pair(
        geometry, method, section["h1"], section["h2"], sigma, queries,
        config["population_resolution"][resolution_name], config["stsp"]["kernel_power"]
    )
    if np.any(fits["failure_mask"]):
        raise RuntimeError(f"{method}/{geometry.key}/{resolution_name}: {sum(fits['failure_mask'])} failed queries")
    true_weights = np.asarray(coefficient_rows[config["deltas"].index(0.0)]["true_weights"])
    correct = true_weights[0]*fits["fine_points"]+true_weights[1]*fits["coarse_points"]
    rows = []
    for coefficient_row in coefficient_rows:
        weights = np.asarray(coefficient_row["weights"])
        estimate = weights[0]*fits["fine_points"]+weights[1]*fits["coarse_points"]
        distances = np.linalg.norm(estimate-correct, axis=1)
        rows.append({
            "delta": coefficient_row["delta"],
            "delta_infinity": float(np.max(distances)),
            "delta_average": float(np.mean(distances)),
            "fit_digest": fits["fit_digest"],
            "query_count": len(queries),
            "resolved_queries": int(np.sum(~fits["failure_mask"])),
        })
    return rows, fits


def _yy_rows(geometry, config):
    sigma = config["true_sigma"]
    radii = config["yy"]["circle_radii" if geometry.q == 1 else "curve_radii"]
    p = config["yy"]["kernel_power"]
    true = yy_weights(geometry.d, geometry.q, radii, sigma, p)
    rows = []
    for delta in config["deltas"]:
        sigma_hat = (1.0+delta)*sigma
        assumed = yy_weights(geometry.d, geometry.q, radii, sigma_hat, p)
        axial = float(abs(assumed["axial"]@(1.0+true["alpha"])))
        if geometry.q == 1:
            complementary_alpha = 0.0
            complementary_beta = 0.0
            summary = axial
        else:
            complementary_alpha = float(abs(assumed["complementary"]@true["alpha"]))
            complementary_beta = float(abs(assumed["complementary"]@(1.0-true["beta"])))
            summary = max(axial, complementary_alpha, complementary_beta)
        rows.append({
            "delta": delta,
            "sigma_hat": sigma_hat,
            "radii": radii,
            "true_kappas": true["kappas"],
            "assumed_kappas": assumed["kappas"],
            "axial_residual": axial,
            "complementary_alpha_residual": complementary_alpha,
            "complementary_beta_residual": complementary_beta,
            "maximum_directional_residual": summary,
            "axial_weights": assumed["axial"],
            "complementary_weights": assumed["complementary"],
            "assumed_constraint_residual_max": float(max(
                np.max(abs(assumed["axial_constraint_residual"])),
                np.max(abs(assumed["complementary_constraint_residual"])),
            )),
        })
    return rows


def run(root):
    root = Path(root)
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    results = root/"results"
    results.mkdir(exist_ok=True)
    cache = results/"checkpoint_blocks"
    cache.mkdir(exist_ok=True)
    payload = {"config": config, "methods": {}}
    for method in ("DR", "STSP"):
        payload["methods"][method] = {}
        for geometry in GEOMETRIES:
            checkpoint = cache/f"{method.lower()}_{geometry.key}.json"
            if checkpoint.exists():
                print(f"resume {method}/{geometry.key}", flush=True)
                payload["methods"][method][geometry.key] = json.loads(checkpoint.read_text(encoding="utf-8"))
                continue
            print(f"start {method}/{geometry.key}", flush=True)
            coefficient_rows = _coefficient_rows(method, geometry, config)
            fine_rows, fine_fits = _geometric_rows(method, geometry, coefficient_rows, config, "fine")
            print(f"fine complete {method}/{geometry.key}", flush=True)
            coarse_rows, _ = _geometric_rows(method, geometry, coefficient_rows, config, "coarse")
            for fine, coarse in zip(fine_rows, coarse_rows):
                fine["coarse_delta_infinity"] = coarse["delta_infinity"]
                fine["fine_coarse_abs_difference"] = abs(fine["delta_infinity"]-coarse["delta_infinity"])
            block = {
                "coefficients": coefficient_rows,
                "geometry": fine_rows,
                "failures": fine_fits["failures"],
            }
            payload["methods"][method][geometry.key] = block
            _write_json(checkpoint, block)
            print(f"checkpoint {method}/{geometry.key}", flush=True)
    payload["methods"]["YY"] = {}
    for geometry in GEOMETRIES:
        checkpoint = cache/f"yy_{geometry.key}.json"
        if checkpoint.exists():
            print(f"resume YY/{geometry.key}", flush=True)
            payload["methods"]["YY"][geometry.key] = json.loads(checkpoint.read_text(encoding="utf-8"))
            continue
        print(f"start YY/{geometry.key}", flush=True)
        coefficient_rows = _yy_rows(geometry, config)
        radii = config["yy"]["circle_radii" if geometry.q == 1 else "curve_radii"]
        queries = geometry.queries(config["query_count"], config["query_phase"])
        common = dict(
            geometry=geometry, queries=queries, sigma=config["true_sigma"], radii=radii,
            c0=config["yy"]["first_stage_radius_over_sigma"],
            c2=config["yy"]["longitudinal_radius_constant"],
            p=config["yy"]["kernel_power"],
        )
        fine_fits = yy_frozen_outputs(
            **common, settings=config["population_resolution"]["yy_fine"]
        )
        if np.any(fine_fits["failure_mask"]):
            raise RuntimeError(
                f"YY/{geometry.key}/fine: {sum(fine_fits['failure_mask'])} failed queries"
            )
        fine_rows = yy_geometric_sensitivity(
            geometry, fine_fits, config["true_sigma"], radii,
            config["deltas"], config["yy"]["kernel_power"]
        )
        print(f"fine complete YY/{geometry.key}", flush=True)
        coarse_fits = yy_frozen_outputs(
            **common, settings=config["population_resolution"]["yy_coarse"]
        )
        if np.any(coarse_fits["failure_mask"]):
            raise RuntimeError(
                f"YY/{geometry.key}/coarse: {sum(coarse_fits['failure_mask'])} failed queries"
            )
        coarse_rows = yy_geometric_sensitivity(
            geometry, coarse_fits, config["true_sigma"], radii,
            config["deltas"], config["yy"]["kernel_power"]
        )
        for fine, coarse in zip(fine_rows, coarse_rows):
            fine["coarse_delta_infinity"] = coarse["delta_infinity"]
            fine["fine_coarse_abs_difference"] = abs(
                fine["delta_infinity"]-coarse["delta_infinity"]
            )
        block = {
            "coefficients": coefficient_rows,
            "geometry": fine_rows,
            "failures": fine_fits["failures"],
            "minimum_first_stage_signal_over_sigma": float(
                np.min(fine_fits["signal_ratios"])
            ),
        }
        payload["methods"]["YY"][geometry.key] = block
        _write_json(checkpoint, block)
        print(f"checkpoint YY/{geometry.key}", flush=True)
    output = results/"e6_results.json"
    _write_json(output, payload)
    _write_csv(results/"e6_summary.csv", payload)
    return payload


def _write_csv(path, payload):
    columns = ["method", "geometry", "delta", "sigma_hat", "coefficient_residual", "delta_infinity",
               "delta_average", "axial_residual", "complementary_alpha_residual",
               "complementary_beta_residual", "maximum_directional_residual"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method in ("DR", "STSP"):
            for geometry, block in payload["methods"][method].items():
                for coefficient, metric in zip(block["coefficients"], block["geometry"]):
                    writer.writerow({
                        "method": method, "geometry": geometry, "delta": coefficient["delta"],
                        "sigma_hat": coefficient["sigma_hat"],
                        "coefficient_residual": coefficient["true_cancellation_residual"],
                        "delta_infinity": metric["delta_infinity"], "delta_average": metric["delta_average"],
                    })
        for geometry, block in payload["methods"]["YY"].items():
            for row, metric in zip(block["coefficients"], block["geometry"]):
                writer.writerow({
                    "method": "YY", "geometry": geometry,
                    "delta": row["delta"], "sigma_hat": row["sigma_hat"],
                    "delta_infinity": metric["delta_infinity"],
                    "delta_average": metric["delta_average"],
                    "axial_residual": row["axial_residual"],
                    "complementary_alpha_residual": row["complementary_alpha_residual"],
                    "complementary_beta_residual": row["complementary_beta_residual"],
                    "maximum_directional_residual": row["maximum_directional_residual"],
                })


def audit(root, payload):
    root = Path(root)
    config = payload["config"]
    gates = config["gates"]
    checks = {}
    coefficient_zero, geometry_zero, weight_constraints = [], [], []
    digest_invariance, resolution = [], []
    for method in ("DR", "STSP"):
        for geometry, block in payload["methods"][method].items():
            zero = config["deltas"].index(0.0)
            coefficient_zero.append(block["coefficients"][zero]["true_cancellation_residual"])
            geometry_zero.append(block["geometry"][zero]["delta_infinity"])
            weight_constraints.extend(max(row["weight_sum_residual"], row["assumed_cancellation_residual"])
                                      for row in block["coefficients"])
            digests = {row["fit_digest"] for row in block["geometry"]}
            digest_invariance.append(len(digests) == 1)
            for row in block["geometry"]:
                scale = max(row["delta_infinity"], row["coarse_delta_infinity"], 1e-14)
                resolution.append((row["fine_coarse_abs_difference"], row["fine_coarse_abs_difference"]/scale))
    yy_zero, yy_constraints = [], []
    for block in payload["methods"]["YY"].values():
        zero = config["deltas"].index(0.0)
        rows = block["coefficients"]
        yy_zero.append(rows[zero]["maximum_directional_residual"])
        yy_zero.append(block["geometry"][zero]["delta_infinity"])
        yy_constraints.extend(row["assumed_constraint_residual_max"] for row in rows)
        digests = {row["fit_digest"] for row in block["geometry"]}
        digest_invariance.append(len(digests) == 1)
        for row in block["geometry"]:
            scale = max(row["delta_infinity"], row["coarse_delta_infinity"], 1e-14)
            resolution.append((
                row["fine_coarse_abs_difference"],
                row["fine_coarse_abs_difference"]/scale,
            ))
    checks["weight_constraints"] = max(weight_constraints+yy_constraints) <= gates["weight_constraint_tolerance"]
    checks["exact_zero_at_correct_sigma"] = max(coefficient_zero+geometry_zero+yy_zero) <= gates["delta_zero_tolerance"]
    checks["frozen_fit_digest_invariant_across_delta"] = all(digest_invariance)
    checks["all_256_queries_resolved"] = all(
        all(row["resolved_queries"] == config["query_count"] for row in block["geometry"])
        for method in ("DR", "STSP", "YY") for block in payload["methods"][method].values()
    )
    checks["fine_coarse_resolution"] = all(
        absolute <= gates["fine_coarse_delta_abs_tolerance"] or relative <= gates["fine_coarse_delta_rel_tolerance"]
        for absolute, relative in resolution
    )
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maxima": {
            "weight_constraint_residual": max(weight_constraints+yy_constraints),
            "delta_zero": max(coefficient_zero+geometry_zero+yy_zero),
            "fine_coarse_abs_difference": max(value[0] for value in resolution),
            "fine_coarse_relative_difference": max(value[1] for value in resolution),
        },
        "note": "Resolution gate is prespecified; zero-delta rows use an absolute floor and are not post-hoc removed.",
    }
    _write_json(root/"results"/"AUDIT.json", report)
    return report


def freeze_hashes(root):
    root = Path(root)
    top_level = {
        "config.json", "requirements.txt", "reproduce.bat", "run_experiment.py",
        "README.md", "RESULTS_REPORT.md", "THEORY_CODE_CONTRACT.md",
    }
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and (
            path.relative_to(root).parts[0] in ("src", "tests")
            or path.relative_to(root).as_posix() in top_level
        )
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root/"SOURCE_SHA256.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
