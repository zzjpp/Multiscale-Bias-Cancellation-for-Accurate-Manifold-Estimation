"""External E6 protocol/metric audit; never fit, change gates or publish files.

The fitted bases and recorded residuals are checked for consistency, not
independently reproduced. Failed queries are retained even when release fails.
Run in a fresh interpreter, separate from the population experiment packages.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "project/E6"
sys.path.insert(0, str(PACKAGE / "src"))
from e6_misspec import pipeline
from e6_misspec.geometry import GEOMETRIES
from e6_misspec.population import _digest

SOURCE_SHA = "891dc684715676ad0d20b1d9db632a0ef14c7b9c9f704a002e204486ac4a5dae"
CONFIG_SHA = "cc682740d943810a0585c303b80918960244d99815659ad68859b96b58e15fd6"
LAUNCHER_SHA = "25dabf5ff07440d20eb1bd9a1d48d1b8368c24eeb87b81e3ab965f861d4c0ddb"
PREVIOUS_SHA = "51ab641ffb3bd3eb5195a7ce4983552a69a95d7fea7c55a9cc7952b8c3c89b1c"
PREVIOUS_AUDIT_SHA = "a5a2a297df63d6bc5ce615fdca83618ae5c461188b1de58ffe5ea0133c5a82d1"
METHODS = ("DR", "STSP")
RESOLUTIONS = ("fine", "coarse")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
DELTAS = [-.3, -.2, -.1, -.05, 0., .05, .1, .2, .3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True,
                               default=pipeline._json_default)
                    + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(left, right):
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    except (ValueError, TypeError):
        return False
    return a.shape == b.shape and bool(np.allclose(
        a, b, atol=2e-14, rtol=2e-12, equal_nan=True))


def same_value(left, right):
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right) and all(
            same_value(left[k], right[k]) for k in left)
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) and all(
            same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int, np.number)) and not isinstance(left, bool):
        return close(left, right)
    return left == right


def expected_source_manifest(package):
    top = {"config.json", "requirements.txt", "reproduce.bat", "run_experiment.py",
           "README.md", "RESULTS_REPORT.md", "THEORY_CODE_CONTRACT.md"}
    files = sorted((p for p in package.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
                    and (p.relative_to(package).parts[0] in ("src", "tests")
                         or p.relative_to(package).as_posix() in top)),
                   key=lambda p: p.relative_to(package).as_posix())
    # Match the original Linux launcher order, not Windows case-folded sorting.
    return ("\n".join(f"{sha(p)}  {p.relative_to(package).as_posix()}" for p in files)
            + "\n").encode()


def validate_fit(packet, geometry, config):
    """Recompute digest and check all recorded query outcomes/bases/coordinates."""
    errors, taxonomy = [], Counter()
    n, D, q = config["query_count"], geometry.D, geometry.q
    fine = np.asarray(packet.get("fine_points"), dtype=float)
    coarse = np.asarray(packet.get("coarse_points"), dtype=float)
    bases = np.asarray(packet.get("reference_normals"), dtype=float)
    mask_raw = packet.get("failure_mask", [])
    mask = np.asarray(mask_raw, dtype=bool)
    records, failures = packet.get("query_records", []), packet.get("failures", [])
    if (fine.shape != (n, D) or coarse.shape != (n, D)
            or bases.shape != (n, D, q) or mask.shape != (n,)
            or len(records) != n or len(failures) != n):
        return ["query_array_shapes_or_counts"], {}, []
    if not all(type(v) is bool for v in mask_raw):
        errors.append("failure_mask_not_boolean")
    if packet.get("fit_digest") != _digest(fine, coarse, bases, mask):
        errors.append("fit_digest")
    if packet.get("matching_pca_kernel") != "K(t)=(1-t)_+^6":
        errors.append("matching_kernel")
    h1 = config["dr"]["h1"]
    if packet.get("reference_pca_radius") != h1:
        errors.append("reference_pca_radius")
    for i, row in enumerate(records):
        if type(row.get("query_id")) is not int or row["query_id"] != i:
            errors.append(f"query_id:{i}")
        if row.get("pca_radius") != h1:
            errors.append(f"query_pca_radius:{i}")
        if mask[i]:
            reason = row.get("failure_reason")
            if row.get("status") != "failed" or not isinstance(reason, str) or not reason:
                errors.append(f"failed_query_record:{i}")
            if failures[i] != f"query {i}: {reason}":
                errors.append(f"failure_array_reason:{i}")
            if not (np.isnan(fine[i]).all() and np.isnan(coarse[i]).all()
                    and np.isnan(bases[i]).all()):
                errors.append(f"failed_query_arrays:{i}")
            taxonomy[str(reason)] += 1
            continue
        if row.get("status") != "success" or row.get("failure_reason") is not None or failures[i] != "":
            errors.append(f"successful_query_record:{i}")
        if not (np.isfinite(fine[i]).all() and np.isfinite(coarse[i]).all()
                and np.isfinite(bases[i]).all()):
            errors.append(f"successful_query_arrays:{i}")
            continue
        if (row.get("normal_space_source") != "population_kernel_weighted_PCA"
                or row.get("matching_kernel") != "K(t)=(1-t)_+^6"):
            errors.append(f"query_matching_protocol:{i}")
        E = np.asarray(row.get("reference_basis"), dtype=float)
        t1 = np.asarray(row.get("t1"), dtype=float)
        t2 = np.asarray(row.get("t2"), dtype=float)
        if E.shape != (D, q) or t1.shape != (q,) or t2.shape != (q,):
            errors.append(f"codimension:{i}")
            continue
        if not np.array_equal(E, bases[i]) or not np.array_equal(row.get("reference_point"), fine[i]):
            errors.append(f"embedded_reference_arrays:{i}")
        if not np.allclose(E.T@E, np.eye(q), rtol=0., atol=2e-13):
            errors.append(f"orthonormal_full_normal_basis:{i}")
        if not np.array_equal(t1, np.zeros(q)):
            errors.append(f"t1_not_zero:{i}")
        if not np.isfinite(t2).all() or np.linalg.norm(t2) > 3*config["dr"]["h2"]**2:
            errors.append(f"coarse_coordinate_or_tube:{i}")
        if not np.allclose(coarse[i], fine[i]+E@t2, rtol=0., atol=2e-14):
            errors.append(f"fixed_fiber_formula:{i}")
        residual = row.get("projected_coarse_residual_norm", float("nan"))
        if not np.isfinite(residual) or not 0. <= residual <= 1e-10:
            errors.append(f"recorded_projected_residual:{i}")
    if mask.any():
        errors.append("original_all_queries_success_gate")
    return errors, dict(taxonomy), records


def extrapolated_metrics(packet, coefficients, config):
    fine = np.asarray(packet["fine_points"], dtype=float)
    E = np.asarray(packet["reference_normals"], dtype=float)
    mask = np.asarray(packet["failure_mask"], dtype=bool)
    t2 = np.full((config["query_count"], E.shape[-1]), np.nan)
    for i, row in enumerate(packet["query_records"]):
        if not mask[i]:
            t2[i] = row["t2"]
    zero = config["deltas"].index(0.)
    true_weights = np.asarray(coefficients[zero]["true_weights"])
    correct = fine+np.einsum("ndq,nq->nd", E, true_weights[1]*t2)
    outputs, metrics = [], []
    for coefficient in coefficients:
        weights = np.asarray(coefficient["weights"])
        points = fine+np.einsum("ndq,nq->nd", E, weights[1]*t2)
        distances = np.linalg.norm(points-correct, axis=1)
        valid = distances[~mask]
        outputs.append((points, distances))
        metrics.append({"delta": coefficient["delta"],
                        "delta_infinity": float(valid.max()) if len(valid) else float("nan"),
                        "delta_average": float(valid.mean()) if len(valid) else float("nan"),
                        "query_count": config["query_count"],
                        "resolved_queries": int((~mask).sum()),
                        "fit_digest": packet["fit_digest"],
                        "matching_pca_kernel": packet["matching_pca_kernel"],
                        "reference_pca_radius": packet["reference_pca_radius"]})
    return metrics, outputs


def validate_block(block, packets, method, geometry, config):
    errors = []
    expected_coefficients = pipeline._coefficient_rows(method, geometry, config)
    if not same_value(block.get("coefficients"), expected_coefficients):
        errors.append("coefficient_rows")
    rows = block.get("geometry", [])
    if len(rows) != len(DELTAS) or [r.get("delta") for r in rows] != DELTAS:
        return errors+["full_delta_grid"], [], []
    metrics_by_resolution = {}
    query_rows = []
    for resolution in RESOLUTIONS:
        packet = packets[resolution]
        metrics, outputs = extrapolated_metrics(packet, expected_coefficients, config)
        metrics_by_resolution[resolution] = metrics
        for coefficient, (points, distances) in zip(expected_coefficients, outputs):
            for i, row in enumerate(packet["query_records"]):
                query_rows.append({"method": method, "geometry": geometry.key,
                                   "resolution": resolution, "query_id": i,
                                   "status": row["status"], "failure_reason": row.get("failure_reason"),
                                   "delta": coefficient["delta"], "pca_radius": row["pca_radius"],
                                   "reference_point": json.dumps(packet["fine_points"][i]),
                                   "coarse_point": json.dumps(packet["coarse_points"][i]),
                                   "output_point": json.dumps(points[i].tolist()),
                                   "distance_to_correct_sigma_output": float(distances[i]),
                                   "recorded_coarse_residual": row.get("projected_coarse_residual_norm")})
    for i, row in enumerate(rows):
        fine, coarse = (metrics_by_resolution[r][i] for r in RESOLUTIONS)
        for field, value in fine.items():
            if not same_value(row.get(field), value):
                errors.append(f"embedded_metric:{i}:{field}")
        difference = abs(fine["delta_infinity"]-coarse["delta_infinity"])
        if not close(row.get("coarse_delta_infinity"), coarse["delta_infinity"]):
            errors.append(f"coarse_delta_infinity:{i}")
        if not close(row.get("fine_coarse_abs_difference"), difference):
            errors.append(f"resolution_difference:{i}")
    if block.get("failures") != packets["fine"]["failures"]:
        errors.append("block_failures")
    return errors, metrics_by_resolution, query_rows


def original_gate_report(payload):
    """Evaluate unchanged E6 gates from independently rechecked result rows."""
    config, values = payload["config"], payload["methods"]
    gates = config["gates"]
    zero = config["deltas"].index(0.)
    weights, zeros, digests, resolved, convergence = [], [], [], [], []
    for method, blocks in values.items():
        for block in blocks.values():
            coeffs, rows = block["coefficients"], block["geometry"]
            if method == "YY":
                weights.extend(r["assumed_constraint_residual_max"] for r in coeffs)
                zeros.append(coeffs[zero]["maximum_directional_residual"])
            else:
                weights.extend(max(r["weight_sum_residual"], r["assumed_cancellation_residual"]) for r in coeffs)
                zeros.append(coeffs[zero]["true_cancellation_residual"])
            zeros.append(rows[zero]["delta_infinity"])
            digests.append(len({r["fit_digest"] for r in rows}) == 1)
            for row in rows:
                resolved.append(row["resolved_queries"] == config["query_count"])
                absolute = abs(row["delta_infinity"]-row["coarse_delta_infinity"])
                relative = absolute/max(row["delta_infinity"], row["coarse_delta_infinity"], 1e-14)
                convergence.append((absolute, relative))
    numeric = np.asarray(weights+zeros+[v for pair in convergence for v in pair])
    checks = {
        "weight_constraints": bool(np.isfinite(weights).all() and max(weights) <= gates["weight_constraint_tolerance"]),
        "exact_zero_at_correct_sigma": bool(np.isfinite(zeros).all() and max(zeros) <= gates["delta_zero_tolerance"]),
        "frozen_fit_digest_invariant_across_delta": all(digests),
        "all_256_queries_resolved": all(resolved),
        "fine_coarse_resolution": bool(np.isfinite(numeric).all() and all(
            a <= gates["fine_coarse_delta_abs_tolerance"] or r <= gates["fine_coarse_delta_rel_tolerance"]
            for a, r in convergence)),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "maxima": {"weight_constraint_residual": max(weights), "delta_zero": max(zeros),
                       "fine_coarse_abs_difference": max(v[0] for v in convergence),
                       "fine_coarse_relative_difference": max(v[1] for v in convergence)}}


def validate_provenance(payload):
    errors = []
    provenance = payload.get("pca_revision_provenance", {})
    expected = {"source_manifest_sha256": SOURCE_SHA, "launcher_sha256": LAUNCHER_SHA,
                "matching_kernel": "K(t)=(1-t)_+^6", "pca_radius": "h1",
                "covariance": "centered_kernel_weighted", "normal_api_poisoned": True,
                "DR_STSP_fits_recomputed": True, "YY_recomputed": False}
    for key, value in expected.items():
        if provenance.get(key) != value:
            errors.append(f"provenance:{key}")
    environment = provenance.get("environment", {})
    for key, value in {"python": "3.12.7", "numpy": "2.1.3", "scipy": "1.18.1",
                       "backend": "cpu deterministic quadrature"}.items():
        if environment.get(key) != value:
            errors.append(f"environment:{key}")
    if environment.get("threads") != {key: "1" for key in THREADS}:
        errors.append("environment:single_thread")
    return errors


def summary_rows(payload, previous):
    rows = []
    for method in ("DR", "STSP", "YY"):
        for geometry, block in payload["methods"][method].items():
            old_rows = previous["methods"][method][geometry]["geometry"]
            for i, (coefficient, metric) in enumerate(zip(block["coefficients"], block["geometry"])):
                rows.append({"method": method, "geometry": geometry, "delta": coefficient["delta"],
                             "sigma_hat": coefficient["sigma_hat"],
                             "coefficient_residual": coefficient.get("true_cancellation_residual", ""),
                             "delta_infinity": metric["delta_infinity"],
                             "delta_average": metric["delta_average"],
                             "axial_residual": coefficient.get("axial_residual", ""),
                             "complementary_alpha_residual": coefficient.get("complementary_alpha_residual", ""),
                             "complementary_beta_residual": coefficient.get("complementary_beta_residual", ""),
                             "maximum_directional_residual": coefficient.get("maximum_directional_residual", ""),
                             "old_delta_infinity": old_rows[i]["delta_infinity"],
                             "old_delta_average": old_rows[i]["delta_average"],
                             "change_in_delta_infinity": metric["delta_infinity"]-old_rows[i]["delta_infinity"],
                             "fine_coarse_abs_difference": metric["fine_coarse_abs_difference"],
                             "resolved_queries": metric["resolved_queries"],
                             "fits_recomputed": method != "YY"})
    return rows


def validate_summary_csv(path, expected):
    if not path.exists():
        return ["missing_production_summary_csv"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = ("method", "geometry", "delta", "sigma_hat", "coefficient_residual",
               "delta_infinity", "delta_average", "axial_residual", "complementary_alpha_residual",
               "complementary_beta_residual", "maximum_directional_residual")
    if len(rows) != 54 or any(set(row) != set(columns) for row in rows):
        return ["production_summary_csv_shape"]
    errors = []
    for i, (observed, wanted) in enumerate(zip(rows, expected)):
        for key in columns:
            if key in ("method", "geometry") or wanted[key] == "":
                match = observed[key] == wanted[key]
            else:
                try:
                    match = close(float(observed[key]), wanted[key])
                except (TypeError, ValueError):
                    match = False
            if not match:
                errors.append(f"production_summary_csv:{i}:{key}")
    return errors


def audit(root=ROOT):
    root = Path(root)
    package, output = root/"project/E6", root/"results/E6_release_audit"
    output.mkdir(parents=True, exist_ok=True)
    errors, receipts, taxonomy, query_rows = [], {}, {}, []
    config = read(package/"config.json")
    previous = read(root/"inputs/e6_previous_results.json")
    for name, path, expected in (
        ("config", package/"config.json", CONFIG_SHA),
        ("launcher", root/"run_e6_revision.py", LAUNCHER_SHA),
        ("previous_results", root/"inputs/e6_previous_results.json", PREVIOUS_SHA),
        ("previous_audit", root/"inputs/e6_previous_audit.json", PREVIOUS_AUDIT_SHA)):
        receipts[name] = sha(path)
        if receipts[name] != expected:
            errors.append(f"frozen_hash:{name}")
    source_blob = expected_source_manifest(package)
    if hashlib.sha256(source_blob).hexdigest() != SOURCE_SHA:
        errors.append("frozen_scientific_source")
    if previous["config"] != config or read(root/"inputs/e6_previous_audit.json")["status"] != "PASS":
        errors.append("unchanged_YY_original_audit_or_config")
    complete_query_packets, audited_records = 0, 0
    blocks_by_key = {}
    cache = package/"results/checkpoint_blocks"
    expected_names = {f"{m.lower()}_{g.key}_{r}_queries.json"
                      for m in METHODS for g in GEOMETRIES for r in RESOLUTIONS}
    actual_names = {p.name for p in cache.glob("*_queries.json")}
    if actual_names-expected_names:
        errors.append("unexpected_query_packets")
    for method in METHODS:
        for geometry in GEOMETRIES:
            packets = {}
            for resolution in RESOLUTIONS:
                path = cache/f"{method.lower()}_{geometry.key}_{resolution}_queries.json"
                if not path.exists():
                    continue
                packet = read(path)
                audited_records += len(packet.get("query_records", []))
                receipts[path.relative_to(root).as_posix()] = sha(path)
                e, t, _ = validate_fit(packet, geometry, config)
                label = f"{method}/{geometry.key}/{resolution}"
                errors.extend(f"{label}:{v}" for v in e)
                taxonomy[label] = t
                complete_query_packets += 1
                if not e:
                    packets[resolution] = packet
                else:
                    for row in packet.get("query_records", []):
                        if row.get("status") == "failed":
                            query_rows.append({"method": method, "geometry": geometry.key,
                                               "resolution": resolution, "query_id": row.get("query_id"),
                                               "status": "failed", "failure_reason": row.get("failure_reason"),
                                               "delta": None, "pca_radius": row.get("pca_radius"),
                                               "reference_point": None, "coarse_point": None,
                                               "output_point": None, "distance_to_correct_sigma_output": None,
                                               "recorded_coarse_residual": None})
            block_path = cache/f"{method.lower()}_{geometry.key}.json"
            if len(packets) == 2 and block_path.exists():
                block = read(block_path)
                receipts[block_path.relative_to(root).as_posix()] = sha(block_path)
                e, _, rows = validate_block(block, packets, method, geometry, config)
                errors.extend(f"{method}/{geometry.key}:{v}" for v in e)
                query_rows.extend(rows)
                blocks_by_key[(method, geometry.key)] = block
    final_path = package/"results/e6_results.json"
    is_complete = complete_query_packets == 8 and len(blocks_by_key) == 4 and final_path.exists()
    original_report = None
    if is_complete:
        payload = read(final_path)
        receipts[final_path.relative_to(root).as_posix()] = sha(final_path)
        if payload.get("config") != config or set(payload.get("methods", {})) != {"DR", "STSP", "YY"}:
            errors.append("final_payload_config_or_methods")
        errors.extend(validate_provenance(payload))
        for method in ("DR", "STSP", "YY"):
            if set(payload["methods"][method]) != {g.key for g in GEOMETRIES}:
                errors.append(f"final_geometry_set:{method}")
        for (method, geometry), block in blocks_by_key.items():
            if payload["methods"][method].get(geometry) != block:
                errors.append(f"final_block:{method}/{geometry}")
        for geometry in GEOMETRIES:
            reused = payload["methods"]["YY"][geometry.key]
            expected = dict(previous["methods"]["YY"][geometry.key], unchanged_YY_provenance={
                "previous_results_sha256": PREVIOUS_SHA, "previous_audit_sha256": PREVIOUS_AUDIT_SHA,
                "reason": "YY same-point estimator and weights unchanged; not a PCA correspondence method"})
            if reused != expected or read(cache/f"yy_{geometry.key}.json") != expected:
                errors.append(f"unchanged_YY_lineage:{geometry.key}")
        original_report = original_gate_report(payload)
        if original_report["status"] != "PASS":
            errors.append("original_numerical_gates")
        for name in ("AUDIT.json", "PCA_FIBER_RELEASE_AUDIT.json", "REVISION_SOURCE.json"):
            if not (package/"results"/name).exists():
                errors.append(f"missing_production_receipt:{name}")
        if not sha_or_none(package/"SOURCE_SHA256.txt") == SOURCE_SHA:
            errors.append("production_source_manifest")
        marker_path = package/"results/REVISION_SOURCE.json"
        marker = read(marker_path) if marker_path.exists() else {}
        if marker != {"source_manifest_sha256": SOURCE_SHA, "config_sha256": CONFIG_SHA}:
            errors.append("production_revision_marker")
        audit_path = package/"results/AUDIT.json"
        embedded_audit = read(audit_path) if audit_path.exists() else {}
        if embedded_audit.get("checks") != original_report["checks"] or not same_value(
                embedded_audit.get("maxima"), original_report["maxima"]) or embedded_audit.get("status") != "PASS":
            errors.append("embedded_original_audit")
        fiber_path = package/"results/PCA_FIBER_RELEASE_AUDIT.json"
        fiber = read(fiber_path) if fiber_path.exists() else {}
        if (fiber.get("status") != "PASS" or fiber.get("audited_query_records") != 2048
                or fiber.get("source_manifest_sha256") != SOURCE_SHA
                or fiber.get("matching_pca_protocol") != "PASS"
                or fiber.get("YY_unchanged_source") != PREVIOUS_SHA
                or fiber.get("base_audit") != embedded_audit):
            errors.append("embedded_fiber_audit")
        expected_summary = summary_rows(payload, previous)
        errors.extend(validate_summary_csv(package/"results/e6_summary.csv", expected_summary))
        write_csv(output/"validated_summary_with_previous.csv", expected_summary)
    status = "FAIL" if errors else "PASS" if is_complete else "WAITING_FOR_COMPLETE_PACKETS"
    report = {"schema": "E6_PCA_EXTERNAL_RELEASE_AUDIT_V1", "status": status,
              "complete_query_packet_count": complete_query_packets, "expected_query_packet_count": 8,
              "audited_query_records": audited_records,
              "recomputed_delta_query_rows": sum(row["delta"] is not None for row in query_rows),
              "preserved_failed_query_rows": sum(row["status"] == "failed" for row in query_rows),
              "errors": errors,
              "failure_taxonomy": taxonomy, "original_numerical_gates": original_report,
              "sha256_receipts": receipts, "formal_publication_permitted": False,
              "unchanged_YY": "Old PASS output copied exactly with its old result/audit hashes; not recomputed.",
              "limitations": ["Consistency audit, not independent reproduction of population fits.",
                              "Query generator, count and phase are source/config pinned; input coordinates are not embedded in E6 fit packets.",
                              "Initial/fine root traces and PCA covariance spectra are not stored; their computation is source-enforced.",
                              "Recorded coarse residual is checked, not independently reevaluated by this auditor.",
                              "E6 is deterministic CPU quadrature; no method-level runtime fields are recorded.",
                              "Plots, archive and formal publication remain separate required steps."]}
    write_json(output/"AUDIT.json", report)
    write_csv(output/"per_delta_query_outputs.csv", query_rows)
    return report


def sha_or_none(path):
    return sha(path) if path.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({k: report[k] for k in ("status", "complete_query_packet_count", "errors")}, indent=2))
    raise SystemExit(1 if report["status"] == "FAIL" else 0)
