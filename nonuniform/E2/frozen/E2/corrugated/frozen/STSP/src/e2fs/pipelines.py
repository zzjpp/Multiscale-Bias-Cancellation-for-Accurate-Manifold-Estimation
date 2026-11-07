from __future__ import annotations

import time
import uuid
import numpy as np
from scipy.spatial import cKDTree

from .operator import EmpiricalSTSP
from .records import OperatorFailure
from .solver import solve_fiber


def _failure(dimension: int, reason: str, stage: str, **diagnostics) -> dict:
    return {"status": "failed", "failure_stage": stage, "failure_reason": reason, "point": np.full(dimension, np.nan), "iterations": diagnostics.pop("iterations", 0), "diagnostics": diagnostics}


def _match(result, dimension: int, stage: str) -> dict:
    if result.status != "success":
        return _failure(dimension, result.status, stage, iterations=result.iterations, **result.diagnostics)
    return {"status": "success", "failure_stage": None, "failure_reason": None, "point": result.point, "iterations": result.iterations, "diagnostics": result.diagnostics, "trace": result.trace, "fiber_coordinate": result.fiber_coordinate, "fiber_basis": result.basis, "normal_space_source": "PCA"}


def estimate_normal(operator: EmpiricalSTSP, point: np.ndarray) -> tuple[np.ndarray, dict]:
    """ENF-only API: deliberately accepts no true-normal argument."""
    state = operator.evaluate(point)
    basis = state.eigenvectors[:, : operator.geometry.q]
    diagnostic = {"estimated_normal_dimension": int(basis.shape[1]), "ambient_dimension": int(basis.shape[0]), "support_count": state.support_count, "effective_support": state.effective_support, "normalized_mass": state.normalized_mass, "eigengap": state.normalized_eigengap}
    return basis, diagnostic


def run_single_path(data, geometry, h1: float, h2: float, queries: np.ndarray) -> dict:
    pipeline_id = "single-" + uuid.uuid4().hex
    total_start = time.perf_counter()
    start = time.perf_counter()
    tree = cKDTree(np.asarray(data))
    fine = EmpiricalSTSP(data, geometry, h1, tree)
    target = EmpiricalSTSP(data, geometry, h2, tree)
    index_time = time.perf_counter() - start
    reference_time = 0.0
    target_time = 0.0
    rows = []
    for query_id, x in enumerate(np.asarray(queries)):
        start = time.perf_counter()
        reference = solve_fiber(fine, x, None, h2)
        reference_time += time.perf_counter() - start
        reference_row = _match(reference, geometry.D, "reference")
        row = {"query_id": int(query_id), "query": x, "reference": reference_row, "Single_h1": reference_row, "Single_h2": _failure(geometry.D, "reference_branch_failure", "target", root_reason=reference.status)}
        if reference.status == "success":
            z = reference.point
            row["reference_branch_displacement"] = float(np.linalg.norm(z - x))
            row["projection_consistency"] = float(np.linalg.norm(geometry.project(z) - x))
            start = time.perf_counter()
            target_result = solve_fiber(target, z, None, h2, pca_operator=fine)
            target_time += time.perf_counter() - start
            row["Single_h2"] = _match(target_result, geometry.D, "target")
        rows.append(row)
    return {"pipeline": "Single", "pipeline_id": pipeline_id, "state_isolation": "independent index/operators/reference/target", "queries": rows, "timing": {"index": index_time, "reference": reference_time, "target_root": target_time, "pca": 0.0, "combination": 0.0, "total": time.perf_counter() - total_start}}


def run_multi_path(data, geometry, h1: float, h2: float, queries: np.ndarray, weights: dict) -> dict:
    pipeline_id = "multi-" + uuid.uuid4().hex
    total_start = time.perf_counter()
    start = time.perf_counter()
    tree = cKDTree(np.asarray(data))
    fine = EmpiricalSTSP(data, geometry, h1, tree)
    target = EmpiricalSTSP(data, geometry, h2, tree)
    index_time = time.perf_counter() - start
    reference_time = 0.0
    pca_time = 0.0
    target_time = 0.0
    combination_time = 0.0
    rows = []
    for query_id, x in enumerate(np.asarray(queries)):
        start = time.perf_counter()
        reference = solve_fiber(fine, x, None, h2)
        reference_time += time.perf_counter() - start
        row = {"query_id": int(query_id), "query": x, "reference": _match(reference, geometry.D, "reference"), "ENF_target": _failure(geometry.D, "reference_branch_failure", "target", root_reason=reference.status), "ENF_Multi": _failure(geometry.D, "reference_branch_failure", "combination", root_reason=reference.status)}
        if reference.status == "success":
            z = reference.point
            row["reference_branch_displacement"] = float(np.linalg.norm(z - x))
            row["projection_consistency"] = float(np.linalg.norm(geometry.project(z) - x))
            try:
                start = time.perf_counter()
                estimated_basis, diagnostic = estimate_normal(fine, z)
                pca_time += time.perf_counter() - start
                row["estimated_normal"] = diagnostic
                start = time.perf_counter()
                target_result = solve_fiber(target, z, estimated_basis, h2)
                target_time += time.perf_counter() - start
                row["ENF_target"] = _match(target_result, geometry.D, "target")
                if target_result.status == "success":
                    start = time.perf_counter()
                    point = z + estimated_basis @ (weights["weights"][1] * target_result.fiber_coordinate)
                    row["normal_guided"] = {"reference_point": z, "basis": estimated_basis, "pca_radius": h1,
                                            "t1": np.zeros(geometry.q), "t2": target_result.fiber_coordinate,
                                            "normal_space_source": "PCA"}
                    combination_time += time.perf_counter() - start
                    row["ENF_Multi"] = {"status": "success", "failure_stage": None, "failure_reason": None, "point": point, "iterations": target_result.iterations, "diagnostics": {**target_result.diagnostics, **diagnostic}}
                else:
                    row["ENF_Multi"] = _failure(geometry.D, "target_branch_failure", "combination", root_reason=target_result.status, **target_result.diagnostics)
            except OperatorFailure as exc:
                row["ENF_Multi"] = _failure(geometry.D, exc.reason, exc.diagnostics.pop("failure_stage", "pca"), **exc.diagnostics)
        rows.append(row)
    return {"pipeline": "ENF_Multi", "pipeline_id": pipeline_id, "state_isolation": "independent index/operators/reference/PCA/target", "queries": rows, "timing": {"index": index_time, "reference": reference_time, "pca": pca_time, "target_root": target_time, "combination": combination_time, "total": time.perf_counter() - total_start}}


def run_cell(data, geometry, h2: float, queries: np.ndarray, weights: dict) -> dict:
    h1 = h2 / 2
    single = run_single_path(data, geometry, h1, h2, queries)
    multi = run_multi_path(data, geometry, h1, h2, queries, weights)
    if single["pipeline_id"] == multi["pipeline_id"]:
        raise RuntimeError("timing path isolation failed")
    ratio = multi["timing"]["total"] / single["timing"]["total"] if single["timing"]["total"] > 0 else float("nan")
    return {"h1": h1, "h2": h2, "single": single, "multi": multi, "runtime_single": single["timing"]["total"], "runtime_multi": multi["timing"]["total"], "runtime_ratio": ratio}
