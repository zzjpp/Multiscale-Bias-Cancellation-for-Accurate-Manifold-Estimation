from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

from .geometry import TAU, arc_table, curve_distances, gamma, normal_basis, normal_projector, project_to_curve, query_grid
from .solver import solve_fiber


def estimate_normal(operator, point):
    """ENF-only API: deliberately cannot accept true Q, basis, curvature, or theta."""
    state = operator.evaluate(point)
    return state.eigenvectors[:, :2], state.Q, {"eigengap": state.normalized_eigengap, "mass": state.normalized_mass, "support_count": state.support_count}


def candidate_positions(M_match: int, phi: float, N_q: int):
    return np.mod(np.arange(M_match, dtype=float) / M_match + 0.37 / N_q + phi / M_match, 1.0)


def largest_uncovered_gap(normalized_arclength):
    u = np.sort(np.mod(np.asarray(normalized_arclength, dtype=float), 1.0))
    if len(u) == 0:
        return 1.0
    return float(np.max(np.diff(np.concatenate((u, [u[0] + 1.0])))))


def build_candidate_library(target_operator, M_match: int, phi: float, N_q: int, table=None):
    table = arc_table() if table is None else table
    u = candidate_positions(M_match, phi, N_q)
    theta = table.invert(u)
    clean = gamma(theta)
    points = np.full((M_match, 3), np.nan)
    status = []
    diagnostics = []
    start = time.perf_counter()
    for i in range(M_match):
        result = solve_fiber(target_operator, clean[i], normal_basis(theta[i]), target_operator.h)
        status.append(result.status)
        diagnostics.append(result.diagnostics)
        if result.status == "success":
            points[i] = result.point
    elapsed = time.perf_counter() - start
    success = np.asarray([s == "success" for s in status])
    return {"M_match": int(M_match), "phi": float(phi), "candidate_normalized_arclength": u, "candidate_theta": theta, "candidate_clean_coordinates": clean, "candidate_fitted_coordinates": points, "candidate_status": status, "candidate_diagnostics": diagnostics, "success_mask": success, "success_count": int(success.sum()), "failure_count": int((~success).sum()), "failure_indices": np.flatnonzero(~success), "largest_uncovered_arclength_gap": largest_uncovered_gap(u[success]), "generation_runtime": elapsed}


def match_nn(reference_points, library):
    mask = library["success_mask"]
    start = time.perf_counter()
    if not np.any(mask):
        return {"points": np.full_like(reference_points, np.nan), "selected_candidate_id": np.full(len(reference_points), -1), "distances": np.full(len(reference_points), np.nan), "duplicate_selection_count": 0, "duplicate_selection_rate": np.nan, "matching_runtime": time.perf_counter() - start, "status": "candidate_library_incomplete"}
    ids = np.flatnonzero(mask)
    tree = cKDTree(library["candidate_fitted_coordinates"][mask])
    distance, local = tree.query(reference_points)
    selected = ids[local]
    unique = len(np.unique(selected))
    duplicate_count = len(selected) - unique
    return {"points": library["candidate_fitted_coordinates"][selected], "selected_candidate_id": selected, "distances": distance, "duplicate_selection_count": int(duplicate_count), "duplicate_selection_rate": float(duplicate_count / len(selected)) if len(selected) else np.nan, "matching_runtime": time.perf_counter() - start, "status": "success"}


def correspondence_decomposition(candidate, target, theta):
    d = np.asarray(candidate) - np.asarray(target)
    Q = normal_projector(theta)
    normal = np.linalg.norm(np.einsum("nij,nj->ni", Q, d), axis=1)
    tangent = np.linalg.norm(d - np.einsum("nij,nj->ni", Q, d), axis=1)
    summarize = lambda a: {"max": float(np.nanmax(a)), "mean": float(np.nanmean(a)), "median": float(np.nanmedian(a))}
    return {"difference": d, "normal_component": normal, "tangent_component": tangent, "normal_summary": summarize(normal), "tangent_summary": summarize(tangent)}


def fitting_metrics(points, coverage_nodes=1024):
    points = np.asarray(points, dtype=float)
    good = np.isfinite(points).all(axis=1)
    failure_count = int((~good).sum())
    if not np.any(good):
        return {"E_infty": np.nan, "E_avg": np.nan, "E_left": np.nan, "H_grid": np.nan, "query_failure_count": failure_count, "query_failure_rate": 1.0, "status": "failed"}
    distances = curve_distances(points[good])
    table = arc_table()
    truth = gamma(table.invert(np.arange(coverage_nodes) / coverage_nodes))
    left = cKDTree(points[good]).query(truth)[0]
    return {"E_infty": float(np.max(distances)), "E_avg": float(np.mean(distances)), "E_left": float(np.max(left)), "H_grid": float(max(np.max(distances), np.max(left))), "point_distances": distances, "query_failure_count": failure_count, "query_failure_rate": float(failure_count / len(points)), "status": "complete" if failure_count == 0 else "partial_without_interpolation"}


def run_main(operator_h1, operator_h2, s: float, weights, N_q=512, M_match=2048, phi=0.37, query_subset=None, table=None, include_metrics=True):
    table = arc_table() if table is None else table
    grid = query_grid(N_q, table)
    if query_subset is not None:
        take = np.asarray(query_subset, dtype=int)
        grid = {key: value[take] for key, value in grid.items()}
    x, theta = grid["coordinates"], grid["theta"]
    count = len(x)
    z = np.full((count, 3), np.nan)
    z_projection = np.full((count, 3), np.nan)
    z_projection_error = np.full(count, np.nan)
    z_residual = np.full(count, np.nan)
    z_displacement = np.full(count, np.nan)
    ZT = np.full((count, 3), np.nan)
    ZE = np.full((count, 3), np.nan)
    Qhat = np.full((count, 3, 3), np.nan)
    E_Q = np.full(count, np.nan)
    statuses = {"reference": [], "TNF": [], "ENF": []}
    diagnostics = {"reference": [], "TNF": [], "ENF": [], "ENF_PCA": []}
    runtime = {"reference": 0.0, "TNF_target": 0.0, "ENF_PCA": 0.0, "ENF_target": 0.0}
    for j in range(count):
        true_basis = normal_basis(theta[j])
        start = time.perf_counter()
        ref = solve_fiber(operator_h1, x[j], true_basis, s / 2)
        runtime["reference"] += time.perf_counter() - start
        statuses["reference"].append(ref.status)
        diagnostics["reference"].append(ref.diagnostics)
        if ref.status != "success":
            statuses["TNF"].append("reference_branch_failure")
            statuses["ENF"].append("reference_branch_failure")
            diagnostics["TNF"].append({"root_reason": ref.status})
            diagnostics["ENF"].append({"root_reason": ref.status})
            diagnostics["ENF_PCA"].append({})
            continue
        z[j] = ref.point
        projection = project_to_curve(z[j])
        z_projection[j] = projection["point"]
        z_projection_error[j] = np.linalg.norm(projection["point"] - x[j])
        z_residual[j] = ref.diagnostics.get("full_residual", np.nan)
        z_displacement[j] = np.linalg.norm(z[j] - x[j])
        start = time.perf_counter()
        tnf = solve_fiber(operator_h2, z[j], true_basis, s)
        runtime["TNF_target"] += time.perf_counter() - start
        statuses["TNF"].append(tnf.status)
        diagnostics["TNF"].append(tnf.diagnostics)
        if tnf.status == "success":
            ZT[j] = tnf.point
        try:
            start = time.perf_counter()
            estimated_basis, qhat, pca_diag = estimate_normal(operator_h1, z[j])
            runtime["ENF_PCA"] += time.perf_counter() - start
            Qhat[j] = qhat
            E_Q[j] = np.linalg.norm(qhat - normal_projector(theta[j]), 2)
            diagnostics["ENF_PCA"].append(pca_diag)
            start = time.perf_counter()
            enf = solve_fiber(operator_h2, z[j], estimated_basis, s)
            runtime["ENF_target"] += time.perf_counter() - start
            statuses["ENF"].append(enf.status)
            diagnostics["ENF"].append(enf.diagnostics)
            if enf.status == "success":
                ZE[j] = enf.point
        except Exception as exc:
            statuses["ENF"].append(getattr(exc, "reason", "other"))
            diagnostics["ENF"].append({"exception": repr(exc)})
            diagnostics["ENF_PCA"].append({})
    valid_reference = np.isfinite(z).all(axis=1)
    library = build_candidate_library(operator_h2, M_match, phi, N_q, table)
    nn = match_nn(z, library) if np.all(valid_reference) else match_nn(z[valid_reference], library)
    ZNN = np.full_like(z, np.nan)
    if np.all(valid_reference):
        ZNN[:] = nn["points"]
    else:
        ZNN[valid_reference] = nn["points"]
        expanded = np.full(count, -1, dtype=int); expanded[valid_reference] = nn["selected_candidate_id"]; nn["selected_candidate_id"] = expanded
        expanded_d = np.full(count, np.nan); expanded_d[valid_reference] = nn["distances"]; nn["distances"] = expanded_d
    w1, w2 = np.asarray(weights)
    yT, yE, yNN = w1 * z + w2 * ZT, w1 * z + w2 * ZE, w1 * z + w2 * ZNN
    valid_e = np.isfinite(ZE).all(axis=1) & np.isfinite(ZT).all(axis=1)
    valid_n = np.isfinite(ZNN).all(axis=1) & np.isfinite(ZT).all(axis=1)
    decomp_e = correspondence_decomposition(ZE[valid_e], ZT[valid_e], theta[valid_e]) if np.any(valid_e) else None
    decomp_n = correspondence_decomposition(ZNN[valid_n], ZT[valid_n], theta[valid_n]) if np.any(valid_n) else None
    direct = {
        "C_E": float(np.max(np.linalg.norm(ZE[valid_e] - ZT[valid_e], axis=1))) if np.any(valid_e) else np.nan,
        "C_NN": float(np.max(np.linalg.norm(ZNN[valid_n] - ZT[valid_n], axis=1))) if np.any(valid_n) else np.nan,
        "Delta_E": float(np.max(np.linalg.norm(yE[valid_e] - yT[valid_e], axis=1))) if np.any(valid_e) else np.nan,
        "Delta_NN": float(np.max(np.linalg.norm(yNN[valid_n] - yT[valid_n], axis=1))) if np.any(valid_n) else np.nan,
        "E_Q": float(np.nanmax(E_Q)) if np.isfinite(E_Q).any() else np.nan,
        "E_Q_query": E_Q,
        "ENF_decomposition": decomp_e,
        "NN_decomposition": decomp_n
    }
    metrics = {}
    if include_metrics:
        metrics = {"Single": fitting_metrics(ZT), "TNF_Multi": fitting_metrics(yT), "ENF_Multi": fitting_metrics(yE), "NN_Multi": fitting_metrics(yNN)}
    runtime.update({"NN_candidate_library": library["generation_runtime"], "NN_lookup": nn["matching_runtime"], "NN_total": library["generation_runtime"] + nn["matching_runtime"]})
    return {"s": float(s), "h1": float(s / 2), "h2": float(s), "sigma": float(s / 4), "N_q_nominal": int(N_q), "query_count": count, "query": grid, "weights": np.asarray(weights), "common_reference": {"x": x, "z": z, "projection": z_projection, "projection_error": z_projection_error, "residual": z_residual, "branch_displacement": z_displacement, "status": statuses["reference"], "diagnostics": diagnostics["reference"]}, "Single": {"Z_T2": ZT, "status": statuses["TNF"]}, "TNF": {"Z_T2": ZT, "y_T": yT, "status": statuses["TNF"], "diagnostics": diagnostics["TNF"]}, "ENF": {"Q_hat": Qhat, "Z_E2": ZE, "y_E": yE, "status": statuses["ENF"], "pca_diagnostics": diagnostics["ENF_PCA"], "target_diagnostics": diagnostics["ENF"]}, "NN": {"Z_NN2": ZNN, "y_NN": yNN, "selected_candidate_id": nn["selected_candidate_id"], "distance_to_reference": nn["distances"], "duplicate_selection_count": nn["duplicate_selection_count"], "duplicate_selection_rate": nn["duplicate_selection_rate"], "library": library}, "direct_diagnostics": direct, "metrics": metrics, "runtime": runtime}


def continuous_nn(reference_points, target_operator, table=None, branch_nodes=256, interpolation_tolerance=1e-7, max_branch_nodes=4096):
    """Continuous fitted-branch NN with mandatory final fresh-root correction.

    Interpolation is only an optimizer accelerator.  Every final point is a
    newly solved h2 residual root at the optimized arclength coordinate.
    """
    table = arc_table() if table is None else table
    initial_nodes = int(branch_nodes)
    refinement_count = 0
    while True:
        u = np.arange(branch_nodes) / branch_nodes
        theta = table.invert(u)
        branch = np.full((branch_nodes, 3), np.nan)
        for i in range(branch_nodes):
            root = solve_fiber(target_operator, gamma(theta[i]), normal_basis(theta[i]), target_operator.h)
            if root.status == "success": branch[i] = root.point
        if not np.isfinite(branch).all():
            return {"status": "candidate_library_incomplete", "points": np.full_like(reference_points, np.nan), "interpolation_verified": False, "final_fresh_root_correction": True}
        spline = CubicSpline(np.concatenate((u, [1.0])), np.vstack((branch, branch[0])), axis=0, bc_type="periodic")
        mid = (u + 0.5 / branch_nodes) % 1.0
        check_ids = np.linspace(0, branch_nodes - 1, min(64, branch_nodes), dtype=int)
        changes = []
        for idx in check_ids:
            th = table.invert(mid[idx])
            root = solve_fiber(target_operator, gamma(th), normal_basis(th), target_operator.h)
            if root.status != "success":
                return {"status": "candidate_branch_failure", "points": np.full_like(reference_points, np.nan), "interpolation_verified": False, "final_fresh_root_correction": True}
            changes.append(np.linalg.norm(spline(mid[idx]) - root.point))
        if max(changes) <= interpolation_tolerance or branch_nodes >= max_branch_nodes:
            break
        branch_nodes *= 2
        refinement_count += 1
    out = np.full_like(reference_points, np.nan)
    interpolation_points = np.full_like(reference_points, np.nan)
    selected_u = np.full(len(reference_points), np.nan)
    final_status = []
    for j, z in enumerate(reference_points):
        coarse = np.sum((spline(u) - z) ** 2, axis=1)
        candidates = np.flatnonzero((coarse <= np.roll(coarse, 1)) & (coarse <= np.roll(coarse, -1)))
        best = None
        for idx in candidates:
            center = u[idx]
            result = minimize_scalar(lambda v: float(np.sum((spline(v % 1.0) - z) ** 2)), bounds=(center - 1 / branch_nodes, center + 1 / branch_nodes), method="bounded", options={"xatol": 1e-13})
            if best is None or result.fun < best.fun:
                best = result
        if best is not None:
            selected_u[j] = best.x % 1.0
            interpolation_points[j] = spline(selected_u[j])
            th = table.invert(selected_u[j])
            corrected = solve_fiber(target_operator, gamma(th), normal_basis(th), target_operator.h)
            final_status.append(corrected.status)
            if corrected.status == "success": out[j] = corrected.point
        else:
            final_status.append("other")
    discrepancies = np.linalg.norm(out - interpolation_points, axis=1)
    success = np.isfinite(out).all(axis=1)
    return {"status": "success" if np.all(success) else "candidate_branch_failure", "points": out, "interpolation_points": interpolation_points, "selected_normalized_arclength": selected_u, "final_root_status": final_status, "final_fresh_root_correction": True, "fresh_root_discrepancy": discrepancies, "max_fresh_root_discrepancy": float(np.nanmax(discrepancies)) if np.any(success) else np.nan, "interpolation_verified": bool(max(changes) <= interpolation_tolerance), "max_interpolation_root_change": float(max(changes)), "interpolation_tolerance": float(interpolation_tolerance), "initial_branch_nodes": initial_nodes, "branch_nodes": int(branch_nodes), "automatic_refinement_count": refinement_count, "objective": "distance_to_fitted_h2_branch"}
