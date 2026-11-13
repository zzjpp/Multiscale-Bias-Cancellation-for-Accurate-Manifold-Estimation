from __future__ import annotations

from dataclasses import dataclass
import hashlib

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from dryy.coefficients import dr_weights, yy_weights
from dryy.dr import DRFailure
from dryy.kernels import effective_count
from dryy.metrics import summarize_points
from dryy.yy import YYFailure


def _float(value) -> float:
    return float(cp.asnumpy(value))


def _array(value) -> np.ndarray:
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _polynomial_cutoff(squared_radius, radius: float, power: int):
    return cp.maximum(1.0 - squared_radius / float(radius) ** 2, 0.0) ** int(power)


def _longitudinal_profile(normalized_radius):
    t = cp.asarray(normalized_radius, dtype=cp.float64)
    out = cp.zeros_like(t)
    out = cp.where(t <= 0.5, 1.0, out)
    mask = (t > 0.5) & (t < 1.0)
    u = 2.0 * t - 1.0
    transition = 1.0 - 35.0 * u**4 + 84.0 * u**5 - 70.0 * u**6 + 20.0 * u**7
    return cp.where(mask, transition, out)


def _effective_count(weights) -> float:
    total = cp.sum(weights)
    square = weights @ weights
    square_value = _float(square)
    return 0.0 if square_value <= 0.0 else _float(total * total / square)


@dataclass
class GPUDRState:
    center: cp.ndarray
    residual: cp.ndarray
    gradient: cp.ndarray
    hessian: cp.ndarray
    eigenvalues: cp.ndarray
    eigenvectors: cp.ndarray
    normal_basis: cp.ndarray
    Q: cp.ndarray
    vectors: cp.ndarray
    normalized_weights: cp.ndarray
    mean: cp.ndarray
    raw_second: cp.ndarray
    mass: float
    support_count: int
    effective_support: float
    scaled_eigengap: float
    scaled_largest_normal: float
    truncation: dict


def _projector_derivative(eigenvalues, eigenvectors, q: int, derivative):
    gap = _float(eigenvalues[q] - eigenvalues[q - 1])
    if not np.isfinite(gap) or gap <= 0.0:
        raise DRFailure("eigengap_failure", raw_gap=gap)
    result = cp.zeros_like(derivative)
    for i in range(q):
        for j in range(q, len(eigenvalues)):
            vi, vj = eigenvectors[:, i], eigenvectors[:, j]
            denominator = eigenvalues[i] - eigenvalues[j]
            factor = (vj @ derivative @ vi) / denominator
            result += factor * (cp.outer(vj, vi) + cp.outer(vi, vj))
    return result


class EmpiricalDRGPU:
    """CuPy implementation algebraically equivalent to ``EmpiricalDR``."""

    def __init__(self, observations, geometry, h: float, controls: dict, tree=None, data_gpu=None):
        self.data_cpu = np.asarray(observations, dtype=float)
        self.data = cp.asarray(self.data_cpu) if data_gpu is None else data_gpu
        self.geometry = geometry
        self.h = float(h)
        self.controls = dict(controls)
        factor = self.controls.get("gaussian_truncation_factor")
        self.truncation_factor = None if factor is None else float(factor)
        self.tree = tree if self.truncation_factor is not None else None
        if self.truncation_factor is not None and self.tree is None:
            self.tree = cKDTree(self.data_cpu)

    def _support(self, center):
        center_gpu = cp.asarray(center, dtype=cp.float64)
        if self.truncation_factor is None:
            return self.data - center_gpu, {"enabled": False, "omitted_count": 0, "relative_mass_bound": 0.0}
        center_cpu = _array(center_gpu)
        radius = self.truncation_factor * self.h
        indices = np.asarray(sorted(self.tree.query_ball_point(center_cpu, radius)), dtype=int)
        if len(indices) == 0:
            raise DRFailure("low_support", support_count=0)
        vectors = self.data[cp.asarray(indices)] - center_gpu
        cutoff_weight = float(np.exp(-0.5 * self.truncation_factor**2))
        return vectors, {
            "enabled": True,
            "factor": self.truncation_factor,
            "radius": radius,
            "omitted_count": int(len(self.data_cpu) - len(indices)),
            "cutoff_weight": cutoff_weight,
        }

    def evaluate(self, center) -> GPUDRState:
        center_gpu = cp.asarray(center, dtype=cp.float64)
        vectors, truncation = self._support(center_gpu)
        squared = cp.sum(vectors * vectors, axis=1)
        weights = cp.exp(-0.5 * squared / self.h**2)
        total_gpu = cp.sum(weights)
        total = _float(total_gpu)
        if not np.isfinite(total) or total <= 0.0:
            raise DRFailure("insufficient_mass", weight_sum=total)
        if truncation["enabled"]:
            omitted_upper = truncation["omitted_count"] * truncation["cutoff_weight"]
            relative = omitted_upper / (total + omitted_upper)
            truncation["omitted_weight_upper"] = float(omitted_upper)
            truncation["relative_mass_bound"] = float(relative)
            limit = float(self.controls["relative_omitted_weight_bound_max"])
            if not np.isfinite(relative) or relative > limit:
                raise DRFailure("truncation_bound_failure", relative_mass_bound=relative, limit=limit)
        normalized = weights / total_gpu
        mean = normalized @ vectors
        raw_second = cp.einsum("n,ni,nj->ij", normalized, vectors, vectors)
        hessian = raw_second / self.h**4 - cp.eye(self.geometry.D, dtype=cp.float64) / self.h**2
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = cp.linalg.eigh(hessian)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        Q = normal_basis @ normal_basis.T
        gradient = mean / self.h**2
        residual = Q @ gradient
        gap = _float((eigenvalues[q] - eigenvalues[q - 1]) * self.h**2)
        largest_normal = _float(eigenvalues[q - 1] * self.h**2)
        effective = _float(total_gpu * total_gpu / (weights @ weights))
        return GPUDRState(
            center=center_gpu.copy(), residual=residual, gradient=gradient, hessian=hessian,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors, normal_basis=normal_basis, Q=Q,
            vectors=vectors, normalized_weights=normalized, mean=mean, raw_second=raw_second,
            mass=total / len(self.data_cpu), support_count=int(len(vectors)), effective_support=effective,
            scaled_eigengap=gap, scaled_largest_normal=largest_normal, truncation=truncation,
        )

    def spectral_gate(self, state: GPUDRState) -> None:
        margin = float(self.controls["normal_negativity_margin"])
        gap_min = float(self.controls["scaled_eigengap_min"])
        if not np.isfinite(state.scaled_largest_normal) or state.scaled_largest_normal > -margin:
            raise DRFailure("normal_negativity_failure", scaled_largest_normal=state.scaled_largest_normal, margin=margin)
        if not np.isfinite(state.scaled_eigengap) or state.scaled_eigengap < gap_min:
            raise DRFailure("eigengap_failure", scaled_eigengap=state.scaled_eigengap, minimum=gap_min)

    def directional_residual(self, state: GPUDRState, direction):
        direction = cp.asarray(direction, dtype=cp.float64)
        projection = state.vectors @ direction
        mean_projection = state.mean @ direction
        weighted_vector = cp.einsum("n,ni,n->i", state.normalized_weights, state.vectors, projection)
        dmean = (weighted_vector - state.mean * mean_projection) / self.h**2 - direction
        weighted_third = cp.einsum(
            "n,ni,nj,n->ij", state.normalized_weights, state.vectors, state.vectors, projection
        )
        draw = (weighted_third - state.raw_second * mean_projection) / self.h**2
        draw -= cp.outer(direction, state.mean) + cp.outer(state.mean, direction)
        dhessian = 0.5 * (draw + draw.T) / self.h**4
        dQ = _projector_derivative(state.eigenvalues, state.eigenvectors, self.geometry.q, dhessian)
        return dQ @ state.gradient + state.Q @ (dmean / self.h**2)

    def fiber_jacobian(self, state: GPUDRState, basis):
        return cp.column_stack(
            [basis.T @ self.directional_residual(state, basis[:, column]) for column in range(basis.shape[1])]
        )


def estimate_pca_normal_gpu(operator: EmpiricalDRGPU, point, state=None):
    """Centered K(t)=(1-t)_+^6 weighted PCA at the radius h, not ridge space."""
    del state
    vectors = operator.data - cp.asarray(point, dtype=cp.float64)
    squared = cp.sum(vectors * vectors, axis=1)
    active = squared <= operator.h**2
    count = int(cp.count_nonzero(active).item())
    if count < 2:
        raise DRFailure("low_support", failure_stage="pca", support_count=count, pca_radius=operator.h)
    vectors, squared = vectors[active], squared[active]
    weights = cp.maximum(1.0 - squared / operator.h**2, 0.0)**6
    weight_sum = _float(cp.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise DRFailure("insufficient_mass", failure_stage="pca",
                        support_count=count, pca_radius=operator.h)
    weights /= weight_sum
    mean = weights @ vectors
    centered = vectors - mean
    covariance = cp.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
    q = operator.geometry.q
    gap = _float((eigenvalues[q] - eigenvalues[q - 1]) / operator.h**2)
    minimum = float(operator.controls["scaled_eigengap_min"])
    if not np.isfinite(gap) or gap < minimum:
        raise DRFailure("eigengap_failure", failure_stage="pca",
                        normalized_pca_eigengap=gap, minimum=minimum, pca_radius=operator.h)
    return eigenvectors[:, :q].copy()


def solve_fiber_gpu(operator: EmpiricalDRGPU, origin, basis, search_h: float, solver: dict) -> dict:
    origin = cp.asarray(origin, dtype=cp.float64)
    basis = cp.asarray(basis, dtype=cp.float64)
    if basis.shape != (operator.geometry.D, operator.geometry.q):
        raise ValueError("PCA fiber must have full codimension")
    if not bool(cp.all(cp.isfinite(basis))) or not bool(cp.allclose(
        basis.T @ basis, cp.eye(operator.geometry.q), rtol=0.0, atol=1e-10
    )):
        raise ValueError("fiber basis must be finite and orthonormal")
    coordinate = cp.zeros(basis.shape[1], dtype=cp.float64)
    radius = float(operator.controls["tube_multiplier"]) * float(search_h) ** 2
    trace = []
    for iteration in range(int(solver["max_iterations"]) + 1):
        if _float(cp.linalg.norm(coordinate)) > radius:
            raise DRFailure("search_radius_exceeded", radius=radius)
        state = operator.evaluate(origin + basis @ coordinate)
        operator.spectral_gate(state)
        discrepancy = _float(cp.linalg.norm(basis @ basis.T - state.Q, ord=2))
        if not np.isfinite(discrepancy) or discrepancy > 0.9:
            raise DRFailure("normal_transversality_failure", discrepancy=discrepancy)
        projected = basis.T @ state.residual
        norm = _float(cp.linalg.norm(projected))
        trace.append({
            "iteration": iteration,
            "projected_residual": norm,
            "scaled_eigengap": state.scaled_eigengap,
            "scaled_largest_normal": state.scaled_largest_normal,
            "normal_discrepancy": discrepancy,
            "truncation": state.truncation,
        })
        if norm <= float(solver["projected_residual_tolerance"]):
            return {"point": _array(origin + basis @ coordinate), "state": state, "iterations": iteration, "trace": trace,
                    "coordinate": _array(coordinate), "fiber_basis": _array(basis),
                    "fiber_origin": _array(origin), "normal_space_source": "PCA"}
        if iteration == int(solver["max_iterations"]):
            raise DRFailure("newton_max_iter", residual=norm)
        jacobian = operator.fiber_jacobian(state, basis)
        condition = _float(cp.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > float(solver["jacobian_condition_max"]):
            raise DRFailure("jacobian_condition_failure", condition=condition)
        try:
            step = cp.linalg.solve(jacobian, -projected)
        except cp.linalg.LinAlgError as error:
            raise DRFailure("jacobian_condition_failure") from error
        accepted = False
        for backtrack in range(int(solver["max_backtracks"])):
            candidate = coordinate + (0.5**backtrack) * step
            if _float(cp.linalg.norm(candidate)) > radius:
                continue
            try:
                trial_state = operator.evaluate(origin + basis @ candidate)
                operator.spectral_gate(trial_state)
                trial = basis.T @ trial_state.residual
            except DRFailure:
                continue
            if _float(cp.linalg.norm(trial)) < norm:
                coordinate = candidate
                accepted = True
                break
        if not accepted:
            raise DRFailure("newton_no_descent", residual=norm)
    raise AssertionError("unreachable")


def _success(point, **diagnostics):
    return {"status": "success", "reason": None, "point": np.asarray(point), "diagnostics": diagnostics}


def _failure(reason: str, **diagnostics):
    return {"status": "failed", "reason": str(reason), "point": None, "diagnostics": diagnostics}


def dr_cell_gpu(data, geometry, queries, sigma: float, h1: float, h2: float, config: dict) -> dict:
    controls = dict(config["dr"])
    controls.setdefault("relative_omitted_weight_bound_max", 0.0)
    tree = cKDTree(data) if controls.get("gaussian_truncation_factor") is not None else None
    data_gpu = cp.asarray(np.asarray(data, dtype=float))
    fine_operator = EmpiricalDRGPU(data, geometry, h1, controls, tree, data_gpu)
    coarse_operator = EmpiricalDRGPU(data, geometry, h2, controls, tree, data_gpu)
    weights = dr_weights(h1, h2, sigma)
    single_rows, multi_rows = [], []
    for truth in queries:
        try:
            single_basis = estimate_pca_normal_gpu(coarse_operator, truth)
            single = solve_fiber_gpu(coarse_operator, truth, single_basis, h2, config["solver"])
        except DRFailure as error:
            single_rows.append(_failure(error.reason, **error.diagnostics))
        else:
            single_rows.append(_success(single["point"]))
        try:
            initial_basis = estimate_pca_normal_gpu(fine_operator, truth)
            fine = solve_fiber_gpu(fine_operator, truth, initial_basis, h1, config["solver"])
            reference_basis = estimate_pca_normal_gpu(fine_operator, fine["point"], state=fine["state"])
            coarse = solve_fiber_gpu(coarse_operator, fine["point"], reference_basis, h2, config["solver"])
        except DRFailure as error:
            multi_rows.append(_failure(error.reason, **error.diagnostics))
        else:
            basis = _array(reference_basis)
            point = fine["point"] + basis @ (weights["weights"][1] * coarse["coordinate"])
            multi_rows.append(_success(point, normal_guided={
                "reference_point": fine["point"], "reference_basis": basis,
                "t1": np.zeros(geometry.q), "t2": coarse["coordinate"],
                "normal_space_source": "PCA", "pca_radius": h1}))
    return {
        "method": "DR", "h1": float(h1), "h2": float(h2), "weights": weights,
        "Single": summarize_points(geometry, single_rows, "point"),
        "Multi": summarize_points(geometry, multi_rows, "point"),
        "single_rows": single_rows, "multi_rows": multi_rows,
    }


def prepare_common_start_gpu(data, tree: cKDTree, geometry, truth, sigma: float, controls: dict) -> dict:
    truth = np.asarray(truth, dtype=float)
    center = geometry.preliminary(truth, sigma)
    power = int(controls["kernel_power"])
    r0 = float(controls["c0"]) * sigma
    indices = np.asarray(sorted(tree.query_ball_point(center, r0)), dtype=int)
    if len(indices) == 0:
        raise YYFailure("first_stage_low_support", support_count=0)
    vectors = cp.asarray(np.asarray(data[indices] - center, dtype=float))
    weights = _polynomial_cutoff(cp.sum(vectors * vectors, axis=1), r0, power)
    ess = _effective_count(weights)
    total = _float(cp.sum(weights))
    minimum = float(controls["minimum_effective_count"])
    if not np.isfinite(total) or total <= 0.0 or ess < minimum:
        raise YYFailure("first_stage_mass_failure", support_count=len(indices), effective_count=ess)
    offset = _array(weights @ vectors / total)
    signal = float(np.linalg.norm(offset) / sigma)
    if not np.isfinite(signal) or signal < float(controls["minimum_signal_over_sigma"]):
        raise YYFailure("first_stage_signal_failure", signal_over_sigma=signal)
    axis = offset / np.linalg.norm(offset)
    return {
        "truth": truth, "center": center, "axis": axis,
        "digest": _digest(center, offset, axis),
        "first_support_count": int(_float(cp.count_nonzero(weights))),
        "first_effective_count": float(ess), "signal_over_sigma": signal,
    }


def make_candidate_cache_gpu(data, tree: cKDTree, start: dict, sigma: float, max_central_kappa: float, controls: dict):
    relative = np.asarray(controls["relative_radii"], dtype=float)
    radii = sigma * float(max_central_kappa) * relative
    r2 = float(controls["C2"]) * sigma * np.sqrt(np.log(1.0 / sigma))
    radius = float(np.hypot(r2, radii.max()))
    indices = np.asarray(sorted(tree.query_ball_point(start["center"], radius)), dtype=int)
    if len(indices) == 0:
        raise YYFailure("second_stage_low_support", support_count=0)
    return cp.asarray(np.asarray(data[indices] - start["center"], dtype=float))


def evaluate_common_start_gpu(data, tree: cKDTree, geometry, start: dict, sigma: float,
                              central_kappa: float, controls: dict, cached_vectors=None) -> dict:
    center = np.asarray(start["center"], dtype=float)
    axis = np.asarray(start["axis"], dtype=float)
    axis_gpu = cp.asarray(axis)
    power = int(controls["kernel_power"])
    relative = np.asarray(controls["relative_radii"], dtype=float)
    kappas = float(central_kappa) * relative
    radii = sigma * kappas
    weights_info = yy_weights(geometry.d, geometry.q, kappas, power)
    tolerance = float(controls.get("weight_residual_tolerance", 1e-10))
    if weights_info["max_abs_residual"] > tolerance:
        raise YYFailure("weight_constraint_failure", residual=weights_info["max_abs_residual"])
    r2 = float(controls["C2"]) * sigma * np.sqrt(np.log(1.0 / sigma))
    if cached_vectors is None:
        indices = np.asarray(sorted(tree.query_ball_point(center, float(np.hypot(r2, radii.max())))), dtype=int)
        if len(indices) == 0:
            raise YYFailure("second_stage_low_support", support_count=0)
        vectors = cp.asarray(np.asarray(data[indices] - center, dtype=float))
    else:
        vectors = cached_vectors
    longitudinal = vectors @ axis_gpu
    perpendicular = vectors - longitudinal[:, None] * axis_gpu
    longitudinal_weights = _longitudinal_profile(cp.abs(longitudinal) / r2)
    perpendicular_squared = cp.sum(perpendicular * perpendicular, axis=1)
    minimum = float(controls["minimum_effective_count"])
    outputs, stages = [], []
    for kappa, radius in zip(kappas, radii):
        weights = longitudinal_weights * _polynomial_cutoff(perpendicular_squared, radius, power)
        total = _float(cp.sum(weights))
        ess = _effective_count(weights)
        stage = {
            "kappa": float(kappa), "r1": float(radius),
            "support_count": int(_float(cp.count_nonzero(weights))), "effective_count": float(ess),
            "common_start_digest": start["digest"],
        }
        if not np.isfinite(total) or total <= 0.0 or ess < minimum:
            outputs.append(None)
            stages.append({**stage, "status": "failed", "reason": "second_stage_mass_failure"})
        else:
            outputs.append(center + _array(weights @ vectors / total))
            stages.append({**stage, "status": "success", "reason": None})
    center_index = np.flatnonzero(np.isclose(kappas, central_kappa, rtol=0.0, atol=1e-14))
    if len(center_index) != 1:
        raise YYFailure("central_scale_missing")
    single = outputs[int(center_index[0])]
    multi = None
    if all(value is not None for value in outputs):
        output_array = np.asarray(outputs)
        deltas = output_array - center
        axial = np.einsum("ki,i->k", deltas, axis)[:, None] * axis[None, :]
        complementary = deltas - axial
        multi = center + np.einsum("k,ki->i", weights_info["axial"], axial)
        multi += np.einsum("k,ki->i", weights_info["complementary"], complementary)
    return {
        "single": single, "multi": multi,
        "single_status": "success" if single is not None else "failed",
        "multi_status": "success" if multi is not None else "failed",
        "common_start_digest": start["digest"], "kappas": kappas, "radii": radii,
        "weights": weights_info, "stages": stages,
    }


def yy_cell_gpu(data, tree, geometry, queries, sigma: float, central_kappa: float, config: dict) -> dict:
    controls = dict(config["yy"])
    controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
    single_rows, multi_rows, starts = [], [], []
    for truth in queries:
        try:
            start = prepare_common_start_gpu(data, tree, geometry, truth, sigma, controls)
            fitted = evaluate_common_start_gpu(data, tree, geometry, start, sigma, central_kappa, controls)
        except YYFailure as error:
            single_rows.append(_failure(error.reason, **error.diagnostics))
            multi_rows.append(_failure(error.reason, **error.diagnostics))
            starts.append(None)
            continue
        starts.append(start["digest"])
        if fitted["single_status"] == "success":
            single_rows.append(_success(fitted["single"], common_start_digest=start["digest"]))
        else:
            stage = fitted["stages"][1]
            single_rows.append(_failure(stage["reason"], common_start_digest=start["digest"], stage=stage))
        if fitted["multi_status"] == "success":
            multi_rows.append(_success(fitted["multi"], common_start_digest=start["digest"]))
        else:
            failed = [stage for stage in fitted["stages"] if stage["status"] != "success"]
            multi_rows.append(_failure("second_stage_incomplete", common_start_digest=start["digest"], failed_stages=failed))
    kappas = central_kappa * np.asarray(controls["relative_radii"], dtype=float)
    return {
        "method": "YY", "central_kappa": float(central_kappa), "r1": float(central_kappa * sigma),
        "kappas": kappas, "radii": sigma * kappas,
        "weights": yy_weights(geometry.d, geometry.q, kappas, controls["kernel_power"]),
        "Single": summarize_points(geometry, single_rows, "point"),
        "Multi": summarize_points(geometry, multi_rows, "point"),
        "single_rows": single_rows, "multi_rows": multi_rows,
        "common_start_digests": starts,
    }
