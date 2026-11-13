from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class DRFailure(RuntimeError):
    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


@dataclass
class DRState:
    center: np.ndarray
    residual: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    normal_basis: np.ndarray
    Q: np.ndarray
    vectors: np.ndarray
    normalized_weights: np.ndarray
    mean: np.ndarray
    raw_second: np.ndarray
    mass: float
    support_count: int
    effective_support: float
    scaled_eigengap: float
    scaled_largest_normal: float
    truncation: dict


def _projector_derivative(eigenvalues, eigenvectors, q: int, derivative: np.ndarray) -> np.ndarray:
    gap = float(eigenvalues[q] - eigenvalues[q - 1])
    if not np.isfinite(gap) or gap <= 0:
        raise DRFailure("eigengap_failure", raw_gap=gap)
    result = np.zeros_like(derivative)
    for i in range(q):
        for j in range(q, len(eigenvalues)):
            vi, vj = eigenvectors[:, i], eigenvectors[:, j]
            denominator = float(eigenvalues[i] - eigenvalues[j])
            factor = float(vj @ derivative @ vi) / denominator
            result += factor * (np.outer(vj, vi) + np.outer(vi, vj))
    return result


class EmpiricalDR:
    """Gaussian KDE ridge operator with an audited optional tail truncation."""

    def __init__(self, observations, geometry, h: float, controls: dict, tree=None):
        self.data = np.asarray(observations, dtype=float)
        self.geometry = geometry
        self.h = float(h)
        self.controls = dict(controls)
        if self.data.ndim != 2 or self.data.shape[1] != geometry.D or len(self.data) == 0:
            raise ValueError("finite n x D observations required")
        factor = self.controls.get("gaussian_truncation_factor")
        self.truncation_factor = None if factor is None else float(factor)
        self.tree = None
        if self.truncation_factor is not None:
            self.tree = tree if tree is not None else cKDTree(self.data)

    def _support(self, center: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.truncation_factor is None:
            return self.data - center, {"enabled": False, "omitted_count": 0, "relative_mass_bound": 0.0}
        radius = self.truncation_factor * self.h
        indices = np.asarray(sorted(self.tree.query_ball_point(center, radius)), dtype=int)
        if len(indices) == 0:
            raise DRFailure("low_support", support_count=0)
        vectors = self.data[indices] - center
        cutoff_weight = float(np.exp(-0.5 * self.truncation_factor**2))
        omitted = len(self.data) - len(indices)
        return vectors, {
            "enabled": True,
            "factor": self.truncation_factor,
            "radius": radius,
            "omitted_count": int(omitted),
            "cutoff_weight": cutoff_weight,
        }

    def evaluate(self, center: np.ndarray) -> DRState:
        center = np.asarray(center, dtype=float)
        vectors, truncation = self._support(center)
        squared = np.sum(vectors * vectors, axis=1)
        weights = np.exp(-0.5 * squared / self.h**2)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise DRFailure("insufficient_mass", weight_sum=total)
        if truncation["enabled"]:
            omitted_upper = truncation["omitted_count"] * truncation["cutoff_weight"]
            relative = omitted_upper / (total + omitted_upper)
            truncation["omitted_weight_upper"] = float(omitted_upper)
            truncation["relative_mass_bound"] = float(relative)
            limit = float(self.controls["relative_omitted_weight_bound_max"])
            if not np.isfinite(relative) or relative > limit:
                raise DRFailure("truncation_bound_failure", relative_mass_bound=relative, limit=limit)
        normalized = weights / total
        mean = normalized @ vectors
        raw_second = np.einsum("n,ni,nj->ij", normalized, vectors, vectors)
        hessian = raw_second / self.h**4 - np.eye(self.geometry.D) / self.h**2
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        Q = normal_basis @ normal_basis.T
        gradient = mean / self.h**2
        residual = Q @ gradient
        gap = float((eigenvalues[q] - eigenvalues[q - 1]) * self.h**2)
        largest_normal = float(eigenvalues[q - 1] * self.h**2)
        effective = total * total / float(weights @ weights)
        return DRState(
            center=center.copy(), residual=residual, gradient=gradient, hessian=hessian,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors, normal_basis=normal_basis, Q=Q,
            vectors=vectors, normalized_weights=normalized, mean=mean, raw_second=raw_second,
            mass=float(total / len(self.data)), support_count=len(vectors), effective_support=float(effective),
            scaled_eigengap=gap, scaled_largest_normal=largest_normal, truncation=truncation,
        )

    def _spectral_gate(self, state: DRState) -> None:
        margin = float(self.controls["normal_negativity_margin"])
        gap_min = float(self.controls["scaled_eigengap_min"])
        if not np.isfinite(state.scaled_largest_normal) or state.scaled_largest_normal > -margin:
            raise DRFailure("normal_negativity_failure", scaled_largest_normal=state.scaled_largest_normal, margin=margin)
        if not np.isfinite(state.scaled_eigengap) or state.scaled_eigengap < gap_min:
            raise DRFailure("eigengap_failure", scaled_eigengap=state.scaled_eigengap, minimum=gap_min)

    def directional_residual(self, state: DRState, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        projection = state.vectors @ direction
        mean_projection = float(state.mean @ direction)
        weighted_vector = np.einsum("n,ni,n->i", state.normalized_weights, state.vectors, projection)
        dmean = (weighted_vector - state.mean * mean_projection) / self.h**2 - direction
        weighted_third = np.einsum(
            "n,ni,nj,n->ij", state.normalized_weights, state.vectors, state.vectors, projection
        )
        draw = (weighted_third - state.raw_second * mean_projection) / self.h**2
        draw -= np.outer(direction, state.mean) + np.outer(state.mean, direction)
        dhessian = 0.5 * (draw + draw.T) / self.h**4
        dQ = _projector_derivative(
            state.eigenvalues, state.eigenvectors, self.geometry.q, dhessian
        )
        return dQ @ state.gradient + state.Q @ (dmean / self.h**2)

    def fiber_jacobian(self, state: DRState, basis: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [basis.T @ self.directional_residual(state, basis[:, column]) for column in range(basis.shape[1])]
        )


def solve_fiber(operator: EmpiricalDR, origin, basis, search_h: float, solver: dict) -> dict:
    origin = np.asarray(origin, dtype=float)
    basis = np.asarray(basis, dtype=float)
    coordinate = np.zeros(basis.shape[1])
    radius = float(operator.controls["tube_multiplier"]) * float(search_h) ** 2
    trace = []
    for iteration in range(int(solver["max_iterations"]) + 1):
        if np.linalg.norm(coordinate) > radius:
            raise DRFailure("search_radius_exceeded", radius=radius)
        state = operator.evaluate(origin + basis @ coordinate)
        operator._spectral_gate(state)
        discrepancy = float(np.linalg.norm(basis @ basis.T - state.Q, 2))
        if not np.isfinite(discrepancy) or discrepancy > 0.9:
            raise DRFailure("normal_transversality_failure", discrepancy=discrepancy)
        projected = basis.T @ state.residual
        norm = float(np.linalg.norm(projected))
        trace.append({
            "iteration": iteration, "projected_residual": norm,
            "scaled_eigengap": state.scaled_eigengap,
            "scaled_largest_normal": state.scaled_largest_normal,
            "normal_discrepancy": discrepancy,
            "truncation": state.truncation,
        })
        if norm <= float(solver["projected_residual_tolerance"]):
            return {"point": origin + basis @ coordinate, "state": state, "iterations": iteration, "trace": trace}
        if iteration == int(solver["max_iterations"]):
            raise DRFailure("newton_max_iter", residual=norm)
        jacobian = operator.fiber_jacobian(state, basis)
        condition = float(np.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > float(solver["jacobian_condition_max"]):
            raise DRFailure("jacobian_condition_failure", condition=condition)
        try:
            step = np.linalg.solve(jacobian, -projected)
        except np.linalg.LinAlgError as error:
            raise DRFailure("jacobian_condition_failure") from error
        accepted = False
        for backtrack in range(int(solver["max_backtracks"])):
            candidate = coordinate + (0.5**backtrack) * step
            if np.linalg.norm(candidate) > radius:
                continue
            try:
                trial_state = operator.evaluate(origin + basis @ candidate)
                operator._spectral_gate(trial_state)
                trial = basis.T @ trial_state.residual
            except DRFailure:
                continue
            if np.linalg.norm(trial) < norm:
                coordinate = candidate
                accepted = True
                break
        if not accepted:
            raise DRFailure("newton_no_descent", residual=norm)
    raise AssertionError("unreachable")


def fit_dr_query(data, geometry, truth, h1: float, h2: float, sigma: float, controls: dict, solver: dict, tree=None) -> dict:
    del sigma  # observations are already noisy; sigma enters only the cancellation weights
    truth = np.asarray(truth, dtype=float)
    true_basis = geometry.normal_basis(truth)
    fine_operator = EmpiricalDR(data, geometry, h1, controls, tree)
    coarse_operator = EmpiricalDR(data, geometry, h2, controls, tree)
    try:
        single = solve_fiber(coarse_operator, truth, true_basis, h2, solver)
    except DRFailure as error:
        single_row = {"status": "failed", "reason": error.reason, "diagnostics": error.diagnostics, "point": None}
    else:
        single_row = {"status": "success", "reason": None, "point": single["point"], "trace": single["trace"]}
    try:
        fine = solve_fiber(fine_operator, truth, true_basis, h1, solver)
        reference_basis = fine["state"].normal_basis
        coarse = solve_fiber(coarse_operator, fine["point"], reference_basis, h2, solver)
    except DRFailure as error:
        multi_row = {"status": "failed", "reason": error.reason, "diagnostics": error.diagnostics, "point": None}
        fine_point = None
        coarse_point = None
    else:
        multi_row = {"status": "fit_success", "reason": None, "point": None}
        fine_point = fine["point"]
        coarse_point = coarse["point"]
    return {"single": single_row, "multi": multi_row, "fine_point": fine_point, "coarse_point": coarse_point}

