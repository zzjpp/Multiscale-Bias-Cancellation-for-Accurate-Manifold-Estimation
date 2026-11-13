from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import gammaln, ive, roots_legendre


@lru_cache(maxsize=16)
def legendre_rule(order: int):
    return roots_legendre(int(order))


def radial_mass(df: int, noncentral_radius, kappa: float, order: int, power: int = 6, kernel_kind: str = "polynomial"):
    noncentral_radius = np.asarray(noncentral_radius, dtype=float).reshape(-1)
    nodes, weights = legendre_rule(int(order))
    radius = float(kappa) * (nodes + 1.0) / 2.0
    transformed = weights * float(kappa) / 2.0
    if kernel_kind == "polynomial":
        kernel_weights = transformed * np.maximum(1.0 - (radius / kappa) ** 2, 0.0) ** int(power)
    elif kernel_kind == "truncated_gaussian":
        kernel_weights = transformed * np.exp(-0.5 * (radius / kappa) ** 2)
    else:
        raise ValueError(kernel_kind)
    answer = np.empty(len(noncentral_radius))
    nu = df / 2.0 - 1.0
    for start in range(0, len(noncentral_radius), 512):
        a = noncentral_radius[start : start + 512]
        density = np.empty((len(a), len(radius)))
        central = a == 0
        if np.any(central):
            density[central] = np.exp(
                (df - 1) * np.log(radius)
                - radius * radius / 2.0
                - (df / 2.0 - 1.0) * np.log(2.0)
                - gammaln(df / 2.0)
            )
        if np.any(~central):
            b = a[~central, None]
            with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
                log_density = (
                    np.log(radius)
                    + nu * (np.log(radius) - np.log(b))
                    - 0.5 * (radius - b) ** 2
                    + np.log(ive(nu, radius * b))
                )
                density[~central] = np.exp(log_density)
        answer[start : start + len(a)] = density @ kernel_weights
    if not np.isfinite(answer).all():
        raise FloatingPointError("non-finite noncentral-chi radial moment")
    return answer


@dataclass
class State:
    residual: np.ndarray
    normal_basis: np.ndarray
    eigenvalues: np.ndarray
    normalized_eigengap: float
    dr_normal_max: float | None
    normalized_mass: float
    pca_normal_basis: np.ndarray | None = None
    pca_normalized_eigengap: float | None = None


class PopulationDR:
    """Gaussian density-ridge population operator with noise integrated analytically."""

    def __init__(self, geometry, h: float, sigma: float, latent_count: int, pca_radial_order: int = 128, pca_padding: float = 12.0):
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        self.eta2 = self.h**2 + self.sigma**2
        latent, weights = geometry.latent_rule(int(latent_count))
        self.latent = latent
        self.latent_weights = weights
        # Radius h is an OBSERVATION-space cutoff, not a latent-manifold cutoff.
        self.pca_operator = PopulationSTSP(
            geometry, h, sigma, latent_count, pca_radial_order,
            noncentral_radius_padding=pca_padding, kernel_kind="polynomial")

    def evaluate(self, point):
        displacement = self.latent - np.asarray(point, dtype=float)
        exponent = -0.5 * np.sum(displacement**2, axis=1) / self.eta2
        shifted = exponent - np.max(exponent)
        weights = self.latent_weights * np.exp(shifted)
        mass = float(np.sum(weights))
        normalized = weights / mass
        mean = np.einsum("n,ni->i", normalized, displacement)
        raw_second = np.einsum("n,ni,nj->ij", normalized, displacement, displacement)
        hessian = raw_second / self.eta2**2 - np.eye(self.geometry.D) / self.eta2
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        projector = normal_basis @ normal_basis.T
        gap = float(eigenvalues[q] - eigenvalues[q - 1]) * self.h**2
        return State(
            projector @ (mean / self.eta2),
            normal_basis,
            eigenvalues,
            gap,
            float(eigenvalues[q - 1]),
            float(np.exp(np.max(exponent)) * mass),
        )


class PopulationSTSP:
    """Compact-kernel STSP population operator with exact Gaussian reduction."""

    def __init__(
        self,
        geometry,
        h: float,
        sigma: float,
        latent_count: int,
        radial_order: int,
        power: int = 6,
        noncentral_radius_padding: float = 12.0,
        kernel_kind: str = "polynomial",
    ):
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        latent, weights = geometry.latent_rule(int(latent_count))
        self.latent = latent
        self.latent_weights = weights
        self.radial_order = int(radial_order)
        self.power = int(power)
        self.kernel_kind = kernel_kind
        self.noncentral_radius_padding = float(noncentral_radius_padding)

    def evaluate(self, point):
        displacement_all = self.latent - np.asarray(point, dtype=float)
        radius_all = np.sqrt(np.maximum(np.sum(displacement_all**2, axis=1), 0.0)) / self.sigma
        kappa = self.h / self.sigma
        active = radius_all <= kappa + self.noncentral_radius_padding
        displacement = displacement_all[active]
        radius = radius_all[active]
        latent_weights = self.latent_weights[active]
        if len(displacement) == 0:
            raise ArithmeticError("STSP population tail cutoff retained no latent nodes")
        m0 = radial_mass(self.geometry.D, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        m1 = radial_mass(self.geometry.D + 2, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        m2 = radial_mass(self.geometry.D + 4, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        mass = float(np.sum(latent_weights * m0))
        if not np.isfinite(mass) or mass <= 0:
            raise ArithmeticError("nonpositive STSP population mass")
        offset = np.einsum("n,n,ni->i", latent_weights, m1, displacement) / mass
        raw_second = self.sigma**2 * np.eye(self.geometry.D) * float(np.sum(latent_weights * m1))
        raw_second += np.einsum("n,n,ni,nj->ij", latent_weights, m2, displacement, displacement)
        covariance = raw_second / mass - np.outer(offset, offset)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        projector = normal_basis @ normal_basis.T
        gap = float(eigenvalues[q] - eigenvalues[q - 1]) / self.h**2
        return State(-projector @ offset, normal_basis, eigenvalues, gap, None, mass / self.h,
                     normal_basis, gap)


def estimate_pca_normal(operator, point, controls: dict):
    """Full codimension-q PCA space; no geometry-normal API is consulted."""
    estimator = getattr(operator, "pca_operator", operator)
    state = estimator.evaluate(point)
    basis = state.pca_normal_basis
    gap = state.pca_normalized_eigengap
    if basis is None or not np.isfinite(basis).all() or gap is None or not np.isfinite(gap):
        raise ArithmeticError("invalid PCA normal space")
    if gap < float(controls["normalized_eigengap_min"]):
        raise ArithmeticError("PCA eigengap gate failed")
    return basis, {"source": "centered_weighted_population_covariance",
                   "normalized_eigengap": float(gap), "dimension": int(basis.shape[1]),
                   "pca_radius": float(operator.h), "kernel": getattr(estimator, "kernel_kind", "polynomial")}


def solve_normal_fiber(operator, origin, estimated_basis, h: float, controls: dict):
    """Solve the unchanged estimator residual along a fixed estimated PCA fiber."""
    origin = np.asarray(origin, dtype=float)
    basis = np.asarray(estimated_basis, dtype=float)
    if basis.shape != (operator.geometry.D, operator.geometry.q):
        raise ValueError("PCA fiber must have full codimension")
    if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(operator.geometry.q), rtol=0.0, atol=1e-10):
        raise ValueError("finite orthonormal PCA fiber required")
    coordinate = np.zeros(basis.shape[1], dtype=float)
    radius = float(controls["search_radius_factor"]) * float(h) ** 2
    trace = []
    evaluations = 0

    def inspect(value):
        nonlocal evaluations
        evaluations += 1
        if np.linalg.norm(value) > radius:
            raise ArithmeticError("search radius exceeded")
        state = operator.evaluate(origin + basis @ value)
        if state.normalized_eigengap < float(controls["normalized_eigengap_min"]):
            raise ArithmeticError("population eigengap gate failed")
        if state.dr_normal_max is not None and state.dr_normal_max >= float(controls["dr_normal_eigenvalue_max"]):
            raise ArithmeticError("DR normal Hessian is not negative")
        projected = basis.T @ state.residual
        return state, projected

    for iteration in range(int(controls["max_iterations"]) + 1):
        state, projected = inspect(coordinate)
        eps = max(1e-6 * float(h), 5e-10)
        jacobian = np.column_stack(
            [
                (
                    inspect(coordinate + eps * np.eye(len(coordinate))[index])[1]
                    - inspect(coordinate - eps * np.eye(len(coordinate))[index])[1]
                )
                / (2.0 * eps)
                for index in range(len(coordinate))
            ]
        )
        correction = np.linalg.solve(jacobian, -projected)
        projected_norm = float(np.linalg.norm(projected))
        full_norm = float(np.linalg.norm(state.residual))
        correction_norm = float(np.linalg.norm(correction))
        trace.append(
            {
                "iteration": iteration,
                "projected_residual": projected_norm,
                "full_residual": full_norm,
                "coordinate_correction": correction_norm,
                "normalized_eigengap": state.normalized_eigengap,
                "dr_normal_max": state.dr_normal_max,
            }
        )
        if (
            projected_norm <= float(controls["projected_residual_tolerance"])
            and full_norm <= float(controls["full_residual_tolerance"])
            and correction_norm <= float(controls["coordinate_correction_tolerance"])
        ):
            return {
                "status": "success",
                "point": origin + basis @ coordinate,
                "coordinate": coordinate,
                "fiber_basis": basis,
                "fiber_origin": origin.copy(),
                "normal_space_source": "PCA",
                "iterations": iteration,
                "function_evaluations": evaluations,
                "trace": trace,
                "final": trace[-1],
            }
        if iteration == int(controls["max_iterations"]):
            break
        accepted = False
        for backtrack in range(int(controls["max_backtracks"])):
            candidate = coordinate + (0.5**backtrack) * correction
            if np.linalg.norm(candidate) > radius:
                continue
            _, trial = inspect(candidate)
            if np.linalg.norm(trial) < projected_norm:
                coordinate = candidate
                accepted = True
                break
        if not accepted:
            raise ArithmeticError("Newton step has no descent inside the O(h^2) tube")
    raise ArithmeticError("maximum population root iterations")
