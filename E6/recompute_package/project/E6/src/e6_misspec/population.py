from __future__ import annotations

from dataclasses import dataclass
import hashlib
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import ncx2


@lru_cache(maxsize=16)
def _legendre_rule(order):
    return leggauss(int(order))


def radial_polynomial_moment(kappa, noncentrality, dimension, p, order, chunk_size=128):
    nodes, quadrature_weights = _legendre_rule(int(order))
    radius = 0.5*float(kappa)*(nodes+1.0)
    transformed = 0.5*float(kappa)*quadrature_weights
    kernel = np.maximum(1.0-radius**2/float(kappa)**2, 0.0)**int(p)
    noncentrality = np.asarray(noncentrality)
    output = np.empty_like(noncentrality, dtype=float)
    # Numerically identical to the vectorized expression, with bounded temporaries.
    for start in range(0, len(noncentrality), int(chunk_size)):
        stop = min(start+int(chunk_size), len(noncentrality))
        density = ncx2.pdf(radius[:, None]**2, int(dimension), noncentrality[None, start:stop])
        output[start:stop] = np.einsum("r,r,rn->n", transformed, kernel*2.0*radius, density)
    return output


@dataclass(frozen=True)
class State:
    residual: np.ndarray
    normal_basis: np.ndarray
    eigenvalues: np.ndarray


class PopulationDR:
    def __init__(self, geometry, h, sigma, latent_count):
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        self.eta2 = self.h**2+self.sigma**2
        self.latent = geometry.latent_nodes(int(latent_count))

    def evaluate(self, point):
        displacement = self.latent-np.asarray(point, dtype=float)
        exponent = -0.5*np.sum(displacement**2, axis=1)/self.eta2
        weights = np.exp(exponent-np.max(exponent))
        normalized = weights/np.sum(weights)
        mean = np.einsum("n,ni->i", normalized, displacement)
        raw_second = np.einsum("n,ni,nj->ij", normalized, displacement, displacement)
        hessian = raw_second/self.eta2**2-np.eye(self.geometry.D)/self.eta2
        hessian = 0.5*(hessian+hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        normal_basis = eigenvectors[:, :self.geometry.q]
        projector = normal_basis@normal_basis.T
        return State(projector@(mean/self.eta2), normal_basis, eigenvalues)


class PopulationSTSP:
    def __init__(self, geometry, h, sigma, latent_count, radial_order, p=6):
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        self.latent = geometry.latent_nodes(int(latent_count))
        self.radial_order = int(radial_order)
        self.p = int(p)

    def evaluate(self, point):
        displacement = self.latent-np.asarray(point, dtype=float)
        noncentrality = np.sum(displacement**2, axis=1)/self.sigma**2
        kappa = self.h/self.sigma
        m0 = radial_polynomial_moment(kappa, noncentrality, self.geometry.D, self.p, self.radial_order)
        m1 = radial_polynomial_moment(kappa, noncentrality, self.geometry.D+2, self.p, self.radial_order)
        m2 = radial_polynomial_moment(kappa, noncentrality, self.geometry.D+4, self.p, self.radial_order)
        mass = float(np.mean(m0))
        if not np.isfinite(mass) or mass <= 0:
            raise ArithmeticError("nonpositive STSP population mass")
        offset = np.einsum("n,ni->i", m1, displacement)/(len(displacement)*mass)
        raw_second = self.sigma**2*np.eye(self.geometry.D)*float(np.mean(m1))
        raw_second += np.einsum("n,ni,nj->ij", m2, displacement, displacement)/len(displacement)
        covariance = 0.5*(raw_second/mass+(raw_second/mass).T)-np.outer(offset, offset)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal_basis = eigenvectors[:, :self.geometry.q]
        projector = normal_basis@normal_basis.T
        return State(-projector@offset, normal_basis, eigenvalues)


def _solve(operator, origin, basis, h, max_iter=20):
    coordinate = np.zeros(basis.shape[1], dtype=float)
    radius = 3.0*float(h)**2
    for iteration in range(max_iter+1):
        point = origin+basis@coordinate
        state = operator.evaluate(point)
        residual = basis.T@state.residual
        norm = float(np.linalg.norm(residual))
        if norm <= 1e-10:
            if np.linalg.norm(coordinate) > radius:
                raise ArithmeticError("solution left O(h^2) tube")
            return point, state
        if iteration == max_iter:
            break
        epsilon = max(1e-6*float(h), 1e-9)
        jacobian = np.column_stack([
            (basis.T@operator.evaluate(point+epsilon*basis[:, j]).residual
             - basis.T@operator.evaluate(point-epsilon*basis[:, j]).residual)/(2*epsilon)
            for j in range(basis.shape[1])
        ])
        step = np.linalg.solve(jacobian, -residual)
        accepted = False
        for backtrack in range(16):
            candidate = coordinate+(0.5**backtrack)*step
            if np.linalg.norm(candidate) > radius:
                continue
            trial = basis.T@operator.evaluate(origin+basis@candidate).residual
            if np.linalg.norm(trial) < norm:
                coordinate = candidate
                accepted = True
                break
        if not accepted:
            raise ArithmeticError("no descent inside O(h^2) tube")
    raise ArithmeticError("maximum local iterations")


def _digest(*arrays):
    sha = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        sha.update(str(array.dtype).encode("ascii"))
        sha.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        sha.update(array.tobytes())
    return sha.hexdigest()


def fit_pair(geometry, method, h1, h2, sigma, queries, resolution, p=6):
    if method == "DR":
        factory = lambda h: PopulationDR(geometry, h, sigma, resolution["dr_latent_count"])
    elif method == "STSP":
        factory = lambda h: PopulationSTSP(
            geometry, h, sigma, resolution["stsp_latent_count"], resolution["stsp_radial_order"], p
        )
    else:
        raise ValueError(method)
    fine_operator, coarse_operator = factory(h1), factory(h2)
    # DR's Gaussian Hessian remains part of its defining residual. Matching
    # uses the same centered compact radial PCA as STSP, at radius h1.
    pca_operator = PopulationSTSP(
        geometry, h1, sigma, resolution["stsp_latent_count"],
        resolution["stsp_radial_order"], p
    )
    fine_points, coarse_points, normals, failures, query_records = [], [], [], [], []
    for index, truth in enumerate(np.asarray(queries, dtype=float)):
        try:
            initial_basis = pca_operator.evaluate(truth).normal_basis
            fine, _ = _solve(fine_operator, truth, initial_basis, h1)
            reference_basis = pca_operator.evaluate(fine).normal_basis
            coarse, coarse_state = _solve(coarse_operator, fine, reference_basis, h2)
            coordinate = reference_basis.T@(coarse-fine)
            np.testing.assert_allclose(
                coarse, fine+reference_basis@coordinate, rtol=0.0, atol=2e-14
            )
            fine_points.append(fine)
            coarse_points.append(coarse)
            normals.append(reference_basis)
            failures.append("")
            query_records.append({
                "query_id": index, "status": "success", "failure_reason": None,
                "reference_point": fine, "reference_basis": reference_basis,
                "t1": np.zeros(geometry.q), "t2": coordinate,
                "normal_space_source": "population_kernel_weighted_PCA",
                "matching_kernel": "K(t)=(1-t)_+^6", "pca_radius": float(h1),
                "projected_coarse_residual_norm": float(np.linalg.norm(
                    reference_basis.T@coarse_state.residual)),
            })
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            fine_points.append(np.full(geometry.D, np.nan))
            coarse_points.append(np.full(geometry.D, np.nan))
            normals.append(np.full((geometry.D, geometry.q), np.nan))
            failures.append(f"query {index}: {error}")
            query_records.append({"query_id": index, "status": "failed",
                                  "failure_reason": str(error), "pca_radius": float(h1)})
    fine_points = np.asarray(fine_points)
    coarse_points = np.asarray(coarse_points)
    normals = np.asarray(normals)
    failure_mask = np.array([bool(value) for value in failures])
    return {
        "fine_points": fine_points,
        "coarse_points": coarse_points,
        "reference_normals": normals,
        "query_records": query_records,
        "matching_pca_kernel": "K(t)=(1-t)_+^6",
        "reference_pca_radius": float(h1),
        "failure_mask": failure_mask,
        "failures": failures,
        "fit_digest": _digest(fine_points, coarse_points, normals, failure_mask),
    }
