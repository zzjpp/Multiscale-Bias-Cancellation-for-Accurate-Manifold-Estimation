from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .records import OperatorFailure, STSPState


def kernel_weight_and_derivative(observations, center, h, direction):
    delta = np.asarray(observations) - center
    base = np.maximum(1 - np.sum(delta * delta, axis=-1) / h**2, 0)
    return base**6, 12 / h**2 * base**5 * (delta @ direction)


def normal_projector_derivative(eigenvalues, eigenvectors, q, derivative_covariance):
    if eigenvalues[q] - eigenvalues[q - 1] <= 0:
        raise OperatorFailure("eigengap_failure", failure_stage="jacobian")
    out = np.zeros_like(derivative_covariance)
    for i in range(q):
        for j in range(q, len(eigenvalues)):
            vi, vj = eigenvectors[:, i], eigenvectors[:, j]
            factor = (vj @ derivative_covariance @ vi) / (eigenvalues[i] - eigenvalues[j])
            out += factor * (np.outer(vj, vi) + np.outer(vi, vj))
    return out


class EmpiricalSTSP:
    def __init__(self, data, geometry, h, tree=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.geometry = geometry
        self.h = float(h)
        if self.data.ndim != 2 or self.data.shape[1] != geometry.D or len(self.data) == 0 or not np.isfinite(self.data).all():
            raise ValueError("finite n x D observations required")
        self.tree = tree if tree is not None else cKDTree(self.data)

    def evaluate(self, center, derivatives=True):
        y = np.asarray(center, dtype=float)
        if y.shape != (self.geometry.D,) or not np.isfinite(y).all():
            raise OperatorFailure("nonfinite", failure_stage="operator_center")
        indices = np.asarray(sorted(self.tree.query_ball_point(y, self.h)), dtype=int)
        if len(indices) < 2:
            raise OperatorFailure("low_support", failure_stage="operator", support_count=int(len(indices)))
        vectors = self.data[indices] - y
        squared = np.sum(vectors * vectors, axis=1)
        base = np.maximum(1 - squared / self.h**2, 0)
        weights = base**6
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise OperatorFailure("insufficient_mass", failure_stage="operator", support_count=int(len(indices)), weight_sum=total)
        offset = np.einsum("n,ni->i", weights, vectors) / total
        centered = vectors - offset
        covariance = np.einsum("n,ni,nj->ij", weights, centered, centered) / total
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(covariance).all():
            raise OperatorFailure("nonfinite", failure_stage="covariance")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        q = self.geometry.q
        Q = eigenvectors[:, :q] @ eigenvectors[:, :q].T
        P = np.eye(self.geometry.D) - Q
        normalized_mass = total / len(self.data) / self.h**self.geometry.d
        gap = float(eigenvalues[q] - eigenvalues[q - 1]) / self.h**2
        effective = total**2 / float(weights @ weights)
        derivative_weights = base**5 / len(self.data) if derivatives else np.empty(0)
        return STSPState(y.copy(), y + offset, covariance, eigenvalues, eigenvectors, P, Q, -Q @ offset, float(normalized_mass), gap, int(np.count_nonzero(weights)), float(effective), vectors if derivatives else np.empty((0, self.geometry.D)), derivative_weights)

    def directional(self, state: STSPState, direction):
        u = np.asarray(direction, dtype=float)
        vectors = state.support_vectors
        a = state.derivative_weights
        if len(a) == 0:
            raise ValueError("derivative support not retained")
        projections = vectors @ u
        M1u = float(a @ projections)
        M2u = np.einsum("n,ni,n->i", a, vectors, projections)
        M3u = np.einsum("n,ni,nj,n->ij", a, vectors, vectors, projections)
        mass = state.normalized_mass * self.h**self.geometry.d
        first = np.einsum("n,ni->i", (np.maximum(1 - np.sum(vectors * vectors, axis=1) / self.h**2, 0) ** 6) / len(self.data), vectors)
        mean_offset = first / mass
        factor = 12 / self.h**2
        Dmu = factor / mass * (M2u - mean_offset * M1u)
        Dcov = factor / mass * (M3u - np.outer(mean_offset, M2u) - np.outer(M2u, mean_offset) + (np.outer(mean_offset, mean_offset) - state.covariance) * M1u)
        Dcov = (Dcov + Dcov.T) / 2
        DQ = normal_projector_derivative(state.eigenvalues, state.eigenvectors, self.geometry.q, Dcov)
        DR = DQ @ (state.center - state.mean) + state.Q @ (u - Dmu)
        return {"Dmu": Dmu, "DSigma": Dcov, "DQ": DQ, "DR": DR}

    def fiber_jacobian(self, state: STSPState, basis: np.ndarray) -> np.ndarray:
        return np.column_stack([basis.T @ self.directional(state, basis[:, j])["DR"] for j in range(basis.shape[1])])
