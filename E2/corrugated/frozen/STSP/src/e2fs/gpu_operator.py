from __future__ import annotations

"""Float64 GPU moments; the original CPU root controller and gates are reused."""

import numpy as np

from .operator import EmpiricalSTSP, normal_projector_derivative
from .records import OperatorFailure, STSPState


class GPUEmpiricalSTSP(EmpiricalSTSP):
    def __init__(self, data, geometry, h, tree=None):
        super().__init__(data, geometry, h, tree)
        import cupy as cp
        self.cp = cp
        self.data_gpu = cp.asarray(self.data, dtype=cp.float64)

    def evaluate(self, center, derivatives=True):
        cp = self.cp
        y = np.asarray(center, dtype=float)
        if y.shape != (self.geometry.D,) or not np.isfinite(y).all():
            raise OperatorFailure("nonfinite", failure_stage="operator_center")
        indices = np.asarray(sorted(self.tree.query_ball_point(y, self.h)), dtype=int)
        if len(indices) < 2:
            raise OperatorFailure("low_support", failure_stage="operator", support_count=int(len(indices)))
        vectors = self.data_gpu[cp.asarray(indices)] - cp.asarray(y)
        base = cp.maximum(1 - cp.sum(vectors * vectors, axis=1) / self.h**2, 0)
        weights = base**6
        total_gpu = cp.sum(weights)
        total = float(total_gpu.get())
        if not np.isfinite(total) or total <= 0:
            raise OperatorFailure("insufficient_mass", failure_stage="operator",
                                  support_count=int(len(indices)), weight_sum=total)
        offset = cp.einsum("n,ni->i", weights, vectors) / total_gpu
        centered = vectors - offset
        covariance = cp.einsum("n,ni,nj->ij", weights, centered, centered) / total_gpu
        covariance = .5 * (covariance + covariance.T)
        if not bool(cp.all(cp.isfinite(covariance)).item()):
            raise OperatorFailure("nonfinite", failure_stage="covariance")
        values, eigenvectors = cp.linalg.eigh(covariance)
        q = self.geometry.q
        Q = eigenvectors[:, :q] @ eigenvectors[:, :q].T
        normalized_mass = total / len(self.data) / self.h**self.geometry.d
        gap = float(((values[q] - values[q - 1]) / self.h**2).get())
        effective = total**2 / float((weights @ weights).get())
        return STSPState(
            y.copy(), y + cp.asnumpy(offset), cp.asnumpy(covariance),
            cp.asnumpy(values), cp.asnumpy(eigenvectors),
            np.eye(self.geometry.D) - cp.asnumpy(Q), cp.asnumpy(Q),
            cp.asnumpy(-Q @ offset), float(normalized_mass), gap,
            int(cp.count_nonzero(weights).get()), float(effective),
            vectors if derivatives else cp.empty((0, self.geometry.D), dtype=cp.float64),
            base**5 / len(self.data) if derivatives else cp.empty(0, dtype=cp.float64))

    def directional(self, state, direction):
        cp = self.cp
        u = cp.asarray(np.asarray(direction, dtype=float))
        vectors, a = state.support_vectors, state.derivative_weights
        if len(a) == 0:
            raise ValueError("derivative support not retained")
        projections = vectors @ u
        M1u = a @ projections
        M2u = cp.einsum("n,ni,n->i", a, vectors, projections)
        M3u = cp.einsum("n,ni,nj,n->ij", a, vectors, vectors, projections)
        mass = state.normalized_mass * self.h**self.geometry.d
        weights = cp.maximum(1 - cp.sum(vectors * vectors, axis=1) / self.h**2, 0)**6 / len(self.data)
        first = cp.einsum("n,ni->i", weights, vectors)
        offset = first / mass
        factor = 12 / self.h**2
        Dmu = factor / mass * (M2u - offset * M1u)
        Dcov = factor / mass * (
            M3u - cp.outer(offset, M2u) - cp.outer(M2u, offset)
            + (cp.outer(offset, offset) - cp.asarray(state.covariance)) * M1u)
        Dcov = .5 * (Dcov + Dcov.T)
        Dmu, Dcov = cp.asnumpy(Dmu), cp.asnumpy(Dcov)
        DQ = normal_projector_derivative(state.eigenvalues, state.eigenvectors, self.geometry.q, Dcov)
        DR = DQ @ (state.center - state.mean) + state.Q @ (np.asarray(direction) - Dmu)
        return {"Dmu": Dmu, "DSigma": Dcov, "DQ": DQ, "DR": DR}
