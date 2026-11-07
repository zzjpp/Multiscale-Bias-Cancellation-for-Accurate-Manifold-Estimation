from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.spatial import cKDTree

from .geometry import arc_table, gamma


class OperatorFailure(RuntimeError):
    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


@dataclass
class STSPState:
    center: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    Q: np.ndarray
    residual: np.ndarray
    normalized_mass: float
    normalized_eigengap: float
    support_count: int
    effective_support: float
    support_points: np.ndarray
    quadrature_weights: np.ndarray
    base: np.ndarray
    extra: dict[str, Any] | None = None


def kernel_weight_and_derivative(observations, center, h, direction):
    delta = np.asarray(observations) - center
    base = np.maximum(1.0 - np.sum(delta * delta, axis=-1) / h**2, 0.0)
    return base**6, 12.0 / h**2 * base**5 * (delta @ direction)


def normal_projector_derivative(eigenvalues, eigenvectors, q, derivative_covariance):
    if eigenvalues[q] <= eigenvalues[q - 1]:
        raise OperatorFailure("eigengap_failure", failure_stage="jacobian")
    out = np.zeros_like(derivative_covariance)
    for i in range(q):
        for j in range(q, len(eigenvalues)):
            vi, vj = eigenvectors[:, i], eigenvectors[:, j]
            factor = (vj @ derivative_covariance @ vi) / (eigenvalues[i] - eigenvalues[j])
            out += factor * (np.outer(vj, vi) + np.outer(vi, vj))
    return out


class WeightedSTSP:
    """STSP operator for empirical equal weights or deterministic cubature weights."""
    def __init__(self, observations, h: float, weights=None, label="empirical"):
        self.data = np.asarray(observations, dtype=float)
        self.h = float(h)
        self.label = label
        if self.data.ndim != 2 or self.data.shape[1] != 3 or not np.isfinite(self.data).all():
            raise ValueError("finite n x 3 observations required")
        if weights is None:
            self.weights = np.full(len(self.data), 1.0 / len(self.data))
        else:
            self.weights = np.asarray(weights, dtype=float)
            self.weights = self.weights / self.weights.sum()
        if self.weights.shape != (len(self.data),) or np.any(self.weights < 0):
            raise ValueError("nonnegative observation weights required")
        self.tree = cKDTree(self.data)

    def evaluate(self, center, derivatives=True) -> STSPState:
        y = np.asarray(center, dtype=float)
        if y.shape != (3,) or not np.isfinite(y).all():
            raise OperatorFailure("nonfinite", failure_stage="operator_center")
        idx = np.asarray(self.tree.query_ball_point(y, self.h), dtype=int)
        if len(idx) < 3:
            raise OperatorFailure("low_support", failure_stage="operator", support_count=int(len(idx)))
        points = self.data[idx]
        delta = points - y
        base = np.maximum(1.0 - np.sum(delta * delta, axis=1) / self.h**2, 0.0)
        wk = self.weights[idx] * base**6
        mass = float(wk.sum())
        if not np.isfinite(mass) or mass <= 0:
            raise OperatorFailure("insufficient_mass", failure_stage="operator", mass=mass)
        mean = np.einsum("n,ni->i", wk, points) / mass
        centered = points - mean
        cov = np.einsum("n,ni,nj->ij", wk, centered, centered) / mass
        cov = (cov + cov.T) / 2
        values, vectors = np.linalg.eigh(cov)
        Q = vectors[:, :2] @ vectors[:, :2].T
        residual = Q @ (y - mean)
        normalized_mass = mass / self.h
        gap = float(values[2] - values[1]) / self.h**2
        effective = float(mass**2 / np.sum(wk**2))
        return STSPState(y.copy(), mean, cov, values, vectors, Q, residual, normalized_mass, gap, int(np.count_nonzero(base)), effective, points if derivatives else np.empty((0, 3)), self.weights[idx] if derivatives else np.empty(0), base if derivatives else np.empty(0))

    def directional(self, state: STSPState, direction):
        u = np.asarray(direction, dtype=float)
        delta = state.support_points - state.center
        dk = 12.0 / self.h**2 * state.base**5 * (delta @ u)
        dw = state.quadrature_weights * dk
        kweight = state.quadrature_weights * state.base**6
        mass = float(kweight.sum())
        dmass = float(dw.sum())
        dmean = (np.einsum("n,ni->i", dw, state.support_points) - state.mean * dmass) / mass
        centered = state.support_points - state.mean
        dcov = np.einsum("n,ni,nj->ij", dw, centered, centered) / mass - state.covariance * dmass / mass
        dcov = (dcov + dcov.T) / 2
        dQ = normal_projector_derivative(state.eigenvalues, state.eigenvectors, 2, dcov)
        dR = dQ @ (state.center - state.mean) + state.Q @ (u - dmean)
        return {"Dmu": dmean, "DSigma": dcov, "DQ": dQ, "DR": dR}

    def fiber_jacobian(self, state, basis):
        return np.column_stack([basis.T @ self.directional(state, basis[:, j])["DR"] for j in range(basis.shape[1])])


def population_cubature(sigma: float, arclength_nodes: int = 2048, gh_order: int = 3):
    """Deterministic expectation quadrature; no random population sample is used."""
    table = arc_table(65537)
    u = np.arange(arclength_nodes, dtype=float) / arclength_nodes
    clean = gamma(table.invert(u))
    nodes, weights = hermgauss(gh_order)
    a, b, c = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    noise = np.column_stack((a.ravel(), b.ravel(), c.ravel())) * (np.sqrt(2.0) * sigma)
    wa, wb, wc = np.meshgrid(weights, weights, weights, indexing="ij")
    nw = (wa * wb * wc).ravel() / np.pi ** 1.5
    observations = (clean[:, None, :] + noise[None, :, :]).reshape(-1, 3)
    quadrature_weights = np.repeat(np.full(arclength_nodes, 1.0 / arclength_nodes), len(noise)) * np.tile(nw, arclength_nodes)
    meta = {"method": "periodic_arclength_trapezoid_x_tensor_gauss_hermite", "arclength_nodes": int(arclength_nodes), "gauss_hermite_order": int(gh_order), "observation_nodes": int(len(observations)), "weight_sum": float(quadrature_weights.sum()), "not_monte_carlo": True}
    return observations, quadrature_weights, meta
