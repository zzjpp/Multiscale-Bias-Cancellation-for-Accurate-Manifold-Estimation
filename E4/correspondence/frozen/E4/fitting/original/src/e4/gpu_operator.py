from __future__ import annotations

"""Double-precision CuPy backend for the empirical E4 STSP operator.

Neighbour discovery remains the same SciPy cKDTree operation used by the CPU
reference.  Kernel moments, covariance construction, and eigendecomposition
run on the selected GPU.  Only small 3-vectors/matrices return to the CPU-side
Newton controller.  This split keeps the branch and tolerance logic identical
to :class:`e4.operator.WeightedSTSP`.
"""

import numpy as np
from scipy.spatial import cKDTree

from .operator import OperatorFailure, STSPState, normal_projector_derivative


def require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - exercised on GPU host
        raise RuntimeError(
            "CuPy is required for the GPU backend; install requirements-gpu-cu13.txt"
        ) from exc
    return cp


def synchronize() -> None:
    cp = require_cupy()
    cp.cuda.get_current_stream().synchronize()


class GPUWeightedSTSP:
    """Numerically matched GPU implementation of ``WeightedSTSP``."""

    def __init__(self, observations, h: float, weights=None, label="empirical-gpu"):
        cp = require_cupy()
        self.cp = cp
        self.data = np.asarray(observations, dtype=np.float64)
        self.h = float(h)
        self.label = label
        if self.data.ndim != 2 or self.data.shape[1] != 3 or not np.isfinite(self.data).all():
            raise ValueError("finite n x 3 observations required")
        if weights is None:
            self.weights = np.full(len(self.data), 1.0 / len(self.data), dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights = self.weights / self.weights.sum()
        if self.weights.shape != (len(self.data),) or np.any(self.weights < 0):
            raise ValueError("nonnegative observation weights required")
        self.tree = cKDTree(self.data)
        self.data_gpu = cp.asarray(self.data, dtype=cp.float64)
        self.weights_gpu = cp.asarray(self.weights, dtype=cp.float64)

    def evaluate(self, center, derivatives=True) -> STSPState:
        cp = self.cp
        y = np.asarray(center, dtype=np.float64)
        if y.shape != (3,) or not np.isfinite(y).all():
            raise OperatorFailure("nonfinite", failure_stage="operator_center")
        idx = np.asarray(self.tree.query_ball_point(y, self.h), dtype=np.int64)
        if len(idx) < 3:
            raise OperatorFailure("low_support", failure_stage="operator", support_count=int(len(idx)))
        gpu_idx = cp.asarray(idx)
        points = self.data_gpu[gpu_idx]
        gpu_y = cp.asarray(y)
        delta = points - gpu_y
        base = cp.maximum(1.0 - cp.sum(delta * delta, axis=1) / self.h**2, 0.0)
        local_weights = self.weights_gpu[gpu_idx]
        wk = local_weights * base**6
        mass_gpu = cp.sum(wk)
        mass = float(mass_gpu.get())
        if not np.isfinite(mass) or mass <= 0:
            raise OperatorFailure("insufficient_mass", failure_stage="operator", mass=mass)
        mean_gpu = cp.einsum("n,ni->i", wk, points) / mass_gpu
        centered = points - mean_gpu
        cov_gpu = cp.einsum("n,ni,nj->ij", wk, centered, centered) / mass_gpu
        cov_gpu = (cov_gpu + cov_gpu.T) / 2
        values_gpu, vectors_gpu = cp.linalg.eigh(cov_gpu)
        q_gpu = vectors_gpu[:, :2] @ vectors_gpu[:, :2].T
        residual_gpu = q_gpu @ (gpu_y - mean_gpu)
        sum_wk2 = float(cp.sum(wk**2).get())
        mean = cp.asnumpy(mean_gpu)
        cov = cp.asnumpy(cov_gpu)
        values = cp.asnumpy(values_gpu)
        vectors = cp.asnumpy(vectors_gpu)
        q = cp.asnumpy(q_gpu)
        residual = cp.asnumpy(residual_gpu)
        normalized_mass = mass / self.h
        gap = float(values[2] - values[1]) / self.h**2
        effective = float(mass**2 / sum_wk2)
        return STSPState(
            y.copy(), mean, cov, values, vectors, q, residual,
            normalized_mass, gap, int(cp.count_nonzero(base).get()), effective,
            points if derivatives else cp.empty((0, 3), dtype=cp.float64),
            local_weights if derivatives else cp.empty(0, dtype=cp.float64),
            base if derivatives else cp.empty(0, dtype=cp.float64),
            extra={"backend": "cupy", "support_indices": idx},
        )

    def directional(self, state: STSPState, direction):
        cp = self.cp
        u = cp.asarray(np.asarray(direction, dtype=np.float64))
        center = cp.asarray(state.center)
        mean = cp.asarray(state.mean)
        delta = state.support_points - center
        dk = 12.0 / self.h**2 * state.base**5 * (delta @ u)
        dw = state.quadrature_weights * dk
        kweight = state.quadrature_weights * state.base**6
        mass_gpu = cp.sum(kweight)
        dmass_gpu = cp.sum(dw)
        dmean_gpu = (
            cp.einsum("n,ni->i", dw, state.support_points) - mean * dmass_gpu
        ) / mass_gpu
        centered = state.support_points - mean
        dcov_gpu = (
            cp.einsum("n,ni,nj->ij", dw, centered, centered) / mass_gpu
            - cp.asarray(state.covariance) * dmass_gpu / mass_gpu
        )
        dcov_gpu = (dcov_gpu + dcov_gpu.T) / 2
        dmean = cp.asnumpy(dmean_gpu)
        dcov = cp.asnumpy(dcov_gpu)
        d_q = normal_projector_derivative(
            state.eigenvalues, state.eigenvectors, 2, dcov
        )
        direction_cpu = np.asarray(direction, dtype=np.float64)
        d_r = d_q @ (state.center - state.mean) + state.Q @ (direction_cpu - dmean)
        return {"Dmu": dmean, "DSigma": dcov, "DQ": d_q, "DR": d_r}

    def fiber_jacobian(self, state, basis):
        return np.column_stack(
            [basis.T @ self.directional(state, basis[:, j])["DR"] for j in range(basis.shape[1])]
        )


def device_metadata() -> dict:
    cp = require_cupy()
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "backend": "cupy",
        "cupy": cp.__version__,
        "device_id_visible": int(device.id),
        "pci_bus_id": device.pci_bus_id,
        "name": name,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "total_global_memory": int(props["totalGlobalMem"]),
        "dtype": "float64",
    }
