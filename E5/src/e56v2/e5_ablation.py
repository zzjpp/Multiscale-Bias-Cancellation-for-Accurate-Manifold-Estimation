from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry_e5 import TAU, gamma4, tangent4


METHODS = ("Single", "Scalar-A", "Scalar-C", "Directional")


def combine_common_start(result: dict, axial_weights, complementary_weights, single_kappa: float = 5.0):
    kappas = np.asarray(result["kappas"], dtype=float)
    outputs = np.asarray(result["outputs"], dtype=float)
    center = np.asarray(result["center"], dtype=float)
    axis = np.asarray(result["axis"], dtype=float)
    a = np.asarray(axial_weights, dtype=float)
    b = np.asarray(complementary_weights, dtype=float)
    if outputs.shape[0] != len(a) or a.shape != b.shape:
        raise ValueError("weights and common-start outputs do not match")
    matches = np.flatnonzero(np.isclose(kappas, float(single_kappa), rtol=0.0, atol=1e-14))
    if len(matches) != 1:
        raise ValueError("single transverse ratio must be present exactly once")
    deltas = outputs - center
    parallel = np.einsum("ki,i->k", deltas, axis)[:, None] * axis[None, :]
    complementary = deltas - parallel
    return {
        "Single": outputs[int(matches[0])],
        "Scalar-A": center + np.einsum("k,ki->i", a, deltas),
        "Scalar-C": center + np.einsum("k,ki->i", b, deltas),
        "Directional": center
        + np.einsum("k,ki->i", a, parallel)
        + np.einsum("k,ki->i", b, complementary),
    }


def distance_to_curve(point: np.ndarray, coarse_count: int = 2048) -> float:
    point = np.asarray(point, dtype=float)
    grid = TAU * np.arange(int(coarse_count), dtype=float) / int(coarse_count)
    squared = np.sum((gamma4(grid) - point) ** 2, axis=1)
    index = int(np.argmin(squared))
    step = TAU / int(coarse_count)
    center = grid[index]

    def objective(offset: float) -> float:
        delta = gamma4((center + offset) % TAU) - point
        return float(delta @ delta)

    optimized = minimize_scalar(objective, bounds=(-step, step), method="bounded", options={"xatol": 1e-14})
    if not optimized.success:
        raise ArithmeticError("nearest-point projection failed")
    return float(np.sqrt(max(optimized.fun, 0.0)))


def component_metrics(theta: float, point: np.ndarray, common_axis: np.ndarray) -> dict:
    truth = gamma4(float(theta))
    tangent = tangent4(float(theta))
    normal_projector = np.eye(4) - np.outer(tangent, tangent)
    normal_axis = normal_projector @ np.asarray(common_axis, dtype=float)
    norm = float(np.linalg.norm(normal_axis))
    if norm <= 0:
        raise ArithmeticError("common axis has no normal component")
    normal_axis /= norm
    axial_projector = np.outer(normal_axis, normal_axis)
    complementary_projector = normal_projector - axial_projector
    error = np.asarray(point, dtype=float) - truth
    return {
        "geometric": distance_to_curve(point),
        "axial_normal": float(np.linalg.norm(axial_projector @ error)),
        "complementary_normal": float(np.linalg.norm(complementary_projector @ error)),
        "tangential": float(np.linalg.norm(np.outer(tangent, tangent) @ error)),
    }
