from __future__ import annotations

import hashlib
from typing import Callable, Iterable

import numpy as np


def _solve_minimum_norm(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve A w = b with the minimum Euclidean norm and verify full row rank."""

    a = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.ndim != 2 or b.shape != (a.shape[0],):
        raise ValueError("incompatible constraint dimensions")
    if np.linalg.matrix_rank(a) != a.shape[0]:
        raise ValueError("weight constraints are rank deficient")
    gram = a @ a.T
    return a.T @ np.linalg.solve(gram, b)


def directional_weights(alpha: Iterable[float], beta: Iterable[float]) -> dict:
    """Return the Eq. (18) axial and complementary minimum-norm weights."""

    alpha = np.asarray(tuple(alpha), dtype=float)
    beta = np.asarray(tuple(beta), dtype=float)
    if alpha.ndim != 1 or beta.shape != alpha.shape or alpha.size < 3:
        raise ValueError("q>1 directional extrapolation needs K>=3 matched moments")
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
        raise ValueError("moments must be finite")

    axial_matrix = np.vstack((np.ones(alpha.size), 1.0 + alpha))
    complementary_matrix = np.vstack((np.ones(alpha.size), alpha, 1.0 - beta))
    a = _solve_minimum_norm(axial_matrix, np.array([1.0, 0.0]))
    b = _solve_minimum_norm(complementary_matrix, np.array([1.0, 0.0, 0.0]))
    residuals = np.array(
        [
            a.sum() - 1.0,
            a @ (1.0 + alpha),
            b.sum() - 1.0,
            b @ alpha,
            b @ (1.0 - beta),
        ]
    )
    return {
        "a": a,
        "b": b,
        "residuals": residuals,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "W": float(np.abs(a).sum() + np.abs(b).sum()),
        "axial_condition": float(np.linalg.cond(axial_matrix)),
        "complementary_condition": float(np.linalg.cond(complementary_matrix)),
    }


def dr_coefficient(d: int, h: float, sigma: float) -> float:
    if d < 1 or h <= 0 or sigma <= 0:
        raise ValueError("d, h, and sigma must be positive")
    return 0.5 * float(d) * (float(h) ** 2 + float(sigma) ** 2)


def two_scale_weights(
    coefficient: Callable[[float, float], float], h1: float, h2: float, sigma_hat: float
) -> np.ndarray:
    """Affine two-scale weights cancelling c(h, sigma_hat)."""

    if not (0 < h1 < h2 and sigma_hat > 0):
        raise ValueError("require 0<h1<h2 and sigma_hat>0")
    values = np.array([coefficient(h1, sigma_hat), coefficient(h2, sigma_hat)], dtype=float)
    denominator = values[1] - values[0]
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
        raise ValueError("coefficient constraints are singular")
    weights = np.array([values[1], -values[0]]) / denominator
    if abs(weights.sum() - 1.0) > 1e-12 or abs(weights @ values) > 1e-12:
        raise ArithmeticError("weight constraints failed")
    return weights


def freeze_fit_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reweight_frozen_fit(
    fine_points: np.ndarray,
    coarse_points: np.ndarray,
    failure_mask: np.ndarray,
    weights: Iterable[float],
) -> dict:
    """Pure reweighting: no estimator, branch search, or correspondence is called."""

    fine = np.asarray(fine_points, dtype=float)
    coarse = np.asarray(coarse_points, dtype=float)
    failed = np.asarray(failure_mask, dtype=bool)
    w = np.asarray(tuple(weights), dtype=float)
    if fine.shape != coarse.shape or fine.ndim != 2 or failed.shape != (fine.shape[0],):
        raise ValueError("frozen fit arrays have incompatible shapes")
    if w.shape != (2,) or abs(w.sum() - 1.0) > 1e-12:
        raise ValueError("invalid affine weights")
    before = freeze_fit_digest(fine, coarse, failed)
    output = w[0] * fine + w[1] * coarse
    output = output.copy()
    output[failed] = np.nan
    after = freeze_fit_digest(fine, coarse, failed)
    if before != after:
        raise RuntimeError("reweighting mutated the frozen fit")
    return {"points": output, "fit_digest": before, "failure_mask": failed.copy()}


def branch_tube_check(displacement: np.ndarray, h: float, multiplier: float) -> dict:
    """Enforce the predeclared O(h^2) local branch/correspondence tube."""

    if h <= 0 or multiplier <= 0:
        raise ValueError("h and multiplier must be positive")
    radius = float(multiplier) * float(h) ** 2
    norm = float(np.linalg.norm(np.asarray(displacement, dtype=float)))
    return {"accepted": bool(np.isfinite(norm) and norm <= radius), "norm": norm, "radius": radius}


def ridge_spectral_check(
    eigenvalues_ascending: Iterable[float], d: int, negativity_margin: float, eigengap_min: float
) -> dict:
    """Check lambda_{d+1}<0 and the tangent/normal Hessian eigengap."""

    values = np.asarray(tuple(eigenvalues_ascending), dtype=float)
    D = values.size
    q = D - int(d)
    if D < 2 or not (1 <= d < D) or negativity_margin < 0 or eigengap_min <= 0:
        raise ValueError("invalid spectral-gate parameters")
    if np.any(np.diff(values) < 0) or not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite and ascending")
    largest_normal = float(values[q - 1])
    smallest_tangent = float(values[q])
    gap = smallest_tangent - largest_normal
    negative = largest_normal <= -float(negativity_margin)
    separated = gap >= float(eigengap_min)
    return {
        "accepted": bool(negative and separated),
        "largest_normal_eigenvalue": largest_normal,
        "smallest_tangent_eigenvalue": smallest_tangent,
        "eigengap": float(gap),
        "negative_normal_block": bool(negative),
        "eigengap_ok": bool(separated),
    }
