from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _minimum_norm(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    if np.linalg.matrix_rank(matrix) != matrix.shape[0]:
        raise ValueError("rank-deficient weight constraints")
    return matrix.T @ np.linalg.solve(matrix @ matrix.T, target)


def dr_weights(h1: float, h2: float, sigma: float) -> dict:
    coefficients = np.array([h1 * h1 + sigma * sigma, h2 * h2 + sigma * sigma], dtype=float)
    weights = np.array([coefficients[1], -coefficients[0]]) / (coefficients[1] - coefficients[0])
    residuals = np.array([weights.sum() - 1.0, weights @ coefficients])
    return {
        "coefficients": coefficients,
        "weights": weights,
        "residuals": residuals,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "sum_abs_weights": float(np.abs(weights).sum()),
    }


def yy_moment(kappa: float, d: int, q: int, p: int = 6) -> tuple[float, float | None]:
    if q == 1:
        return float(kappa) ** 2 / float(d + 2 * p + 2), None
    m = q - 1
    exponent = float(p) + 0.5 * float(d)

    def density(radius: float) -> float:
        return radius ** (m - 1) * max(1.0 - (radius / kappa) ** 2, 0.0) ** exponent * np.exp(-0.5 * radius**2)

    mass = quad(density, 0.0, float(kappa), epsabs=2e-13, epsrel=2e-13, limit=300)[0]
    second = quad(lambda r: r * r * density(r), 0.0, float(kappa), epsabs=2e-13, epsrel=2e-13, limit=300)[0]
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(second):
        raise ArithmeticError("YY moment quadrature failed")
    beta = second / (m * mass)
    alpha = (float(kappa) ** 2 - m * beta) / float(d + 2 * p + 2)
    return float(alpha), float(beta)


def yy_weights(d: int, q: int, kappas, p: int = 6) -> dict:
    kappas = np.asarray(tuple(kappas), dtype=float)
    moments = [yy_moment(value, d, q, p) for value in kappas]
    alpha = np.array([row[0] for row in moments])
    axial_matrix = np.vstack((np.ones(len(kappas)), 1.0 + alpha))
    axial = _minimum_norm(axial_matrix, np.array([1.0, 0.0]))
    if q == 1:
        beta = None
        complementary_matrix = axial_matrix
        complementary = axial.copy()
        residuals = np.concatenate((axial_matrix @ axial - [1.0, 0.0], complementary_matrix @ complementary - [1.0, 0.0]))
    else:
        if len(kappas) != 3:
            raise ValueError("higher-codimension YY requires exactly three radii")
        beta = np.array([row[1] for row in moments])
        complementary_matrix = np.vstack((np.ones(len(kappas)), alpha, 1.0 - beta))
        complementary = np.linalg.solve(complementary_matrix, np.array([1.0, 0.0, 0.0]))
        residuals = np.concatenate((axial_matrix @ axial - [1.0, 0.0], complementary_matrix @ complementary - [1.0, 0.0, 0.0]))
    return {
        "kappas": kappas,
        "alpha": alpha,
        "beta": beta,
        "axial": axial,
        "complementary": complementary,
        "residuals": residuals,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "W": float(np.abs(axial).sum() + np.abs(complementary).sum()),
        "axial_condition": float(np.linalg.cond(axial_matrix)),
        "complementary_condition": float(np.linalg.cond(complementary_matrix)),
    }

