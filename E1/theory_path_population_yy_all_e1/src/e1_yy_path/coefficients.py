from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def yy_alpha_beta(d: int, q: int, kappa: float, p: int = 6):
    m = int(q) - 1
    J = int(d) + 2 * int(p) + 2
    if m == 0:
        return float(kappa) ** 2 / J, None

    def base(radius, extra):
        return (
            radius ** (m - 1 + extra)
            * max(1.0 - radius**2 / float(kappa) ** 2, 0.0) ** (p + 0.5 * d)
            * np.exp(-0.5 * radius**2)
        )

    options = dict(epsabs=2e-13, epsrel=2e-13, limit=300)
    i0 = quad(lambda radius: base(radius, 0), 0.0, float(kappa), **options)[0]
    i2 = quad(lambda radius: base(radius, 2), 0.0, float(kappa), **options)[0]
    beta = i2 / (m * i0)
    alpha = (float(kappa) ** 2 - m * beta) / J
    return alpha, beta


def yy_weights(d: int, q: int, kappas, p: int = 6):
    kappas = np.asarray(kappas, dtype=float)
    moments = [yy_alpha_beta(d, q, value, p) for value in kappas]
    alpha = np.asarray([row[0] for row in moments])
    beta = None if q == 1 else np.asarray([row[1] for row in moments])
    axial_matrix = np.vstack((np.ones(len(kappas)), 1.0 + alpha))
    axial = np.linalg.lstsq(axial_matrix, np.asarray([1.0, 0.0]), rcond=None)[0]
    if q == 1:
        complementary_matrix = axial_matrix
        complementary = axial.copy()
        target = np.asarray([1.0, 0.0])
    else:
        complementary_matrix = np.vstack((np.ones(len(kappas)), alpha, 1.0 - beta))
        complementary = np.linalg.solve(complementary_matrix, np.asarray([1.0, 0.0, 0.0]))
        target = np.asarray([1.0, 0.0, 0.0])
    axial_residual = axial_matrix @ axial - np.asarray([1.0, 0.0])
    complementary_residual = complementary_matrix @ complementary - target
    return {
        "kappas": kappas,
        "alpha": alpha,
        "beta": beta,
        "axial": axial,
        "complementary": complementary,
        "axial_constraint_residual": axial_residual,
        "complementary_constraint_residual": complementary_residual,
        "max_abs_residual": float(
            max(np.max(np.abs(axial_residual)), np.max(np.abs(complementary_residual)))
        ),
    }
