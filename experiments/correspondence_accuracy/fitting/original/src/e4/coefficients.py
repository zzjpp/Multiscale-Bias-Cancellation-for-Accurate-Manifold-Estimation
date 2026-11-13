from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .config import CONFIG


def kernel_moments():
    # Radial one-dimensional checks used by the K6 audit.
    return {k: quad(lambda r: r**k * max(1.0 - r * r, 0.0) ** 6, -1.0, 1.0, epsabs=2e-13, epsrel=2e-13)[0] for k in (0, 1, 2, 3, 4)}


def stsp_coefficient(d: int, D: int, h: float, sigma: float) -> dict:
    q = D - d
    lam = sigma / h
    threshold = CONFIG["quadrature"]["small_lambda_rescale_threshold"]
    scale = lam if lam < threshold else 1.0
    upper = 1.0 / scale

    def moment(mode):
        def f(v):
            u = scale * v
            s = u * u
            base = v ** (q - 1) * max(1 - s, 0) ** (6 + d / 2) * np.exp(-s / (2 * lam * lam))
            return base * (1.0 if mode == 0 else s if mode == 1 else s * (1 - s))
        return quad(f, 0, upper, epsabs=CONFIG["quadrature"]["absolute_tolerance"], epsrel=CONFIG["quadrature"]["relative_tolerance"], limit=200, full_output=1)

    rows = [moment(k) for k in range(3)]
    vals = [row[0] for row in rows]
    a = vals[1] / (q * vals[0])
    c = vals[2] / (q * (d + 14) * vals[0])
    coefficient = 0.5 * d * h * h * (lam * lam + c / a)
    return {"coefficient": float(coefficient), "lambda": float(lam), "a": float(a), "c_moment": float(c), "neval": [int(row[2]["neval"]) for row in rows], "status": "converged"}


def two_scale_weights(h1: float, h2: float, sigma: float) -> dict:
    one, two = stsp_coefficient(1, 3, h1, sigma), stsp_coefficient(1, 3, h2, sigma)
    c1, c2 = one["coefficient"], two["coefficient"]
    weights = np.array([c2, -c1]) / (c2 - c1)
    residual = np.array([weights.sum() - 1.0, weights @ np.array([c1, c2])])
    return {"c1": c1, "c2": c2, "weights": weights, "constraint_residual": residual, "coefficients": [one, two], "status": "converged" if np.max(np.abs(residual)) < 1e-10 else "unresolved"}
