from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def dr_coefficient(d: int, h: float, sigma: float):
    return {
        "value": 0.5 * float(d) * (float(h) ** 2 + float(sigma) ** 2),
        "quadrature_error": 0.0,
    }


def stsp_coefficient(d: int, D: int, h: float, sigma: float, power: int, options: dict):
    q = int(D) - int(d)
    lam = float(sigma) / float(h)

    def base(radius):
        return (
            radius ** (q - 1)
            * max(1.0 - radius**2, 0.0) ** (int(power) + 0.5 * int(d))
            * np.exp(-radius**2 / (2.0 * lam**2))
        )

    quad_options = {
        "epsabs": float(options["absolute_tolerance"]),
        "epsrel": float(options["relative_tolerance"]),
        "limit": int(options["limit"]),
    }
    m0, e0 = quad(base, 0.0, 1.0, **quad_options)
    m2, e2 = quad(lambda r: r**2 * base(r), 0.0, 1.0, **quad_options)
    mixed, em = quad(lambda r: r**2 * (1.0 - r**2) * base(r), 0.0, 1.0, **quad_options)
    a_lam = m2 / (q * m0)
    c_lam = mixed / (q * (int(d) + 2 * int(power) + 2) * m0)
    return {
        "value": 0.5 * float(d) * float(h) ** 2 * (lam**2 + c_lam / a_lam),
        "quadrature_error": float(max(e0, e2, em)),
        "lambda": lam,
        "a_lambda": float(a_lam),
        "c_lambda": float(c_lam),
    }


def two_scale_weights(method: str, geometry, h1: float, h2: float, sigma: float, power: int, options: dict):
    if method == "DR":
        rows = [dr_coefficient(geometry.d, h, sigma) for h in (h1, h2)]
    elif method == "STSP":
        rows = [stsp_coefficient(geometry.d, geometry.D, h, sigma, power, options) for h in (h1, h2)]
    else:
        raise ValueError(method)
    coefficients = np.asarray([row["value"] for row in rows], dtype=float)
    weights = np.array([coefficients[1], -coefficients[0]]) / (coefficients[1] - coefficients[0])
    residual = np.array([weights.sum() - 1.0, weights @ coefficients])
    return {
        "coefficients": rows,
        "weights": weights,
        "constraint_residual": residual,
        "sum_abs_weights": float(np.abs(weights).sum()),
    }
