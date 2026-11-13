from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import quad

from .config import CONFIG
from .records import IntegralDiagnostic, OperatorFailure


def adaptive_integral(function, lo: float, hi: float, name: str):
    cfg = CONFIG["quadrature"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = quad(function, lo, hi, epsabs=cfg["absolute_tolerance"], epsrel=cfg["relative_tolerance"], limit=cfg["coefficient_subinterval_limit"], full_output=True)
    value, error, info = result[:3]
    messages = [str(w.message) for w in caught] + [str(x) for x in result[3:]]
    ok = not messages and np.isfinite(value) and error <= max(cfg["absolute_tolerance"], cfg["relative_tolerance"] * abs(value))
    diagnostic = IntegralDiagnostic(name, int(info["neval"]), float(error), int(info["last"]), "converged" if ok else "unresolved", {"interval": [lo, hi], "messages": messages, "value": float(value)})
    return float(value), diagnostic


def stsp_coefficient(d: int, D: int, h: float, sigma: float) -> dict:
    q = D - d
    lam = sigma / h
    if d < 1 or q < 1 or h <= 0 or sigma <= 0:
        raise ValueError("invalid STSP parameters")
    scale = lam if lam < CONFIG["quadrature"]["small_lambda_rescale_threshold"] else 1.0
    upper = 1 / scale

    def integrand(v, mode):
        u = scale * v
        s = u * u
        base = v ** (q - 1) * max(1 - s, 0) ** (6 + d / 2) * np.exp(-s / (2 * lam * lam))
        return base * (1 if mode == 0 else s if mode == 1 else s * (1 - s))

    rows = [adaptive_integral(lambda v, m=m: integrand(v, m), 0, upper, f"STSP_S_moment_{m}") for m in range(3)]
    values = [row[0] for row in rows]
    normalizer = values[0]
    if normalizer <= 0 or not np.isfinite(normalizer):
        raise OperatorFailure("nonfinite", failure_stage="coefficient", normalizer=normalizer)
    a = values[1] / (q * normalizer)
    c_moment = values[2] / (q * (d + 14) * normalizer)
    coefficient = 0.5 * d * h * h * (lam * lam + c_moment / a)
    return {"d": d, "D": D, "q": q, "h": float(h), "sigma": float(sigma), "lambda": float(lam), "a": float(a), "c_moment": float(c_moment), "coefficient": float(coefficient), "diagnostics": [row[1] for row in rows], "status": "converged" if all(row[1].status == "converged" for row in rows) else "unresolved"}


def two_scale_weights(d: int, D: int, h1: float, h2: float, sigma: float) -> dict:
    coefficients = [stsp_coefficient(d, D, h, sigma) for h in (h1, h2)]
    c1, c2 = [row["coefficient"] for row in coefficients]
    if h1 >= h2 or c2 == c1:
        raise ValueError("distinct increasing scales required")
    weights = np.array([c2, -c1]) / (c2 - c1)
    residuals = np.array([weights.sum() - 1, weights @ np.array([c1, c2])])
    status = "converged" if all(row["status"] == "converged" for row in coefficients) and np.max(np.abs(residuals)) < CONFIG["validation"]["weight_residual_tolerance"] else "unresolved"
    return {"coefficients": coefficients, "c1": c1, "c2": c2, "weights": weights, "weight_constraint_residuals": residuals, "sum_abs_weights": float(np.abs(weights).sum()), "denominator": c2 - c1, "condition": float(np.linalg.cond(np.array([[1.0, 1.0], [c1, c2]]))), "status": status}
