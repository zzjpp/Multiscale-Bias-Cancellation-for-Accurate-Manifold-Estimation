from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import ncx2


@lru_cache(maxsize=64)
def legendre_rule(order: int):
    return roots_legendre(int(order))


def eta_c3(value):
    value = np.asarray(value, dtype=float)
    return 1.0 - 35.0 * value**4 + 84.0 * value**5 - 70.0 * value**6 + 20.0 * value**7


def longitudinal_profile(value):
    value = np.asarray(value, dtype=float)
    result = np.zeros_like(value)
    result[value <= 0.5] = 1.0
    mask = (value > 0.5) & (value < 1.0)
    result[mask] = eta_c3(2.0 * value[mask] - 1.0)
    return result


def radial_polynomial_moment(kappa, noncentral_radius, dimension, power, order):
    """E[(1-R^2/kappa^2)^p_+] for noncentral chi R."""
    nodes, quadrature_weights = legendre_rule(int(order))
    radius = 0.5 * float(kappa) * (nodes + 1.0)
    weights = 0.5 * float(kappa) * quadrature_weights
    noncentrality = np.asarray(noncentral_radius, dtype=float).reshape(-1) ** 2
    density = ncx2.pdf(radius[:, None] ** 2, int(dimension), noncentrality[None, :])
    kernel = np.maximum(1.0 - radius**2 / float(kappa) ** 2, 0.0) ** int(power)
    answer = np.einsum("r,r,rn->n", weights, kernel * 2.0 * radius, density)
    if not np.all(np.isfinite(answer)):
        raise FloatingPointError("non-finite CPU noncentral-chi radial moment")
    return answer


def longitudinal_moments(mean, sigma, radius, order):
    nodes, quadrature_weights = legendre_rule(int(order))
    location = float(radius) * nodes
    weights = float(radius) * quadrature_weights
    mean = np.asarray(mean, dtype=float).reshape(-1)
    answer_mass = np.empty(len(mean))
    answer_first = np.empty(len(mean))
    kernel = longitudinal_profile(np.abs(location) / float(radius))
    for start in range(0, len(mean), 4096):
        current = mean[start : start + 4096]
        density = np.exp(-0.5 * ((location[:, None] - current[None, :]) / sigma) ** 2)
        density /= sigma * np.sqrt(2.0 * np.pi)
        answer_mass[start : start + len(current)] = np.einsum(
            "r,r,rn->n", weights, kernel, density
        )
        answer_first[start : start + len(current)] = np.einsum(
            "r,r,r,rn->n", weights, location, kernel, density
        )
    if not np.all(np.isfinite(answer_mass)) or not np.all(np.isfinite(answer_first)):
        raise FloatingPointError("non-finite CPU longitudinal moment")
    return answer_mass, answer_first
