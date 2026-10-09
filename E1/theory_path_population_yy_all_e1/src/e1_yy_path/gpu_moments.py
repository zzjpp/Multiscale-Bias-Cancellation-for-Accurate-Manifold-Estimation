from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import gammaln, roots_legendre


def cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy.special import i0e, i1e
    except ImportError as error:  # pragma: no cover - GPU host only
        raise RuntimeError("CuPy with cupyx.scipy.special.i0e/i1e is required") from error
    return cp, i0e, i1e


@lru_cache(maxsize=64)
def legendre_host(order: int):
    return roots_legendre(int(order))


def _series_scaled_iv(cp, nu, value, terms=96):
    tiny = cp.finfo(cp.float64).tiny
    safe = cp.maximum(value, tiny)
    leading = cp.exp(-safe + float(nu) * cp.log(safe / 2.0) - float(gammaln(float(nu) + 1.0)))
    term = cp.ones_like(safe)
    total = term.copy()
    z = safe * safe / 4.0
    for index in range(1, int(terms)):
        term = term * z / (float(index) * (float(nu) + float(index)))
        total = total + term
    result = leading * total
    return cp.where(value == 0.0, 1.0 if float(nu) == 0.0 else 0.0, result)


def scaled_iv(cp, i0e, i1e, nu, value):
    """Stable exp(-x) I_nu(x) for integer and half-integer orders used here."""
    value = cp.asarray(value, dtype=cp.float64)
    small = value < 16.0
    series = _series_scaled_iv(cp, float(nu), value)
    safe = cp.maximum(value, 16.0)
    if abs(float(nu) - round(float(nu))) < 1e-12:
        order = int(round(float(nu)))
        previous = i0e(safe)
        current = i1e(safe)
        if order == 0:
            large = previous
        elif order == 1:
            large = current
        else:
            for index in range(1, order):
                previous, current = current, previous - (2.0 * index / safe) * current
            large = current
    else:
        target = int(round(float(nu) - 0.5))
        scaled_i_half = cp.sqrt(2.0 / (cp.pi * safe)) * (-0.5 * cp.expm1(-2.0 * safe))
        scaled_i_minus_half = cp.sqrt(2.0 / (cp.pi * safe)) * (0.5 * (1.0 + cp.exp(-2.0 * safe)))
        if target == 0:
            large = scaled_i_half
        else:
            previous = scaled_i_minus_half
            current = scaled_i_half
            current_order = 0.5
            for _ in range(target):
                following = previous - (2.0 * current_order / safe) * current
                previous, current = current, following
                current_order += 1.0
            large = current
    return cp.where(small, series, large)


def radial_polynomial_moment_gpu(dimension, noncentral_radius, kappa, power, order):
    cp, i0e, i1e = cupy_modules()
    noncentral_radius = cp.asarray(noncentral_radius, dtype=cp.float64).reshape(-1)
    host_nodes, host_weights = legendre_host(int(order))
    radius = float(kappa) * (cp.asarray(host_nodes) + 1.0) / 2.0
    weights = cp.asarray(host_weights) * float(kappa) / 2.0
    kernel_weights = weights * cp.maximum(1.0 - (radius / float(kappa)) ** 2, 0.0) ** int(power)
    result = cp.empty(len(noncentral_radius), dtype=cp.float64)
    nu = int(dimension) / 2.0 - 1.0
    for start in range(0, len(noncentral_radius), 4096):
        a = noncentral_radius[start : start + 4096]
        central = a == 0.0
        central_density = cp.exp(
            (int(dimension) - 1) * cp.log(radius)
            - radius * radius / 2.0
            - (int(dimension) / 2.0 - 1.0) * np.log(2.0)
            - float(gammaln(int(dimension) / 2.0))
        )
        safe_a = cp.maximum(a[:, None], cp.finfo(cp.float64).tiny)
        argument = radius[None, :] * safe_a
        bessel = cp.maximum(
            scaled_iv(cp, i0e, i1e, nu, argument), cp.finfo(cp.float64).tiny
        )
        log_density = (
            cp.log(radius)[None, :]
            + nu * (cp.log(radius)[None, :] - cp.log(safe_a))
            - 0.5 * (radius[None, :] - safe_a) ** 2
            + cp.log(bessel)
        )
        density = cp.exp(log_density)
        density[central] = central_density
        result[start : start + len(a)] = density @ kernel_weights
    if not bool(cp.all(cp.isfinite(result)).item()):
        raise FloatingPointError("non-finite GPU noncentral-chi radial moment")
    return result


def _eta_c3_gpu(cp, value):
    return 1.0 - 35.0 * value**4 + 84.0 * value**5 - 70.0 * value**6 + 20.0 * value**7


def longitudinal_moments_gpu(mean, sigma, radius, order):
    cp, _, _ = cupy_modules()
    host_nodes, host_weights = legendre_host(int(order))
    location = float(radius) * cp.asarray(host_nodes)
    weights = float(radius) * cp.asarray(host_weights)
    scaled = cp.abs(location) / float(radius)
    kernel = cp.where(
        scaled <= 0.5,
        1.0,
        cp.where(scaled < 1.0, _eta_c3_gpu(cp, 2.0 * scaled - 1.0), 0.0),
    )
    mean = cp.asarray(mean, dtype=cp.float64).reshape(-1)
    mass = cp.empty(len(mean), dtype=cp.float64)
    first = cp.empty(len(mean), dtype=cp.float64)
    for start in range(0, len(mean), 4096):
        current = mean[start : start + 4096]
        density = cp.exp(-0.5 * ((location[:, None] - current[None, :]) / float(sigma)) ** 2)
        density /= float(sigma) * np.sqrt(2.0 * np.pi)
        mass[start : start + len(current)] = (weights * kernel) @ density
        first[start : start + len(current)] = (weights * location * kernel) @ density
    if not bool(cp.all(cp.isfinite(mass)).item()) or not bool(cp.all(cp.isfinite(first)).item()):
        raise FloatingPointError("non-finite GPU longitudinal moment")
    return mass, first
