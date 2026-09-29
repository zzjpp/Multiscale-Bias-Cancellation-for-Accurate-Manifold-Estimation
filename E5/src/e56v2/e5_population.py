from __future__ import annotations

import hashlib

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.stats import ncx2

from .geometry_e5 import gamma4, latent_nodes, preliminary_point
from .kernels import longitudinal_profile


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def radial_polynomial_moment(kappa: float, noncentrality, dimension: int, p: int, order: int):
    """E[(1-R^2/kappa^2)^p_+] for noncentral chi R."""

    if kappa <= 0 or dimension < 1 or p < 4 or order < 16:
        raise ValueError("invalid radial quadrature parameters")
    nodes, quadrature_weights = leggauss(int(order))
    radius = 0.5 * float(kappa) * (nodes + 1.0)
    transformed_weights = 0.5 * float(kappa) * quadrature_weights
    density = ncx2.pdf(radius[:, None] ** 2, int(dimension), np.asarray(noncentrality)[None, :])
    kernel = np.maximum(1.0 - radius**2 / float(kappa) ** 2, 0.0) ** int(p)
    return np.einsum("r,r,rn->n", transformed_weights, kernel * 2.0 * radius, density)


def longitudinal_moments(mean, sigma: float, radius: float, order: int):
    """Mass and first raw moment under the exact Eq. (58) profile."""

    if sigma <= 0 or radius <= 0 or order < 16:
        raise ValueError("invalid longitudinal quadrature parameters")
    nodes, quadrature_weights = leggauss(int(order))
    location = float(radius) * nodes
    weights = float(radius) * quadrature_weights
    density = np.exp(-0.5 * ((location[:, None] - np.asarray(mean)[None, :]) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    kernel = longitudinal_profile(np.abs(location) / float(radius))
    mass = np.einsum("r,r,rn->n", weights, kernel, density)
    first = np.einsum("r,r,r,rn->n", weights, location, kernel, density)
    return mass, first


def _first_stage(clean: np.ndarray, z: np.ndarray, sigma: float, c0: float, p: int, radial_order: int):
    displacement = clean - z
    noncentrality = np.sum(displacement**2, axis=1) / sigma**2
    mass_nodes = radial_polynomial_moment(c0, noncentrality, clean.shape[1], p, radial_order)
    mean_nodes = radial_polynomial_moment(c0, noncentrality, clean.shape[1] + 2, p, radial_order)
    mass = float(np.mean(mass_nodes))
    if not np.isfinite(mass) or mass <= 0:
        raise ArithmeticError("first-stage population mass failed")
    relative_mean = np.einsum("n,ni->i", mean_nodes, displacement) / (len(clean) * mass)
    signal = float(np.linalg.norm(relative_mean))
    if not np.isfinite(signal) or signal <= 0:
        raise ArithmeticError("first-stage population signal failed")
    axis = relative_mean / signal
    return relative_mean, axis, mass


def _second_stage(
    clean: np.ndarray,
    z: np.ndarray,
    axis: np.ndarray,
    sigma: float,
    kappa: float,
    c2: float,
    p: int,
    radial_order: int,
    longitudinal_order: int,
):
    displacement = clean - z
    longitudinal_mean = displacement @ axis
    perpendicular_mean = displacement - longitudinal_mean[:, None] * axis
    r2 = float(c2) * sigma * np.sqrt(np.log(1.0 / sigma))
    longitudinal_mass, longitudinal_first = longitudinal_moments(
        longitudinal_mean, sigma, r2, longitudinal_order
    )
    noncentrality = np.sum(perpendicular_mean**2, axis=1) / sigma**2
    dimension = clean.shape[1] - 1
    perpendicular_mass = radial_polynomial_moment(kappa, noncentrality, dimension, p, radial_order)
    perpendicular_mean_factor = radial_polynomial_moment(
        kappa, noncentrality, dimension + 2, p, radial_order
    )
    node_mass = longitudinal_mass * perpendicular_mass
    mass = float(np.mean(node_mass))
    if not np.isfinite(mass) or mass <= 0:
        raise ArithmeticError("second-stage population mass failed")
    numerator = np.einsum("n,i->i", longitudinal_first * perpendicular_mass, axis)
    numerator += np.einsum(
        "n,ni->i", longitudinal_mass * perpendicular_mean_factor, perpendicular_mean
    )
    relative_mean = numerator / (len(clean) * mass)
    return z + relative_mean, mass


def common_start_population_query(
    theta: float,
    sigma: float,
    kappas,
    *,
    c0: float = 3.0,
    c2: float = 12.0,
    p: int = 6,
    latent_count: int = 2048,
    radial_order: int = 96,
    longitudinal_order: int = 96,
):
    """Evaluate Eqs. (10)--(11) with one frozen center and axis."""

    if not (0 < sigma < 1):
        raise ValueError("the population regime requires 0<sigma<1")
    kappas = tuple(float(value) for value in kappas)
    _, clean = latent_nodes(int(latent_count))
    z = preliminary_point(float(theta), float(sigma))
    relative_first, axis, first_mass = _first_stage(clean, z, sigma, c0, p, radial_order)
    outputs, masses = [], []
    axis_digest = _array_digest(z, axis, relative_first)
    for kappa in kappas:
        output, mass = _second_stage(
            clean,
            z,
            axis,
            sigma,
            kappa,
            c2,
            p,
            radial_order,
            longitudinal_order,
        )
        outputs.append(output)
        masses.append(mass)
    return {
        "theta": float(theta),
        "truth": gamma4(float(theta)),
        "center": z,
        "axis": axis,
        "axis_digest": axis_digest,
        "first_signal_over_sigma": float(np.linalg.norm(relative_first) / sigma),
        "first_mass": first_mass,
        "kappas": kappas,
        "outputs": np.asarray(outputs),
        "second_masses": np.asarray(masses),
        "common_start_verified": True,
    }
