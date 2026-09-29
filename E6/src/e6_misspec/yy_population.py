from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss

from .coefficients import yy_weights
from .geometry import gamma3, gamma3_prime
from .population import radial_polynomial_moment


def _eta_c3(u):
    u = np.asarray(u, dtype=float)
    return 1.0 - 35.0*u**4 + 84.0*u**5 - 70.0*u**6 + 20.0*u**7


def _longitudinal_profile(t):
    t = np.asarray(t, dtype=float)
    value = np.zeros_like(t)
    value[t <= 0.5] = 1.0
    mask = (t > 0.5) & (t < 1.0)
    value[mask] = _eta_c3(2.0*t[mask]-1.0)
    return value


def _longitudinal_moments(mean, sigma, radius, order):
    nodes, quadrature_weights = leggauss(int(order))
    location = float(radius)*nodes
    weights = float(radius)*quadrature_weights
    density = np.exp(-0.5*((location[:, None]-np.asarray(mean)[None, :])/sigma)**2)
    density /= sigma*np.sqrt(2.0*np.pi)
    kernel = _longitudinal_profile(np.abs(location)/float(radius))
    mass = np.einsum("r,r,rn->n", weights, kernel, density)
    first = np.einsum("r,r,r,rn->n", weights, location, kernel, density)
    return mass, first


def _gamma3_second(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5*np.cos(t), -np.sin(t), -1.2*np.sin(2.0*t)), axis=-1)


def _curve_theta_from_point(point):
    grid_theta = np.linspace(0.0, 2.0*np.pi, 32768, endpoint=False)
    grid = gamma3(grid_theta)
    index = int(np.argmin(np.sum((grid-np.asarray(point, dtype=float))**2, axis=1)))
    theta0 = grid_theta[index]
    step = 2.0*np.pi/len(grid_theta)
    candidates = theta0 + np.linspace(-step, step, 65)
    values = gamma3(candidates)
    return float(candidates[int(np.argmin(np.sum((values-point)**2, axis=1)))] % (2.0*np.pi))


def _preliminary_point(geometry, truth, sigma):
    truth = np.asarray(truth, dtype=float)
    if geometry.key == "circle_R2":
        theta = float(np.arctan2(truth[1], truth[0]) % (2.0*np.pi))
        axial = truth/np.linalg.norm(truth)
        center = truth + sigma*(1.0+0.1*np.cos(3.0*theta))*axial
        return center, theta
    theta = _curve_theta_from_point(truth)
    tangent = gamma3_prime(theta)
    tangent /= np.linalg.norm(tangent)
    acceleration = _gamma3_second(theta)
    curvature = acceleration-tangent*np.dot(acceleration, tangent)
    axial = curvature/np.linalg.norm(curvature)
    complementary = np.cross(tangent, axial)
    complementary /= np.linalg.norm(complementary)
    center = (
        truth
        + sigma*(1.0+0.1*np.cos(3.0*theta))*axial
        + 0.25*sigma**2*np.sin(2.0*theta)*complementary
    )
    return center, theta


def _first_stage(clean, center, sigma, c0, p, radial_order):
    displacement = clean-center
    noncentrality = np.sum(displacement**2, axis=1)/sigma**2
    mass_nodes = radial_polynomial_moment(c0, noncentrality, clean.shape[1], p, radial_order)
    mean_nodes = radial_polynomial_moment(c0, noncentrality, clean.shape[1]+2, p, radial_order)
    mass = float(np.mean(mass_nodes))
    if not np.isfinite(mass) or mass <= 0:
        raise ArithmeticError("first-stage population mass failed")
    relative_mean = np.einsum("n,ni->i", mean_nodes, displacement)/(len(clean)*mass)
    signal = float(np.linalg.norm(relative_mean))
    if not np.isfinite(signal) or signal <= 0:
        raise ArithmeticError("first-stage population signal failed")
    return relative_mean/signal, signal/sigma, mass


def _second_stage(clean, center, axis, sigma, kappa, c2, p, radial_order, longitudinal_order):
    displacement = clean-center
    longitudinal_mean = displacement@axis
    perpendicular_mean = displacement-longitudinal_mean[:, None]*axis
    radius = float(c2)*sigma*np.sqrt(np.log(1.0/sigma))
    longitudinal_mass, longitudinal_first = _longitudinal_moments(
        longitudinal_mean, sigma, radius, longitudinal_order
    )
    noncentrality = np.sum(perpendicular_mean**2, axis=1)/sigma**2
    dimension = clean.shape[1]-1
    perpendicular_mass = radial_polynomial_moment(
        kappa, noncentrality, dimension, p, radial_order
    )
    perpendicular_mean_factor = radial_polynomial_moment(
        kappa, noncentrality, dimension+2, p, radial_order
    )
    node_mass = longitudinal_mass*perpendicular_mass
    mass = float(np.mean(node_mass))
    if not np.isfinite(mass) or mass <= 0:
        raise ArithmeticError("second-stage population mass failed")
    numerator = np.einsum("n,i->i", longitudinal_first*perpendicular_mass, axis)
    numerator += np.einsum(
        "n,ni->i", longitudinal_mass*perpendicular_mean_factor, perpendicular_mean
    )
    return center+numerator/(len(clean)*mass), mass


def _digest(*arrays):
    sha = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        sha.update(str(array.dtype).encode("ascii"))
        sha.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        sha.update(array.tobytes())
    return sha.hexdigest()


def frozen_outputs(geometry, queries, sigma, radii, settings, *, c0=3.0, c2=12.0, p=6):
    clean = geometry.latent_nodes(int(settings["latent_count"]))
    kappas = np.asarray(radii, dtype=float)/float(sigma)
    query_array = np.asarray(queries, dtype=float)

    def evaluate(item):
        index, truth = item
        try:
            center, _ = _preliminary_point(geometry, truth, sigma)
            axis, signal_ratio, _ = _first_stage(
                clean, center, sigma, c0, p, int(settings["radial_order"])
            )
            scale_outputs = []
            for kappa in kappas:
                output, _ = _second_stage(
                    clean, center, axis, sigma, float(kappa), c2, p,
                    int(settings["radial_order"]), int(settings["longitudinal_order"])
                )
                scale_outputs.append(output)
            return center, axis, scale_outputs, "", signal_ratio
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            return (
                np.full(geometry.D, np.nan),
                np.full(geometry.D, np.nan),
                np.full((len(radii), geometry.D), np.nan),
                f"query {index}: {error}",
                np.nan,
            )

    workers = int(settings.get("workers", 1))
    indexed = list(enumerate(query_array))
    if workers == 1:
        evaluated = [evaluate(item) for item in indexed]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(evaluate, indexed))
    centers, axes, outputs, failures, signal_ratios = map(list, zip(*evaluated))
    centers = np.asarray(centers)
    axes = np.asarray(axes)
    outputs = np.asarray(outputs)
    failure_mask = np.array([bool(value) for value in failures])
    return {
        "centers": centers,
        "axes": axes,
        "outputs": outputs,
        "failure_mask": failure_mask,
        "failures": failures,
        "signal_ratios": np.asarray(signal_ratios),
        "fit_digest": _digest(centers, axes, outputs, failure_mask),
    }

def geometric_sensitivity(geometry, fits, sigma, radii, deltas, p=6):
    centers = fits["centers"]
    axes = fits["axes"]
    outputs = fits["outputs"]
    displacement = outputs-centers[:, None, :]
    longitudinal_scalar = np.einsum("nkd,nd->nk", displacement, axes)
    longitudinal = longitudinal_scalar[:, :, None]*axes[:, None, :]
    complementary = displacement-longitudinal
    true = yy_weights(geometry.d, geometry.q, radii, sigma, p)

    def combine(weights):
        axial_part = np.einsum("k,nkd->nd", weights["axial"], longitudinal)
        complementary_part = np.einsum("k,nkd->nd", weights["complementary"], complementary)
        return centers+axial_part+complementary_part

    reference = combine(true)
    rows = []
    for delta in deltas:
        assumed = yy_weights(geometry.d, geometry.q, radii, (1.0+delta)*sigma, p)
        estimate = combine(assumed)
        distance = np.linalg.norm(estimate-reference, axis=1)
        rows.append({
            "delta": float(delta),
            "delta_infinity": float(np.max(distance)),
            "delta_average": float(np.mean(distance)),
            "fit_digest": fits["fit_digest"],
            "query_count": int(len(centers)),
            "resolved_queries": int(np.sum(~fits["failure_mask"])),
        })
    return rows