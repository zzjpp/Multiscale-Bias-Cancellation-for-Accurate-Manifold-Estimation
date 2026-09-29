from __future__ import annotations

import hashlib

import numpy as np

from .geometry_e5 import arclength_theta, gamma4, preliminary_point
from .kernels import longitudinal_profile


class EmpiricalQueryFailure(RuntimeError):
    """A declared E5 query failure; callers must retain it in the output."""


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.tobytes()).hexdigest()


def sample_noisy_curve(sample_size: int, sigma: float, rng: np.random.Generator) -> dict:
    if sample_size < 1 or sigma <= 0:
        raise ValueError("sample_size and sigma must be positive")
    theta = arclength_theta(rng.random(int(sample_size)))
    clean = gamma4(theta)
    observations = clean + float(sigma) * rng.standard_normal(clean.shape)
    return {
        "theta": theta,
        "clean": clean,
        "observations": observations,
        "observation_digest": _digest(observations),
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    square = float(np.sum(weights**2))
    return 0.0 if square <= 0 else total**2 / square


def empirical_common_start_query(
    observations: np.ndarray,
    theta: float,
    sigma: float,
    kappas,
    *,
    c0: float = 3.0,
    c2: float = 12.0,
    p: int = 6,
    minimum_effective_count: float = 8.0,
    minimum_signal_over_sigma: float = 0.1,
) -> dict:
    """Empirical Eqs. (10)--(11) with one immutable center and axis.

    Every transverse scale uses the same observations, supplied preliminary
    point, longitudinal radius, and first-stage axis.  No failed query is
    repaired or restarted from another scale.
    """

    data = np.asarray(observations, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4 or sigma <= 0:
        raise ValueError("E5 observations must have shape (n,4) and sigma>0")
    kappas = tuple(float(value) for value in kappas)
    center = preliminary_point(float(theta), float(sigma))
    displacement = data - center

    r0 = float(c0) * float(sigma)
    squared = np.sum(displacement**2, axis=1) / r0**2
    first_weights = np.maximum(1.0 - squared, 0.0) ** int(p)
    first_ess = _effective_sample_size(first_weights)
    first_mass = float(np.mean(first_weights))
    if not np.isfinite(first_mass) or first_mass <= 0 or first_ess < minimum_effective_count:
        raise EmpiricalQueryFailure("first-stage nondegenerate-mass gate")
    relative_first = first_weights @ displacement / np.sum(first_weights)
    signal_over_sigma = float(np.linalg.norm(relative_first) / sigma)
    if not np.isfinite(signal_over_sigma) or signal_over_sigma < minimum_signal_over_sigma:
        raise EmpiricalQueryFailure("first-stage displacement-signal gate")
    axis = relative_first / np.linalg.norm(relative_first)
    axis_digest = _digest(np.concatenate((center, axis, relative_first)))

    longitudinal = displacement @ axis
    perpendicular = displacement - longitudinal[:, None] * axis[None, :]
    r2 = float(c2) * float(sigma) * np.sqrt(np.log(1.0 / float(sigma)))
    longitudinal_weights = longitudinal_profile(np.abs(longitudinal) / r2)
    outputs, masses, effective_counts = [], [], []
    for kappa in kappas:
        r1 = float(kappa) * float(sigma)
        transverse_squared = np.sum(perpendicular**2, axis=1) / r1**2
        weights = longitudinal_weights * np.maximum(1.0 - transverse_squared, 0.0) ** int(p)
        ess = _effective_sample_size(weights)
        mass = float(np.mean(weights))
        if not np.isfinite(mass) or mass <= 0 or ess < minimum_effective_count:
            raise EmpiricalQueryFailure(f"second-stage nondegenerate-mass gate at kappa={kappa:g}")
        outputs.append(center + weights @ displacement / np.sum(weights))
        masses.append(mass)
        effective_counts.append(ess)

    return {
        "theta": float(theta),
        "truth": gamma4(float(theta)),
        "center": center,
        "axis": axis,
        "axis_digest": axis_digest,
        "first_signal_over_sigma": signal_over_sigma,
        "first_mass": first_mass,
        "first_effective_count": first_ess,
        "kappas": kappas,
        "outputs": np.asarray(outputs),
        "second_masses": np.asarray(masses),
        "second_effective_counts": np.asarray(effective_counts),
        "common_start_verified": True,
    }


def retained_fill_distance(retained_truth: np.ndarray, dense_count: int = 8192) -> float:
    retained = np.asarray(retained_truth, dtype=float)
    if retained.ndim != 2 or retained.shape[1] != 4 or len(retained) == 0:
        raise EmpiricalQueryFailure("no retained query set for fill distance")
    dense_theta = arclength_theta(np.arange(int(dense_count), dtype=float) / int(dense_count))
    dense = gamma4(dense_theta)
    minimum_squared = np.full(len(dense), np.inf)
    for start in range(0, len(dense), 512):
        block = dense[start : start + 512]
        minimum_squared[start : start + len(block)] = np.min(
            np.sum((block[:, None, :] - retained[None, :, :]) ** 2, axis=2), axis=1
        )
    return float(np.sqrt(np.max(minimum_squared)))
