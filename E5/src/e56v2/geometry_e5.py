from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_simpson


TAU = 2.0 * np.pi


def gamma4(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((1.5 * np.cos(t), np.sin(t), 0.3 * np.sin(2.0 * t), np.zeros_like(t)), axis=-1)


def gamma4_prime(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5 * np.sin(t), np.cos(t), 0.6 * np.cos(2.0 * t), np.zeros_like(t)), axis=-1)


def gamma4_second(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5 * np.cos(t), -np.sin(t), -1.2 * np.sin(2.0 * t), np.zeros_like(t)), axis=-1)


def tangent4(theta):
    derivative = gamma4_prime(theta)
    return derivative / np.linalg.norm(derivative, axis=-1, keepdims=True)


def curvature4(theta):
    first = gamma4_prime(theta)
    second = gamma4_second(theta)
    speed2 = np.sum(first * first, axis=-1, keepdims=True)
    return second / speed2 - first * (np.sum(first * second, axis=-1, keepdims=True) / speed2**2)


def _normal_frame(theta):
    curvature = curvature4(theta)
    curvature_direction = curvature / np.linalg.norm(curvature, axis=-1, keepdims=True)
    e4 = np.zeros_like(curvature_direction)
    e4[..., 3] = 1.0
    axial = (curvature_direction + e4) / np.sqrt(2.0)
    complementary = (curvature_direction - e4) / np.sqrt(2.0)
    return axial, complementary


def preliminary_point(theta, sigma: float):
    """A supplied common M_pre normal graph satisfying the Assumption 3.9 scaling.

    The order-sigma axial displacement enforces the independent signal regime;
    the complementary offset is order sigma^2 and is retained for Eq. (18).
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    t = np.asarray(theta, dtype=float)
    axial, complementary = _normal_frame(t)
    axial_amplitude = float(sigma) * (1.0 + 0.1 * np.cos(3.0 * t))
    complementary_amplitude = 0.25 * float(sigma) ** 2 * np.sin(2.0 * t)
    return gamma4(t) + axial_amplitude[..., None] * axial + complementary_amplitude[..., None] * complementary


@lru_cache(maxsize=2)
def _arc_table(count: int = 131073):
    theta = np.linspace(0.0, TAU, int(count))
    speed = np.linalg.norm(gamma4_prime(theta), axis=1)
    length = cumulative_simpson(speed, x=theta, initial=0.0)
    return theta, length / length[-1]


def arclength_theta(normalized):
    theta, fraction = _arc_table()
    return np.interp(np.mod(np.asarray(normalized, dtype=float), 1.0), fraction, theta)


def latent_nodes(count: int):
    if count < 32:
        raise ValueError("population quadrature requires at least 32 latent nodes")
    theta = arclength_theta(np.arange(int(count), dtype=float) / int(count))
    return theta, gamma4(theta)


def query_grid(count: int, phase: float = 0.37):
    theta = arclength_theta((np.arange(int(count), dtype=float) + float(phase)) / int(count))
    return theta, preliminary_point(theta, 1.0) * 0.0 + gamma4(theta)
