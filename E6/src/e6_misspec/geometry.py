from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_simpson

TAU = 2.0 * np.pi


def gamma3(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((1.5 * np.cos(t), np.sin(t), 0.3 * np.sin(2.0 * t)), axis=-1)


def gamma3_prime(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5 * np.sin(t), np.cos(t), 0.6 * np.cos(2.0 * t)), axis=-1)


@lru_cache(maxsize=2)
def _curve_arc_table(count: int = 131073):
    theta = np.linspace(0.0, TAU, int(count))
    speed = np.linalg.norm(gamma3_prime(theta), axis=1)
    length = cumulative_simpson(speed, x=theta, initial=0.0)
    return theta, length / length[-1]


def _curve_theta(normalized):
    theta, fraction = _curve_arc_table()
    return np.interp(np.mod(np.asarray(normalized, dtype=float), 1.0), fraction, theta)


def _canonical_columns(matrix):
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        index = int(np.argmax(np.abs(result[:, column])))
        if result[index, column] < 0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class Geometry:
    key: str
    label: str
    d: int
    D: int

    @property
    def q(self):
        return self.D - self.d

    def queries(self, count, phase=0.37):
        u = (np.arange(int(count), dtype=float) + float(phase)) / int(count)
        if self.key == "circle_R2":
            theta = TAU * u
            return np.column_stack((np.cos(theta), np.sin(theta)))
        return gamma3(_curve_theta(u))

    def latent_nodes(self, count):
        u = np.arange(int(count), dtype=float) / int(count)
        if self.key == "circle_R2":
            theta = TAU * u
            return np.column_stack((np.cos(theta), np.sin(theta)))
        return gamma3(_curve_theta(u))

    def tangent(self, point):
        x = np.asarray(point, dtype=float)
        if self.key == "circle_R2":
            tangent = np.array([-x[1], x[0]])
        else:
            grid = gamma3(_curve_theta(np.arange(4096, dtype=float) / 4096.0))
            theta = _curve_theta([np.argmin(np.sum((grid - x) ** 2, axis=1)) / 4096.0])[0]
            tangent = gamma3_prime(theta)
        return tangent / np.linalg.norm(tangent)

    def normal_basis(self, point):
        tangent = self.tangent(point)
        _, _, vh = np.linalg.svd(tangent[None, :], full_matrices=True)
        return _canonical_columns(vh[1:].T)


CIRCLE = Geometry("circle_R2", r"$S^1\subset\mathbb{R}^2$", 1, 2)
CURVE3 = Geometry("variable_curve_R3", r"$\gamma(t)=(1.5\cos t,\sin t,0.3\sin 2t)$", 1, 3)
GEOMETRIES = (CIRCLE, CURVE3)
