from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree
from scipy.special import roots_jacobi


TAU = 2.0 * np.pi


# Frozen E7 corrugated-torus geometry.  These constants and formulas are
# intentionally duplicated here so this E1 population extension is a
# self-contained scientific artifact rather than an import from quarantine.
E7_MAJOR_RADIUS = 2.0


def corrugated_minor_radius(v):
    v = np.asarray(v, dtype=float)
    return 0.7 * (1.0 + 0.18 * np.cos(3.0 * v) + 0.10 * np.sin(2.0 * v))


def corrugated_minor_radius_prime(v):
    v = np.asarray(v, dtype=float)
    return 0.7 * (-0.54 * np.sin(3.0 * v) + 0.20 * np.cos(2.0 * v))


def corrugated_rho(v):
    v = np.asarray(v, dtype=float)
    return E7_MAJOR_RADIUS + corrugated_minor_radius(v) * np.cos(v)


def corrugated_zeta(v):
    v = np.asarray(v, dtype=float)
    return corrugated_minor_radius(v) * np.sin(v)


def corrugated_area_jacobian(v):
    v = np.asarray(v, dtype=float)
    radius = corrugated_minor_radius(v)
    derivative = corrugated_minor_radius_prime(v)
    return corrugated_rho(v) * np.sqrt(radius**2 + derivative**2)


def corrugated_torus(u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    radial = corrugated_rho(v)
    return np.stack((radial * np.cos(u), radial * np.sin(u), corrugated_zeta(v)), axis=-1)


def corrugated_derivatives(u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    radius = corrugated_minor_radius(v)
    derivative = corrugated_minor_radius_prime(v)
    radial = corrugated_rho(v)
    radial_prime = derivative * np.cos(v) - radius * np.sin(v)
    z_prime = derivative * np.sin(v) + radius * np.cos(v)
    xu = np.stack((-radial * np.sin(u), radial * np.cos(u), np.zeros_like(u)), axis=-1)
    xv = np.stack((radial_prime * np.cos(u), radial_prime * np.sin(u), z_prime), axis=-1)
    return xu, xv


@lru_cache(maxsize=2)
def corrugated_area_cdf(intervals: int = 65536):
    grid = np.linspace(0.0, TAU, int(intervals) + 1, dtype=float)
    cumulative = np.concatenate(([0.0], cumulative_trapezoid(corrugated_area_jacobian(grid), grid)))
    cumulative /= cumulative[-1]
    return grid, cumulative


def corrugated_equal_area_v(count: int):
    grid, cumulative = corrugated_area_cdf()
    quantiles = (np.arange(int(count), dtype=float) + 0.5) / int(count)
    return np.interp(quantiles, cumulative, grid)


class CorrugatedMeridianOptimizer:
    """Accurate point-to-surface distance via the generating meridian."""

    def __init__(self, count: int = 8192):
        self.count = int(count)
        self.step = TAU / self.count
        self.v = TAU * np.arange(self.count, dtype=float) / self.count
        self.points = np.column_stack((corrugated_rho(self.v), corrugated_zeta(self.v)))
        self.tree = cKDTree(self.points)

    def closest(self, meridian_point):
        target = np.asarray(meridian_point, dtype=float)
        _, indices = self.tree.query(target, k=8)
        candidates = []
        selected = []
        for raw_index in np.atleast_1d(indices):
            index = int(raw_index)
            if any(min(abs(index - old), self.count - abs(index - old)) <= 4 for old in selected):
                continue
            selected.append(index)
        for index in selected:
            center = self.v[index]

            def objective(value):
                wrapped = value % TAU
                return (corrugated_rho(wrapped) - target[0]) ** 2 + (corrugated_zeta(wrapped) - target[1]) ** 2

            coarse = minimize_scalar(
                objective,
                bounds=(center - 2.0 * self.step, center + 2.0 * self.step),
                method="bounded",
                options={"xatol": 1e-14},
            )

            def residual(value):
                wrapped = value[0] % TAU
                return np.asarray(
                    [corrugated_rho(wrapped) - target[0], corrugated_zeta(wrapped) - target[1]]
                )

            refined = least_squares(
                residual,
                np.asarray([coarse.x]),
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=50,
            )
            value = float(refined.x[0] % TAU)
            candidates.append((float(objective(value)), value))
        return min(candidates)


_CORRUGATED_MERIDIAN = CorrugatedMeridianOptimizer()


@lru_cache(maxsize=32)
def sphere_latitude_rule(d: int, order: int):
    alpha = 0.5 * (int(d) - 2)
    nodes, weights = roots_jacobi(int(order), alpha, alpha)
    return nodes, weights / np.sum(weights)


def signed_axes(dimension: int):
    eye = np.eye(int(dimension))
    return np.concatenate((eye, -eye), axis=0)


@dataclass(frozen=True)
class Geometry:
    key: str
    label: str
    d: int
    D: int

    @property
    def q(self):
        return self.D - self.d

    @property
    def torus(self):
        return self.key == "clifford_torus_r4"

    def query_grid(self, count: int, phase: float):
        if int(count) != 1:
            raise ValueError("the E1 homogeneous population order check uses one canonical query")
        if self.torus:
            radius = 1.0 / np.sqrt(2.0)
            points = np.asarray([[radius, 0.0, radius, 0.0]])
        else:
            points = np.zeros((1, self.D))
            points[0, 0] = 1.0
        return points, points.copy()

    def integration_node_count(self, resolution: int):
        if self.torus:
            return int(resolution) ** 2
        if self.key == "circle_r2":
            return int(resolution)
        return int(resolution) * 2 * int(self.d)

    def latent_rule(self, resolution: int):
        resolution = int(resolution)
        if self.key == "circle_r2":
            angle = TAU * np.arange(resolution, dtype=float) / resolution
            nodes = np.column_stack((np.cos(angle), np.sin(angle)))
            return nodes, np.full(resolution, 1.0 / resolution)
        if self.torus:
            angle = TAU * np.arange(resolution, dtype=float) / resolution
            u, v = np.meshgrid(angle, angle, indexing="ij")
            radius = 1.0 / np.sqrt(2.0)
            nodes = radius * np.column_stack(
                (np.cos(u.ravel()), np.sin(u.ravel()), np.cos(v.ravel()), np.sin(v.ravel()))
            )
            return nodes, np.full(len(nodes), 1.0 / len(nodes))
        latitude, latitude_weights = sphere_latitude_rule(self.d, resolution)
        directions = signed_axes(self.d)
        tangent_radius = np.sqrt(np.maximum(1.0 - latitude**2, 0.0))
        nodes = np.zeros((resolution * len(directions), self.D))
        nodes[:, 0] = np.repeat(latitude, len(directions))
        nodes[:, 1 : self.d + 1] = (
            np.repeat(tangent_radius, len(directions))[:, None]
            * np.tile(directions, (resolution, 1))
        )
        weights = np.repeat(latitude_weights / len(directions), len(directions))
        return nodes, weights

    def normal_basis_at(self, parameter):
        point = np.asarray(parameter, dtype=float)
        if self.torus:
            output = np.zeros((4, 2))
            output[:2, 0] = point[:2] / np.linalg.norm(point[:2])
            output[2:, 1] = point[2:] / np.linalg.norm(point[2:])
            return output
        output = np.zeros((self.D, self.q))
        output[: self.d + 1, 0] = point[: self.d + 1] / np.linalg.norm(point[: self.d + 1])
        if self.q > 1:
            output[self.d + 1 :, 1:] = np.eye(self.q - 1)
        return output

    def distances(self, points: np.ndarray):
        points = np.asarray(points, dtype=float)
        if self.torus:
            radius = 1.0 / np.sqrt(2.0)
            return np.hypot(
                np.linalg.norm(points[..., :2], axis=-1) - radius,
                np.linalg.norm(points[..., 2:], axis=-1) - radius,
            )
        return np.sqrt(
            (np.linalg.norm(points[..., : self.d + 1], axis=-1) - 1.0) ** 2
            + np.sum(points[..., self.d + 1 :] ** 2, axis=-1)
        )


@dataclass(frozen=True)
class CorrugatedTorusGeometry:
    key: str = "corrugated_torus_r3"
    label: str = r"Corrugated torus $\subset\mathbb{R}^3$"
    d: int = 2
    D: int = 3

    @property
    def q(self):
        return 1

    def query_grid(self, count: int, phase: float = 0.0):
        del phase
        v = corrugated_equal_area_v(int(count))
        u = np.zeros_like(v)
        points = corrugated_torus(u, v)
        return points, points.copy()

    def integration_node_count(self, resolution: int):
        return int(resolution) ** 2

    def latent_rule(self, resolution: int):
        resolution = int(resolution)
        u = TAU * (np.arange(resolution, dtype=float) + 0.5) / resolution
        v = corrugated_equal_area_v(resolution)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        nodes = corrugated_torus(uu.ravel(), vv.ravel())
        return nodes, np.full(len(nodes), 1.0 / len(nodes))

    def _closest_parameters(self, points):
        value = np.asarray(points, dtype=float)
        flat = value.reshape(-1, 3)
        u = np.mod(np.arctan2(flat[:, 1], flat[:, 0]), TAU)
        radial = np.hypot(flat[:, 0], flat[:, 1])
        best = [_CORRUGATED_MERIDIAN.closest(np.asarray([rr, zz])) for rr, zz in zip(radial, flat[:, 2])]
        squared = np.asarray([row[0] for row in best])
        v = np.asarray([row[1] for row in best])
        return u.reshape(value.shape[:-1]), v.reshape(value.shape[:-1]), squared.reshape(value.shape[:-1])

    def normal_basis_at(self, parameter):
        point = np.asarray(parameter, dtype=float)
        u, v, _ = self._closest_parameters(point)
        xu, xv = corrugated_derivatives(u, v)
        normal = np.cross(xu, xv)
        normal /= np.linalg.norm(normal)
        return normal.reshape(3, 1)

    def distances(self, points):
        return np.sqrt(np.maximum(self._closest_parameters(points)[2], 0.0))


GEOMETRIES = {
    geometry.key: geometry
    for geometry in (
        Geometry("circle_r2", r"$S^1\subset\mathbb{R}^2$", 1, 2),
        Geometry("sphere2_r3", r"$S^2\subset\mathbb{R}^3$", 2, 3),
        Geometry("clifford_torus_r4", r"Clifford torus $\subset\mathbb{R}^4$", 2, 4),
        Geometry("sphere3_r4", r"$S^3\subset\mathbb{R}^4$", 3, 4),
        Geometry("sphere2_r10", r"$S^2\subset\mathbb{R}^{10}$", 2, 10),
        CorrugatedTorusGeometry(),
    )
}
