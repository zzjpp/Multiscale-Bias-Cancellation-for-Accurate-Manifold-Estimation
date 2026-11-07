from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi


TAU = 2.0 * np.pi


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


GEOMETRIES = {
    geometry.key: geometry
    for geometry in (
        Geometry("circle_r2", r"$S^1\subset\mathbb{R}^2$", 1, 2),
        Geometry("sphere2_r3", r"$S^2\subset\mathbb{R}^3$", 2, 3),
        Geometry("clifford_torus_r4", r"Clifford torus $\subset\mathbb{R}^4$", 2, 4),
        Geometry("sphere3_r4", r"$S^3\subset\mathbb{R}^4$", 3, 4),
        Geometry("sphere2_r10", r"$S^2\subset\mathbb{R}^{10}$", 2, 10),
    )
}
