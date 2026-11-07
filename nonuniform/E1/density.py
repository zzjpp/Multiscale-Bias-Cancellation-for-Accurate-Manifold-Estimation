"""Predeclared nonuniform surface measures and valid symmetry reductions."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from e1_theory_path.geometry import (GEOMETRIES, TAU, corrugated_area_jacobian,
                                    corrugated_equal_area_v, corrugated_torus)


@lru_cache(maxsize=8)
def corrugated_constants(amplitude=.5):
    options = dict(epsabs=2e-13, epsrel=2e-13, limit=300)
    area, area_error = quad(lambda v: float(corrugated_area_jacobian(v)), 0, TAU, **options)
    mass, mass_error = quad(lambda v: float((1+amplitude*np.cos(v))*corrugated_area_jacobian(v)), 0, TAU, **options)
    return dict(area=TAU*area, normalizer=TAU*mass,
                relative_uniform_normalizer=mass/area,
                quadrature_error=TAU*max(area_error, mass_error))


def sphere_rotation(point, d):
    """Quadrature coordinate frame only; no fitting normal is returned."""
    direction = np.asarray(point, dtype=float)[:d+1]
    direction = direction/np.linalg.norm(direction)
    e0 = np.eye(d+1)[0]
    v = e0-direction
    if np.dot(v, v) < 1e-28:
        return np.eye(d+1)
    return np.eye(d+1)-2*np.outer(v, v)/np.dot(v, v)


@lru_cache(maxsize=3)
def fixed_weighted_rule(key, resolution, amplitude):
    base = GEOMETRIES[key]
    nodes, weights = base.latent_rule(resolution)
    if key == 'circle_r2':
        factor = 1+amplitude*nodes[:, 0]
    elif key == 'clifford_torus_r4':
        factor = 1+amplitude*np.sqrt(2)*nodes[:, 0]
    elif key == 'corrugated_torus_r3':
        v = np.tile(corrugated_equal_area_v(resolution), resolution)
        factor = (1+amplitude*np.cos(v))/corrugated_constants(amplitude)['relative_uniform_normalizer']
    else:
        raise ValueError(key)
    return nodes, weights*factor


class DensityGeometry:
    def __init__(self, key, amplitude=.5):
        if not 0 <= amplitude < 1:
            raise ValueError('Strictly positive smooth density required')
        self.base = GEOMETRIES[key]
        self.amplitude = float(amplitude)

    def __getattr__(self, name):
        if name == 'normal_basis_at':
            raise AssertionError('True geometry normal is forbidden in fitting')
        return getattr(self.base, name)

    @property
    def spherical(self):
        return self.key.startswith('sphere')

    def oriented_rule(self, resolution, point):
        if not self.spherical:
            return self.latent_rule(resolution)
        canonical, weights = self.base.latent_rule(resolution)
        nodes = canonical.copy()
        rotation = sphere_rotation(point, self.d)
        nodes[:, :self.d+1] = canonical[:, :self.d+1] @ rotation.T
        return nodes, weights*(1+self.amplitude*nodes[:, 0])

    def latent_rule(self, resolution):
        if self.spherical:
            point = np.eye(self.D)[0]
            return self.oriented_rule(resolution, point)
        return fixed_weighted_rule(self.key, int(resolution), self.amplitude)

    def query_grid(self, count, phase=0):
        if self.key == 'corrugated_torus_r3':
            return self.base.query_grid(count, phase)
        if self.spherical:
            theta = np.linspace(0, np.pi, int(count))
            points = np.zeros((int(count), self.D))
            points[:, 0] = np.cos(theta)
            points[:, 1] = np.sin(theta)
        elif self.key == 'circle_r2':
            theta = TAU*np.arange(count)/count
            points = np.column_stack((np.cos(theta), np.sin(theta)))
        else:
            theta = TAU*np.arange(count)/count
            r = 1/np.sqrt(2)
            points = r*np.column_stack((np.cos(theta), np.sin(theta), np.ones(count), np.zeros(count)))
        return points, points.copy()


def cpu_operator(geometry, h, sigma, resolution, radial, padding, power=6):
    from e1_theory_path.population import PopulationSTSP

    class WeightedCPU(PopulationSTSP):
        def evaluate(self, point):
            if self.geometry.spherical:
                self.latent, self.latent_weights = self.geometry.oriented_rule(resolution, point)
            return super().evaluate(point)

    return WeightedCPU(geometry, h, sigma, resolution, radial, power,
                       noncentral_radius_padding=padding)


def gpu_operator(geometry, h, sigma, resolution, radial, padding, device, power=6):
    from fast_gpu import FastGpuPopulationSTSP

    class WeightedGPU(FastGpuPopulationSTSP):
        def evaluate(self, point):
            if self.geometry.spherical:
                nodes, weights = self.geometry.oriented_rule(resolution, point)
                self.latent = self.cp.asarray(nodes, dtype=self.cp.float64)
                self.latent_weights = self.cp.asarray(weights, dtype=self.cp.float64)
            return super().evaluate(point)

    return WeightedGPU(geometry, h, sigma, resolution, radial, power,
                       noncentral_radius_padding=padding, device=device)
