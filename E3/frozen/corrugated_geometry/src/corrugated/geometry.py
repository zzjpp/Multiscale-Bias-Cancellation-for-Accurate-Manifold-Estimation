from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree
from .config import CONFIG

TAU = 2.0 * np.pi

def minor_radius(v):
    v = np.asarray(v, dtype=np.float64)
    return 0.7 * (1.0 + 0.18 * np.cos(3.0 * v) + 0.10 * np.sin(2.0 * v))

def r_prime(v):
    v = np.asarray(v, dtype=np.float64)
    return 0.7 * (-0.54 * np.sin(3.0 * v) + 0.20 * np.cos(2.0 * v))

def rho(v):
    v = np.asarray(v, dtype=np.float64)
    return CONFIG["R"] + minor_radius(v) * np.cos(v)

def zeta(v):
    v = np.asarray(v, dtype=np.float64)
    return minor_radius(v) * np.sin(v)

def area_jacobian(v):
    v = np.asarray(v, dtype=np.float64)
    return rho(v) * np.sqrt(minor_radius(v) ** 2 + r_prime(v) ** 2)

def corrugated_torus(u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    radial = rho(v)
    return np.stack((radial * np.cos(u), radial * np.sin(u), zeta(v)), axis=-1).astype(np.float64, copy=False)

def surface_derivatives(u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    rr, rp = minor_radius(v), r_prime(v)
    radial, radial_p = rho(v), rp * np.cos(v) - rr * np.sin(v)
    z_p = rp * np.sin(v) + rr * np.cos(v)
    xu = np.stack((-radial * np.sin(u), radial * np.cos(u), np.zeros_like(u)), axis=-1)
    xv = np.stack((radial_p * np.cos(u), radial_p * np.sin(u), z_p), axis=-1)
    return xu, xv

@lru_cache(maxsize=2)
def area_cdf(intervals: int | None = None):
    intervals = int(intervals or CONFIG["area_cdf_intervals"])
    grid = np.linspace(0.0, TAU, intervals + 1, dtype=np.float64)
    weights = area_jacobian(grid)
    cumulative = np.concatenate(([0.0], cumulative_trapezoid(weights, grid)))
    cumulative /= cumulative[-1]
    return grid, cumulative

def inverse_area_cdf(q):
    grid, cumulative = area_cdf()
    q = np.asarray(q, dtype=np.float64)
    return np.interp(q, cumulative, grid).astype(np.float64, copy=False)

def sample_parameters(n: int, rng: np.random.Generator):
    u = rng.uniform(0.0, TAU, int(n))
    v = inverse_area_cdf(rng.uniform(0.0, 1.0, int(n)))
    return u, v

def sample_corrugated_torus(n: int, rng: np.random.Generator) -> np.ndarray:
    u, v = sample_parameters(n, rng)
    return corrugated_torus(u, v)

def tensor_query_set(n_u: int, n_v: int):
    u = TAU * (np.arange(n_u, dtype=np.float64) + 0.5) / n_u
    q = (np.arange(n_v, dtype=np.float64) + 0.5) / n_v
    v = inverse_area_cdf(q)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    return uv, corrugated_torus(uv[:, 0], uv[:, 1])

class MeridianOptimizer:
    def __init__(self, count: int | None = None):
        self.count = int(count or CONFIG["distance_coarse_grid"])
        self.step = TAU / self.count
        self.v = TAU * np.arange(self.count, dtype=np.float64) / self.count
        self.points = np.column_stack((rho(self.v), zeta(self.v)))
        self.tree = cKDTree(self.points)

    def closest(self, meridian_point: np.ndarray) -> tuple[float, float]:
        target = np.asarray(meridian_point, dtype=np.float64)
        _, indices = self.tree.query(target, k=8)
        candidates = []
        selected = []
        for index in np.atleast_1d(indices):
            index = int(index)
            if any(min(abs(index-old), self.count-abs(index-old)) <= 4 for old in selected):
                continue
            selected.append(index)
        for index in selected:
            center = self.v[int(index)]
            objective = lambda value: (rho(value % TAU) - target[0]) ** 2 + (zeta(value % TAU) - target[1]) ** 2
            result = minimize_scalar(objective, bounds=(center - 2 * self.step, center + 2 * self.step), method="bounded", options={"xatol": 1e-14})
            residual = lambda value: np.array([rho(value[0] % TAU) - target[0], zeta(value[0] % TAU) - target[1]])
            refined = least_squares(residual, np.array([result.x]), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=50)
            refined_v = float(refined.x[0] % TAU)
            candidates.append((float(objective(refined_v)), refined_v))
        return min(candidates)

_MERIDIAN = MeridianOptimizer()

def closest_parameters(points: np.ndarray):
    value = np.asarray(points, dtype=np.float64)
    flat = value.reshape(-1, 3)
    u = np.mod(np.arctan2(flat[:, 1], flat[:, 0]), TAU)
    q = np.hypot(flat[:, 0], flat[:, 1])
    best = [_MERIDIAN.closest(np.array([qq, zz])) for qq, zz in zip(q, flat[:, 2])]
    squared = np.asarray([row[0] for row in best])
    v = np.asarray([row[1] for row in best])
    return u.reshape(value.shape[:-1]), v.reshape(value.shape[:-1]), squared.reshape(value.shape[:-1])

def distance_to_corrugated_torus(points: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(closest_parameters(points)[2], 0.0))

@dataclass
class CorrugatedTorusGeometry:
    id: int = 7
    key: str = "corrugated_torus_r3"
    label: str = "Corrugated torus in R^3"
    d: int = 2
    D: int = 3
    _normal_cache: dict[bytes, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def q(self): return 1
    @property
    def torus(self): return False
    def sample_clean(self, rng, n): return sample_corrugated_torus(n, rng)
    def query_grid(self, count=512):
        shapes = {128:(16,8), 256:(32,8), 512:(32,16), 1024:(64,16), 65536:(256,256)}
        if count not in shapes: raise ValueError(f"unsupported deterministic query count {count}")
        return tensor_query_set(*shapes[count])[1]
    def distance(self, y): return distance_to_corrugated_torus(y)
    def project(self, y):
        value = np.asarray(y, dtype=np.float64)
        u, v, _ = closest_parameters(value)
        return corrugated_torus(u, v)
    def normal_basis(self, x):
        x = np.asarray(x, dtype=np.float64)
        key = x.tobytes()
        if key not in self._normal_cache:
            u, v, _ = closest_parameters(x)
            xu, xv = surface_derivatives(u, v)
            normal = np.cross(xu, xv)
            normal /= np.linalg.norm(normal)
            self._normal_cache[key] = normal.reshape(3, 1)
        return self._normal_cache[key].copy()

GEOMETRY = CorrugatedTorusGeometry()
