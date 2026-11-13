"""Authorized new finite geometry; E1 formulas/distance are imported unchanged.

Uniform surface-area observations and deterministic 12x12 area-balanced queries.
True normals are available ONLY for YY's prescribed O(sigma) initialization;
STSP/DR matching is prohibited from consulting them.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'frozen'))
import corrugated_e1_geometry as canonical

N_VALUES = (2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000)

def coordinates_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()

def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode('ascii'))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class CorrugatedGeometry:
    allow_prescribed_normal: bool = False
    id: int = 6
    key: str = 'corrugated_torus_r3'
    label: str = 'Corrugated torus subset R^3'
    d: int = 2
    D: int = 3
    q: int = 1
    torus: bool = True  # E2 STSP reverse reference uses a square surface grid.

    def sample_clean(self, rng, n):
        uv = rng.uniform(0., 1., (int(n), 2))
        grid, cumulative = canonical.corrugated_area_cdf()
        v = np.interp(uv[:, 1], cumulative, grid)
        return canonical.corrugated_torus(canonical.TAU * uv[:, 0], v)

    def query_grid(self, count=144, query_seed=None):
        del query_seed  # deterministic, same Cartesian surface grid for all methods.
        side = int(round(np.sqrt(count)))
        if side * side != count:
            raise ValueError('Surface query/reference count must be a perfect square')
        u = canonical.TAU * np.arange(side) / side
        v = canonical.corrugated_equal_area_v(side)
        uu, vv = np.meshgrid(u, v, indexing='ij')
        return canonical.corrugated_torus(uu.ravel(), vv.ravel())

    def distance(self, points):
        return canonical.CorrugatedTorusGeometry().distances(points)

    def project(self, point):
        u, v, _ = canonical.CorrugatedTorusGeometry()._closest_parameters(point)
        return canonical.corrugated_torus(u, v)

    def normal_basis(self, point):
        if not self.allow_prescribed_normal:
            raise AssertionError('STSP/DR fitting attempted to consult a true normal')
        point = np.asarray(point)
        u = np.arctan2(point[1], point[0])
        v = np.arctan2(point[2], np.hypot(point[0], point[1]) - canonical.E7_MAJOR_RADIUS)
        xu, xv = canonical.corrugated_derivatives(u, v)
        normal = np.cross(xu, xv)
        return (normal / np.linalg.norm(normal)).reshape(3, 1)

    def preliminary(self, point, sigma):
        # Unit-normal offset: sphere q=1 rule x+sigma*x generalized to this surface.
        # This is a prescribed known-geometry initialization, not a learned normal.
        return np.asarray(point) + float(sigma) * self.normal_basis(point)[:, 0]

def sample(method, repeat, n=500000):
    if method not in {'STSP', 'DR', 'YY'} or repeat not in range(20):
        raise ValueError('Unauthorized method/repeat')
    geometry = CorrugatedGeometry(method == 'YY')
    experiment_id = 2 if method == 'STSP' else 202
    common = [20260911, experiment_id, 6, repeat]
    clean_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(common + [0])))
    noise_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(common + [1])))
    clean = geometry.sample_clean(clean_rng, n)
    noise = noise_rng.standard_normal(clean.shape)
    observed = clean + .005 * noise
    lineage = dict(algorithm='PCG64DXSM', clean_seed=common+[0], noise_seed=common+[1],
        clean_coordinates_sha256=coordinates_sha(clean), noise_coordinates_sha256=coordinates_sha(noise),
        observed_coordinates_sha256=coordinates_sha(observed), observed_array_sha256=array_sha(observed),
        nested_prefix_coordinates_sha256={str(size): coordinates_sha(observed[:size]) for size in N_VALUES if size <= n})
    return geometry, clean, noise, observed, lineage
