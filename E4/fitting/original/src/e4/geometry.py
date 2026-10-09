from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline

TAU = 2.0 * np.pi


def gamma(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((1.5 * np.cos(t), np.sin(t), 0.3 * np.sin(2 * t)), axis=-1)


def gamma_prime(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5 * np.sin(t), np.cos(t), 0.6 * np.cos(2 * t)), axis=-1)


def gamma_second(theta):
    t = np.asarray(theta, dtype=float)
    return np.stack((-1.5 * np.cos(t), -np.sin(t), -1.2 * np.sin(2 * t)), axis=-1)


def tangent(theta):
    gp = gamma_prime(theta)
    return gp / np.linalg.norm(gp, axis=-1, keepdims=True)


def normal_projector(theta):
    T = tangent(theta)
    eye = np.eye(3)
    return eye - np.einsum("...i,...j->...ij", T, T)


def curvature_vector(theta):
    gp, gpp = gamma_prime(theta), gamma_second(theta)
    speed2 = np.sum(gp * gp, axis=-1, keepdims=True)
    return gpp / speed2 - gp * (np.sum(gp * gpp, axis=-1, keepdims=True) / speed2**2)


def normal_basis(theta) -> np.ndarray:
    """Deterministic 3x2 basis for the true normal plane."""
    T = np.asarray(tangent(theta), dtype=float)
    if T.shape != (3,):
        raise ValueError("scalar theta required")
    axis = np.array([0.0, 0.0, 1.0]) if abs(T[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = axis - T * (axis @ T)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(T, e1)
    e2 /= np.linalg.norm(e2)
    return np.column_stack((e1, e2))


@dataclass(frozen=True)
class ArcTable:
    theta: np.ndarray
    length: np.ndarray
    total: float
    nodes: int

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(np.asarray(self.theta, dtype="<f8").tobytes())
        h.update(np.asarray(self.length, dtype="<f8").tobytes())
        return h.hexdigest()

    def invert(self, normalized):
        u = np.mod(np.asarray(normalized, dtype=float), 1.0)
        # Cubic inverse interpolation is required for the 65,537 -> 131,073
        # coordinate-refinement contract; linear inversion stalls near 1e-9.
        return CubicSpline(self.length, self.theta)(u * self.total)


def arc_table(nodes: int = 65537) -> ArcTable:
    if nodes < 3 or (nodes - 1) & (nodes - 2):
        # The production values are 2^k+1. Permit other values, but make mistakes visible.
        pass
    theta = np.linspace(0.0, TAU, int(nodes))
    speed = np.linalg.norm(gamma_prime(theta), axis=1)
    length = cumulative_simpson(speed, x=theta, initial=0.0)
    return ArcTable(theta, length, float(length[-1]), int(nodes))


def query_grid(count: int, table: ArcTable, offset: float = 0.37):
    u = (np.arange(count, dtype=float) + offset) / count
    theta = table.invert(u)
    return {"normalized_arclength": u, "theta": theta, "coordinates": gamma(theta)}


def sample_arclength(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    accepted = []
    total = 0
    bound = np.sqrt(2.61)
    while total < n:
        size = max(1024, int((n - total) * 1.8))
        theta = rng.uniform(0.0, TAU, size)
        keep = rng.random(size) <= np.linalg.norm(gamma_prime(theta), axis=1) / bound
        batch = theta[keep]
        accepted.append(batch)
        total += len(batch)
    theta = np.concatenate(accepted)[:n]
    return theta, gamma(theta)


def project_to_curve(y, coarse_nodes: int = 4096, xtol: float = 5e-15, rtol: float = 1e-14):
    """Global periodic point-to-C3 projection using every coarse local basin."""
    y = np.asarray(y, dtype=float)
    grid = TAU * np.arange(coarse_nodes) / coarse_nodes
    pts = gamma(grid)
    d2 = np.sum((pts - y) ** 2, axis=1)
    local = np.flatnonzero((d2 <= np.roll(d2, 1)) & (d2 <= np.roll(d2, -1)))

    def stationarity(t):
        return float(2 * ((gamma(t) - y) @ gamma_prime(t)))

    roots = []
    step = TAU / coarse_nodes
    for idx in local:
        center = grid[idx]
        lo, hi = center - step, center + step
        flo, fhi = stationarity(lo), stationarity(hi)
        if flo == 0:
            root = lo
        elif fhi == 0:
            root = hi
        elif flo * fhi < 0:
            root = brentq(stationarity, lo, hi, xtol=xtol, rtol=rtol)
        else:
            # Rare flat/discretization basin: subdivide, retaining only bracketed roots.
            sub = np.linspace(lo, hi, 17)
            vals = [stationarity(v) for v in sub]
            brackets = [(sub[k], sub[k + 1]) for k in range(16) if vals[k] * vals[k + 1] <= 0]
            if not brackets:
                continue
            a, b = min(brackets, key=lambda ab: abs((ab[0] + ab[1]) / 2 - center))
            root = brentq(stationarity, a, b, xtol=xtol, rtol=rtol)
        roots.append(root % TAU)
    if not roots:
        raise RuntimeError("curve projection produced no stationary candidate")
    candidates = np.asarray(roots)
    distances = np.linalg.norm(gamma(candidates) - y, axis=1)
    k = int(np.argmin(distances))
    return {"theta": float(candidates[k]), "point": gamma(candidates[k]), "distance": float(distances[k]), "candidate_count": len(candidates)}


def curve_distances(points, coarse_nodes: int = 4096):
    return np.asarray([project_to_curve(point, coarse_nodes=coarse_nodes)["distance"] for point in np.asarray(points)])


def projector_variation(theta):
    Q = normal_projector(theta)
    return np.linalg.norm(Q - np.roll(Q, 1, axis=0), axis=(1, 2))
