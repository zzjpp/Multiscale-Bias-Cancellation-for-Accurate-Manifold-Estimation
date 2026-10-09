from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import CONFIG
from .records import array_hash


@dataclass(frozen=True)
class Geometry:
    id: int
    key: str
    label: str
    d: int
    D: int

    @property
    def q(self) -> int:
        return self.D - self.d

    @property
    def torus(self) -> bool:
        return self.key == "clifford_torus_r4"

    def sample_clean(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.key == "circle_r2":
            angle = rng.uniform(0.0, 2 * np.pi, n)
            return np.column_stack((np.cos(angle), np.sin(angle)))
        if self.torus:
            uv = rng.uniform(0.0, 2 * np.pi, (n, 2))
            r = 1 / np.sqrt(2)
            return r * np.column_stack((np.cos(uv[:, 0]), np.sin(uv[:, 0]), np.cos(uv[:, 1]), np.sin(uv[:, 1])))
        active = rng.standard_normal((n, self.d + 1))
        active /= np.linalg.norm(active, axis=1)[:, None]
        out = np.zeros((n, self.D))
        out[:, : self.d + 1] = active
        return out

    def query_grid(self, count: int = 256) -> np.ndarray:
        if self.key == "circle_r2":
            angle = 2 * np.pi * np.arange(count) / count
            return np.column_stack((np.cos(angle), np.sin(angle)))
        if self.torus:
            side = int(round(np.sqrt(count)))
            if side * side != count:
                raise ValueError("torus query count must be a square")
            angle = 2 * np.pi * np.arange(side) / side
            u, v = np.meshgrid(angle, angle, indexing="ij")
            r = 1 / np.sqrt(2)
            return r * np.column_stack((np.cos(u.ravel()), np.sin(u.ravel()), np.cos(v.ravel()), np.sin(v.ravel())))
        # One deterministic infinite-prefix sequence per sphere geometry.
        seed = np.random.SeedSequence([CONFIG["query_seed"], self.id])
        rng = np.random.Generator(np.random.PCG64(seed))
        active = rng.standard_normal((count, self.d + 1))
        active /= np.linalg.norm(active, axis=1)[:, None]
        out = np.zeros((count, self.D))
        out[:, : self.d + 1] = active
        return out

    def normal_basis(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.torus:
            out = np.zeros((4, 2))
            out[:2, 0] = x[:2] / np.linalg.norm(x[:2])
            out[2:, 1] = x[2:] / np.linalg.norm(x[2:])
            return out
        out = np.zeros((self.D, self.q))
        out[: self.d + 1, 0] = x[: self.d + 1] / np.linalg.norm(x[: self.d + 1])
        if self.q > 1:
            out[self.d + 1 :, 1:] = np.eye(self.q - 1)
        return out

    def distance(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.torus:
            r = 1 / np.sqrt(2)
            return np.hypot(np.linalg.norm(y[..., :2], axis=-1) - r, np.linalg.norm(y[..., 2:], axis=-1) - r)
        return np.sqrt((np.linalg.norm(y[..., : self.d + 1], axis=-1) - 1) ** 2 + np.sum(y[..., self.d + 1 :] ** 2, axis=-1))

    def project(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.torus:
            out = y.copy()
            r = 1 / np.sqrt(2)
            out[:2] = r * y[:2] / np.linalg.norm(y[:2])
            out[2:] = r * y[2:] / np.linalg.norm(y[2:])
            return out
        out = np.zeros(self.D)
        active = y[: self.d + 1]
        out[: self.d + 1] = active / np.linalg.norm(active)
        return out


GEOMETRIES = tuple(Geometry(**row) for row in CONFIG["geometries"])
BY_KEY = {geometry.key: geometry for geometry in GEOMETRIES}


def seed_lineage(geometry: Geometry, repeat: int) -> dict:
    if repeat not in range(CONFIG["repeats"]):
        raise ValueError("repeat out of range")
    common = [CONFIG["base_seed"], CONFIG["experiment_id"], geometry.id, repeat]
    return {"algorithm": "PCG64DXSM", "seed_sequence": "numpy.random.SeedSequence", "clean": common + [0], "noise": common + [1], "repeat_index_base": 0}


def noisy_sample(geometry: Geometry, repeat: int, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = CONFIG["n"] if n is None else int(n)
    lineage = seed_lineage(geometry, repeat)
    clean_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(lineage["clean"])))
    noise_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(lineage["noise"])))
    clean = geometry.sample_clean(clean_rng, n)
    noise = noise_rng.standard_normal((n, geometry.D))
    observed = clean + CONFIG["sigma"] * noise
    metadata = dict(lineage, n=n, clean_sha256=array_hash(clean), noise_sha256=array_hash(noise), observed_sha256=array_hash(observed))
    return clean, noise, observed, metadata
