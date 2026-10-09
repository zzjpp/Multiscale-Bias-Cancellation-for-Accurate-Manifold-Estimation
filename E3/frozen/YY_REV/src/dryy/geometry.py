from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


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
            angle = rng.uniform(0.0, 2.0 * np.pi, int(n))
            return np.column_stack((np.cos(angle), np.sin(angle)))
        if self.torus:
            uv = rng.uniform(0.0, 2.0 * np.pi, (int(n), 2))
            radius = 1.0 / np.sqrt(2.0)
            return radius * np.column_stack(
                (np.cos(uv[:, 0]), np.sin(uv[:, 0]), np.cos(uv[:, 1]), np.sin(uv[:, 1]))
            )
        active = rng.standard_normal((int(n), self.d + 1))
        active /= np.linalg.norm(active, axis=1)[:, None]
        out = np.zeros((int(n), self.D))
        out[:, : self.d + 1] = active
        return out

    def query_grid(self, count: int, query_seed: int) -> np.ndarray:
        count = int(count)
        if self.key == "circle_r2":
            angle = 2.0 * np.pi * np.arange(count) / count
            return np.column_stack((np.cos(angle), np.sin(angle)))
        if self.torus:
            side = int(round(np.sqrt(count)))
            if side * side != count:
                raise ValueError("torus query count must be a square")
            angle = 2.0 * np.pi * np.arange(side) / side
            u, v = np.meshgrid(angle, angle, indexing="ij")
            radius = 1.0 / np.sqrt(2.0)
            return radius * np.column_stack(
                (np.cos(u.ravel()), np.sin(u.ravel()), np.cos(v.ravel()), np.sin(v.ravel()))
            )
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(query_seed), self.id])))
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
            radius = 1.0 / np.sqrt(2.0)
            return np.hypot(
                np.linalg.norm(y[..., :2], axis=-1) - radius,
                np.linalg.norm(y[..., 2:], axis=-1) - radius,
            )
        return np.sqrt(
            (np.linalg.norm(y[..., : self.d + 1], axis=-1) - 1.0) ** 2
            + np.sum(y[..., self.d + 1 :] ** 2, axis=-1)
        )

    def preliminary(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """The supplied O(sigma) preliminary graph from the audited YY E1."""
        x = np.asarray(x, dtype=float)
        if self.q == 1:
            return x + float(sigma) * x
        basis = self.normal_basis(x)
        if self.torus:
            amplitude = 1.0 + 0.2 * x[0] * np.sqrt(2.0)
            direction = (basis[:, 0] + 2.0 * basis[:, 1]) / np.sqrt(5.0)
        else:
            amplitude = 1.0 + 0.2 * x[0]
            direction = (basis[:, 0] + basis[:, 1]) / np.sqrt(2.0)
        return x + float(sigma) * amplitude * direction


def geometries_from_config(config: dict) -> tuple[Geometry, ...]:
    return tuple(Geometry(**row) for row in config["geometries"])


def noisy_sample(config: dict, geometry: Geometry, repeat: int, n: int) -> tuple[np.ndarray, dict]:
    common = [int(config["base_seed"]), int(config["experiment_id"]), geometry.id, int(repeat)]
    clean_seed = common + [0]
    noise_seed = common + [1]
    clean_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(clean_seed)))
    noise_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(noise_seed)))
    clean = geometry.sample_clean(clean_rng, int(n))
    noise = noise_rng.standard_normal(clean.shape)
    observed = clean + float(config["sigma"]) * noise
    return observed, {
        "algorithm": "PCG64DXSM",
        "clean_seed": clean_seed,
        "noise_seed": noise_seed,
        "observed_sha256": array_hash(observed),
    }

