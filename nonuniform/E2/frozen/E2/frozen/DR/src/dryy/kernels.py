from __future__ import annotations

import numpy as np


def polynomial_cutoff(squared_radius, radius: float, power: int = 6) -> np.ndarray:
    if radius <= 0 or power < 4:
        raise ValueError("invalid smooth-kernel parameters")
    return np.maximum(1.0 - np.asarray(squared_radius, dtype=float) / float(radius) ** 2, 0.0) ** int(power)


def longitudinal_profile(normalized_radius) -> np.ndarray:
    t = np.asarray(normalized_radius, dtype=float)
    out = np.zeros_like(t)
    out[t <= 0.5] = 1.0
    mask = (t > 0.5) & (t < 1.0)
    u = 2.0 * t[mask] - 1.0
    out[mask] = 1.0 - 35.0 * u**4 + 84.0 * u**5 - 70.0 * u**6 + 20.0 * u**7
    return out


def effective_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    square = float(weights @ weights)
    return 0.0 if square <= 0 else total * total / square

