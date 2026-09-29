from __future__ import annotations

import numpy as np


def polynomial_cutoff_from_squared(squared_radius, radius: float, p: int):
    """(1-s^2/r^2)^p_+ from Eq. (10)/(11)."""

    if radius <= 0 or p < 4:
        raise ValueError("the theory requires radius>0 and p>=4")
    value = 1.0 - np.asarray(squared_radius, dtype=float) / float(radius) ** 2
    return np.maximum(value, 0.0) ** int(p)


def eta_c3(unit_interval):
    """Polynomial eta in Eq. (58), evaluated for u in [0,1]."""

    u = np.asarray(unit_interval, dtype=float)
    return 1.0 - 35.0 * u**4 + 84.0 * u**5 - 70.0 * u**6 + 20.0 * u**7


def longitudinal_profile(normalized_radius):
    """The exact C^3 plateau profile k from Eq. (58)."""

    t = np.asarray(normalized_radius, dtype=float)
    result = np.zeros_like(t)
    result[t <= 0.5] = 1.0
    transition = (t > 0.5) & (t < 1.0)
    result[transition] = eta_c3(2.0 * t[transition] - 1.0)
    return result
