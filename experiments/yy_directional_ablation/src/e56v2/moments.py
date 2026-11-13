from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class DirectionalMoment:
    kappa: float
    alpha: float
    beta: float
    mass_integral: float
    radial_second_integral: float
    mass_error: float
    radial_second_error: float


def directional_moment(kappa: float, d: int, p: int, m: int) -> DirectionalMoment:
    """Compute alpha_m(kappa), beta_m(kappa) from the law in Eq. (14)."""

    if kappa <= 0 or d < 1 or p < 4 or m < 1:
        raise ValueError("require kappa>0, d>=1, p>=4, and m>=1")
    kappa = float(kappa)
    exponent = float(p) + 0.5 * float(d)

    def radial_density(radius: float) -> float:
        remaining = max(1.0 - (radius / kappa) ** 2, 0.0)
        return radius ** (m - 1) * remaining**exponent * np.exp(-0.5 * radius**2)

    mass, mass_error = quad(radial_density, 0.0, kappa, epsabs=2e-13, epsrel=2e-13, limit=300)
    radial_second, radial_second_error = quad(
        lambda radius: radius**2 * radial_density(radius),
        0.0,
        kappa,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=300,
    )
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(radial_second):
        raise ArithmeticError("directional moment quadrature failed")
    beta = radial_second / (float(m) * mass)
    alpha = (kappa**2 - float(m) * beta) / float(d + 2 * p + 2)
    return DirectionalMoment(
        kappa=kappa,
        alpha=float(alpha),
        beta=float(beta),
        mass_integral=float(mass),
        radial_second_integral=float(radial_second),
        mass_error=float(mass_error),
        radial_second_error=float(radial_second_error),
    )


def directional_moments(kappas: Iterable[float], d: int, p: int, m: int) -> list[DirectionalMoment]:
    values = tuple(float(value) for value in kappas)
    if len(values) < 3 or len(set(values)) != len(values):
        raise ValueError("need at least three distinct transverse ratios")
    return [directional_moment(value, d=d, p=p, m=m) for value in values]
