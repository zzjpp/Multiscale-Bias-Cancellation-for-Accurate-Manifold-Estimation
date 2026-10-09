from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def two_scale_weights(coefficients):
    c1, c2 = map(float, coefficients)
    if not np.isfinite(c1 + c2) or c1 == c2:
        raise ValueError("two distinct finite coefficients are required")
    weights = np.array([c2, -c1], dtype=float) / (c2 - c1)
    return weights


def dr_coefficient(d, h, sigma):
    return 0.5 * float(d) * (float(h) ** 2 + float(sigma) ** 2)


def stsp_moments(d, D, lam, p=6):
    q = int(D) - int(d)
    if not (1 <= d < D) or lam <= 0:
        raise ValueError("invalid STSP moment parameters")

    def base(radius):
        return (radius ** (q - 1) * max(1.0 - radius**2, 0.0) ** (p + 0.5 * d)
                * np.exp(-radius**2 / (2.0 * lam**2)))

    options = dict(epsabs=2e-13, epsrel=2e-13, limit=300)
    mass = quad(base, 0.0, 1.0, **options)[0]
    normal_second = quad(lambda r: r**2 * base(r), 0.0, 1.0, **options)[0]
    mixed = quad(lambda r: r**2 * (1.0-r**2) * base(r), 0.0, 1.0, **options)[0]
    a_lam = normal_second / (q * mass)
    c_lam = mixed / (q * (d + 2*p + 2) * mass)
    return a_lam, c_lam


def stsp_coefficient(d, D, h, sigma, p=6):
    lam = float(sigma) / float(h)
    a_lam, c_lam = stsp_moments(d, D, lam, p)
    return 0.5 * float(d) * float(h)**2 * (lam**2 + c_lam/a_lam)


def yy_alpha_beta(d, q, kappa, p=6):
    m = int(q) - 1
    J = int(d) + 2*int(p) + 2
    if m == 0:
        return float(kappa)**2/J, None

    def base(radius, extra):
        return (radius ** (m-1+extra)
                * max(1.0-radius**2/float(kappa)**2, 0.0)**(p+0.5*d)
                * np.exp(-0.5*radius**2))

    options = dict(epsabs=2e-13, epsrel=2e-13, limit=300)
    i0 = quad(lambda r: base(r, 0), 0.0, float(kappa), **options)[0]
    i2 = quad(lambda r: base(r, 2), 0.0, float(kappa), **options)[0]
    beta = i2/(m*i0)
    alpha = (float(kappa)**2-m*beta)/J
    return alpha, beta


def yy_weights(d, q, radii, sigma, p=6):
    kappas = np.asarray(radii, dtype=float)/float(sigma)
    moments = [yy_alpha_beta(d, q, k, p) for k in kappas]
    alpha = np.array([row[0] for row in moments])
    beta = None if q == 1 else np.array([row[1] for row in moments])
    axial_matrix = np.vstack((np.ones(len(kappas)), 1.0+alpha))
    axial = np.linalg.lstsq(axial_matrix, np.array([1.0, 0.0]), rcond=None)[0]
    if q == 1:
        complementary_matrix = axial_matrix
        complementary = axial.copy()
    else:
        complementary_matrix = np.vstack((np.ones(len(kappas)), alpha, 1.0-beta))
        complementary = np.linalg.solve(complementary_matrix, np.array([1.0, 0.0, 0.0]))
    return {
        "kappas": kappas,
        "alpha": alpha,
        "beta": beta,
        "axial": axial,
        "complementary": complementary,
        "axial_constraint_residual": axial_matrix@axial-np.array([1.0, 0.0]),
        "complementary_constraint_residual": (
            complementary_matrix@complementary
            - (np.array([1.0, 0.0]) if q == 1 else np.array([1.0, 0.0, 0.0]))
        ),
    }
