from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import gammaln, roots_legendre

from .population import State


def _cupy():
    try:
        import cupy as cp
        from cupyx.scipy.special import i0e, i1e
    except ImportError as error:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("CuPy with cupyx.scipy.special.i0e/i1e is required") from error
    return cp, i0e, i1e


@lru_cache(maxsize=32)
def _legendre_host(order: int):
    return roots_legendre(int(order))


def _series_scaled_iv(cp, nu: float, x, terms: int = 80):
    """Stable exp(-x) I_nu(x) power series for small nonnegative x."""
    tiny = cp.finfo(cp.float64).tiny
    safe_x = cp.maximum(x, tiny)
    leading = cp.exp(-safe_x + nu * cp.log(safe_x / 2.0) - float(gammaln(nu + 1.0)))
    term = cp.ones_like(safe_x)
    total = term.copy()
    z = safe_x * safe_x / 4.0
    for index in range(1, int(terms)):
        term = term * z / (float(index) * (nu + float(index)))
        total = total + term
    value = leading * total
    if nu == 0.0:
        value = cp.where(x == 0.0, 1.0, value)
    else:
        value = cp.where(x == 0.0, 0.0, value)
    return value


def scaled_iv(cp, i0e, i1e, nu: float, x):
    """exp(-x) I_nu(x) for every integer/half-integer order used in E1."""
    x = cp.asarray(x, dtype=cp.float64)
    small = x < 16.0
    series = _series_scaled_iv(cp, float(nu), x)
    safe = cp.maximum(x, 16.0)
    if nu == 0.0:
        large = i0e(safe)
    elif nu == 1.0:
        large = i1e(safe)
    elif nu == 2.0:
        large = i0e(safe) - 2.0 * i1e(safe) / safe
    elif abs(nu - round(nu)) < 1e-12:
        order = int(round(nu))
        previous = i0e(safe)
        current = i1e(safe)
        if order == 0:
            large = previous
        elif order == 1:
            large = current
        else:
            for index in range(1, order):
                following = previous - (2.0 * index / safe) * current
                previous, current = current, following
            large = current
    else:
        prefactor = cp.sqrt(2.0 / (cp.pi * safe))
        half = prefactor * (-0.5 * cp.expm1(-2.0 * safe))
        minus_half = prefactor * (0.5 * (1.0 + cp.exp(-2.0 * safe)))
        three_half = minus_half - half / safe
        if nu == 0.5:
            large = half
        elif nu == 1.5:
            large = three_half
        elif nu == 2.5:
            large = half - 3.0 * three_half / safe
        else:  # pragma: no cover - current odd ambient dimension is D=3
            raise ValueError(f"unsupported Bessel order: {nu}")
    return cp.where(small, series, large)


def radial_mass_gpu(df: int, noncentral_radius, kappa: float, order: int, power: int = 6, kernel_kind: str = "polynomial"):
    cp, i0e, i1e = _cupy()
    noncentral_radius = cp.asarray(noncentral_radius, dtype=cp.float64).reshape(-1)
    host_nodes, host_weights = _legendre_host(int(order))
    nodes = cp.asarray(host_nodes)
    weights = cp.asarray(host_weights)
    radius = float(kappa) * (nodes + 1.0) / 2.0
    transformed = weights * float(kappa) / 2.0
    if kernel_kind == "polynomial":
        kernel_weights = transformed * cp.maximum(1.0 - (radius / float(kappa)) ** 2, 0.0) ** int(power)
    elif kernel_kind == "truncated_gaussian":
        kernel_weights = transformed * cp.exp(-0.5 * (radius / float(kappa)) ** 2)
    else:
        raise ValueError(kernel_kind)
    answer = cp.empty(len(noncentral_radius), dtype=cp.float64)
    nu = df / 2.0 - 1.0
    for start in range(0, len(noncentral_radius), 4096):
        a = noncentral_radius[start : start + 4096]
        density = cp.empty((len(a), len(radius)), dtype=cp.float64)
        central = a == 0.0
        central_density = cp.exp(
            (df - 1) * cp.log(radius)
            - radius * radius / 2.0
            - (df / 2.0 - 1.0) * np.log(2.0)
            - float(gammaln(df / 2.0))
        )
        b = cp.maximum(a[:, None], cp.finfo(cp.float64).tiny)
        argument = radius[None, :] * b
        bessel = cp.maximum(scaled_iv(cp, i0e, i1e, nu, argument), cp.finfo(cp.float64).tiny)
        log_density = (
            cp.log(radius)[None, :]
            + nu * (cp.log(radius)[None, :] - cp.log(b))
            - 0.5 * (radius[None, :] - b) ** 2
            + cp.log(bessel)
        )
        density[...] = cp.exp(log_density)
        density[central] = central_density
        answer[start : start + len(a)] = density @ kernel_weights
    if not bool(cp.all(cp.isfinite(answer)).item()):
        raise FloatingPointError("non-finite GPU noncentral-chi radial moment")
    return answer


class GpuPopulationDR:
    """CuPy implementation of the analytically reduced DR population operator."""

    def __init__(self, geometry, h: float, sigma: float, latent_count: int, device: int | None = None, pca_radial_order: int = 128, pca_padding: float = 12.0):
        cp, _, _ = _cupy()
        self.cp = cp
        self.device = int(device or 0)
        cp.cuda.Device(self.device).use()
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        self.eta2 = self.h**2 + self.sigma**2
        latent, weights = geometry.latent_rule(int(latent_count))
        self.latent = cp.asarray(latent, dtype=cp.float64)
        self.latent_weights = cp.asarray(weights, dtype=cp.float64)
        self.pca_operator = GpuPopulationSTSP(
            geometry, h, sigma, latent_count, pca_radial_order,
            noncentral_radius_padding=pca_padding, device=device,
            kernel_kind="polynomial")

    def evaluate(self, point):
        cp = self.cp
        cp.cuda.Device(self.device).use()
        displacement = self.latent - cp.asarray(point, dtype=cp.float64)
        exponent = -0.5 * cp.sum(displacement**2, axis=1) / self.eta2
        maximum = cp.max(exponent)
        weights = self.latent_weights * cp.exp(exponent - maximum)
        mass = cp.sum(weights)
        normalized = weights / mass
        mean = cp.einsum("n,ni->i", normalized, displacement)
        raw_second = cp.einsum("n,ni,nj->ij", normalized, displacement, displacement)
        hessian = raw_second / self.eta2**2 - cp.eye(self.geometry.D, dtype=cp.float64) / self.eta2
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = cp.linalg.eigh(hessian)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        projector = normal_basis @ normal_basis.T
        residual = projector @ (mean / self.eta2)
        return State(
            cp.asnumpy(residual),
            cp.asnumpy(normal_basis),
            cp.asnumpy(eigenvalues),
            float(((eigenvalues[q] - eigenvalues[q - 1]) * self.h**2).item()),
            float(eigenvalues[q - 1].item()),
            float((cp.exp(maximum) * mass).item()),
        )


class GpuPopulationSTSP:
    """CuPy implementation of the exact Gaussian-reduced STSP operator."""

    def __init__(
        self,
        geometry,
        h: float,
        sigma: float,
        latent_count: int,
        radial_order: int,
        power: int = 6,
        noncentral_radius_padding: float = 12.0,
        device: int | None = None,
        kernel_kind: str = "polynomial",
    ):
        cp, _, _ = _cupy()
        self.cp = cp
        self.device = int(device or 0)
        cp.cuda.Device(self.device).use()
        self.geometry = geometry
        self.h = float(h)
        self.sigma = float(sigma)
        latent, weights = geometry.latent_rule(int(latent_count))
        self.latent = cp.asarray(latent, dtype=cp.float64)
        self.latent_weights = cp.asarray(weights, dtype=cp.float64)
        self.radial_order = int(radial_order)
        self.power = int(power)
        self.kernel_kind = kernel_kind
        self.noncentral_radius_padding = float(noncentral_radius_padding)

    def evaluate(self, point):
        cp = self.cp
        cp.cuda.Device(self.device).use()
        displacement_all = self.latent - cp.asarray(point, dtype=cp.float64)
        radius_all = cp.sqrt(cp.maximum(cp.sum(displacement_all**2, axis=1), 0.0)) / self.sigma
        kappa = self.h / self.sigma
        active = radius_all <= kappa + self.noncentral_radius_padding
        displacement = displacement_all[active]
        radius = radius_all[active]
        latent_weights = self.latent_weights[active]
        if len(displacement) == 0:
            raise ArithmeticError("STSP population tail cutoff retained no latent nodes")
        m0 = radial_mass_gpu(self.geometry.D, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        m1 = radial_mass_gpu(self.geometry.D + 2, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        m2 = radial_mass_gpu(self.geometry.D + 4, radius, kappa, self.radial_order, self.power, self.kernel_kind)
        mass = cp.sum(latent_weights * m0)
        if not bool(cp.isfinite(mass).item()) or float(mass.item()) <= 0.0:
            raise ArithmeticError("nonpositive GPU STSP population mass")
        offset = cp.einsum("n,n,ni->i", latent_weights, m1, displacement) / mass
        raw_second = self.sigma**2 * cp.eye(self.geometry.D, dtype=cp.float64) * cp.sum(latent_weights * m1)
        raw_second += cp.einsum("n,n,ni,nj->ij", latent_weights, m2, displacement, displacement)
        covariance = raw_second / mass - cp.outer(offset, offset)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
        q = self.geometry.q
        normal_basis = eigenvectors[:, :q]
        projector = normal_basis @ normal_basis.T
        residual = -projector @ offset
        return State(
            cp.asnumpy(residual),
            cp.asnumpy(normal_basis),
            cp.asnumpy(eigenvalues),
            float(((eigenvalues[q] - eigenvalues[q - 1]) / self.h**2).item()),
            None,
            float((mass / self.h).item()),
            cp.asnumpy(normal_basis),
            float(((eigenvalues[q] - eigenvalues[q - 1]) / self.h**2).item()),
        )
