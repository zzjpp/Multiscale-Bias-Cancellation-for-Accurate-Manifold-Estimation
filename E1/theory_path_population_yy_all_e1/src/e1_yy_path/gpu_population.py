from __future__ import annotations

import numpy as np

from .coefficients import yy_weights
from .gpu_moments import cupy_modules, longitudinal_moments_gpu, radial_polynomial_moment_gpu


class GpuPopulationYY:
    def __init__(
        self,
        geometry,
        sigma,
        latent_resolution,
        radial_order,
        longitudinal_order,
        kappas,
        c0,
        c2,
        power,
        tail_padding,
        device=0,
    ):
        cp, _, _ = cupy_modules()
        self.cp = cp
        self.device = int(device)
        cp.cuda.Device(self.device).use()
        self.geometry = geometry
        self.sigma = float(sigma)
        self.radial_order = int(radial_order)
        self.longitudinal_order = int(longitudinal_order)
        self.kappas = np.asarray(kappas, dtype=float)
        self.c0 = float(c0)
        self.c2 = float(c2)
        self.power = int(power)
        self.tail_padding = float(tail_padding)
        latent, latent_weights = geometry.latent_rule(int(latent_resolution))
        self.latent = cp.asarray(latent, dtype=cp.float64)
        self.latent_weights = cp.asarray(latent_weights, dtype=cp.float64)
        self.weights = yy_weights(geometry.d, geometry.q, self.kappas, self.power)
        self.axial_weights = cp.asarray(self.weights["axial"], dtype=cp.float64)
        self.complementary_weights = cp.asarray(self.weights["complementary"], dtype=cp.float64)
        self.single_index = int(np.flatnonzero(np.isclose(self.kappas, 5.0))[0])

    def evaluate(self, amplitude):
        cp = self.cp
        cp.cuda.Device(self.device).use()
        center = cp.asarray(
            self.geometry.preliminary_canonical(float(amplitude), self.sigma), dtype=cp.float64
        )
        displacement_all = self.latent - center
        radius_all = cp.linalg.norm(displacement_all, axis=1) / self.sigma
        active_first = radius_all <= self.c0 + self.tail_padding
        displacement = displacement_all[active_first]
        radii = radius_all[active_first]
        latent_weights = self.latent_weights[active_first]
        if len(displacement) == 0:
            raise ArithmeticError("first-stage tail cutoff retained no latent nodes")
        mass_nodes = radial_polynomial_moment_gpu(
            self.geometry.D, radii, self.c0, self.power, self.radial_order
        )
        mean_nodes = radial_polynomial_moment_gpu(
            self.geometry.D + 2, radii, self.c0, self.power, self.radial_order
        )
        mass = cp.sum(latent_weights * mass_nodes)
        relative_mean = cp.einsum("n,n,ni->i", latent_weights, mean_nodes, displacement) / mass
        signal = cp.linalg.norm(relative_mean)
        if not bool(cp.isfinite(mass).item()) or float(mass.item()) <= 0.0:
            raise ArithmeticError("invalid first-stage YY population mass")
        if not bool(cp.isfinite(signal).item()) or float(signal.item()) <= 0.0:
            raise ArithmeticError("invalid first-stage YY population signal")
        axis = relative_mean / signal

        longitudinal_all = displacement_all @ axis
        perpendicular_all = displacement_all - longitudinal_all[:, None] * axis
        perpendicular_radius_all = cp.linalg.norm(perpendicular_all, axis=1) / self.sigma
        r2 = self.c2 * self.sigma * np.sqrt(np.log(1.0 / self.sigma))
        outputs = []
        second_masses = []
        active_counts = []
        for kappa in self.kappas:
            active = (
                (cp.abs(longitudinal_all) <= r2 + self.tail_padding * self.sigma)
                & (perpendicular_radius_all <= float(kappa) + self.tail_padding)
            )
            longitudinal = longitudinal_all[active]
            perpendicular = perpendicular_all[active]
            perpendicular_radius = perpendicular_radius_all[active]
            latent_weights = self.latent_weights[active]
            if len(longitudinal) == 0:
                raise ArithmeticError("second-stage tail cutoff retained no latent nodes")
            longitudinal_mass, longitudinal_first = longitudinal_moments_gpu(
                longitudinal, self.sigma, r2, self.longitudinal_order
            )
            perpendicular_mass = radial_polynomial_moment_gpu(
                self.geometry.D - 1,
                perpendicular_radius,
                float(kappa),
                self.power,
                self.radial_order,
            )
            perpendicular_mean = radial_polynomial_moment_gpu(
                self.geometry.D + 1,
                perpendicular_radius,
                float(kappa),
                self.power,
                self.radial_order,
            )
            stage_mass = cp.sum(latent_weights * longitudinal_mass * perpendicular_mass)
            numerator = axis * cp.sum(latent_weights * longitudinal_first * perpendicular_mass)
            numerator += cp.einsum(
                "n,n,n,ni->i", latent_weights, longitudinal_mass, perpendicular_mean, perpendicular
            )
            if not bool(cp.isfinite(stage_mass).item()) or float(stage_mass.item()) <= 0.0:
                raise ArithmeticError("invalid second-stage YY population mass")
            outputs.append(center + numerator / stage_mass)
            second_masses.append(float(stage_mass.item()))
            active_counts.append(int(cp.sum(active).item()))
        outputs = cp.stack(outputs)
        deltas = outputs - center
        longitudinal = (deltas @ axis)[:, None] * axis[None, :]
        complementary = deltas - longitudinal
        multi = center + cp.einsum("k,ki->i", self.axial_weights, longitudinal)
        multi += cp.einsum("k,ki->i", self.complementary_weights, complementary)
        return {
            "center": cp.asnumpy(center),
            "axis": cp.asnumpy(axis),
            "first_mass": float(mass.item()),
            "first_signal_over_sigma": float((signal / self.sigma).item()),
            "scale_outputs": cp.asnumpy(outputs),
            "single": cp.asnumpy(outputs[self.single_index]),
            "multi": cp.asnumpy(multi),
            "second_masses": second_masses,
            "first_active_count": int(cp.sum(active_first).item()),
            "second_active_counts": active_counts,
        }
