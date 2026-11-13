from __future__ import annotations

import numpy as np

from .coefficients import yy_weights
from .moments import longitudinal_moments, radial_polynomial_moment


class PopulationYY:
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
    ):
        self.geometry = geometry
        self.sigma = float(sigma)
        self.radial_order = int(radial_order)
        self.longitudinal_order = int(longitudinal_order)
        self.kappas = np.asarray(kappas, dtype=float)
        self.c0 = float(c0)
        self.c2 = float(c2)
        self.power = int(power)
        self.tail_padding = float(tail_padding)
        self.latent, self.latent_weights = geometry.latent_rule(int(latent_resolution))
        self.weights = yy_weights(geometry.d, geometry.q, self.kappas, self.power)
        self.single_index = int(np.flatnonzero(np.isclose(self.kappas, 5.0))[0])

    def evaluate(self, amplitude):
        center = self.geometry.preliminary_canonical(float(amplitude), self.sigma)
        displacement_all = self.latent - center
        radius_all = np.linalg.norm(displacement_all, axis=1) / self.sigma
        active = radius_all <= self.c0 + self.tail_padding
        displacement = displacement_all[active]
        radii = radius_all[active]
        latent_weights = self.latent_weights[active]
        if len(displacement) == 0:
            raise ArithmeticError("first-stage tail cutoff retained no latent nodes")
        mass_nodes = radial_polynomial_moment(
            self.c0, radii, self.geometry.D, self.power, self.radial_order
        )
        mean_nodes = radial_polynomial_moment(
            self.c0, radii, self.geometry.D + 2, self.power, self.radial_order
        )
        mass = float(np.sum(latent_weights * mass_nodes))
        relative_mean = np.einsum("n,n,ni->i", latent_weights, mean_nodes, displacement) / mass
        signal = float(np.linalg.norm(relative_mean))
        if not np.isfinite(mass) or mass <= 0 or not np.isfinite(signal) or signal <= 0:
            raise ArithmeticError("invalid first-stage YY population mass or signal")
        axis = relative_mean / signal

        displacement_all = self.latent - center
        longitudinal_all = displacement_all @ axis
        perpendicular_all = displacement_all - longitudinal_all[:, None] * axis
        perpendicular_radius_all = np.linalg.norm(perpendicular_all, axis=1) / self.sigma
        r2 = self.c2 * self.sigma * np.sqrt(np.log(1.0 / self.sigma))
        outputs = []
        second_masses = []
        active_counts = []
        for kappa in self.kappas:
            active = (
                (np.abs(longitudinal_all) <= r2 + self.tail_padding * self.sigma)
                & (perpendicular_radius_all <= float(kappa) + self.tail_padding)
            )
            displacement = displacement_all[active]
            longitudinal = longitudinal_all[active]
            perpendicular = perpendicular_all[active]
            perpendicular_radius = perpendicular_radius_all[active]
            latent_weights = self.latent_weights[active]
            if len(displacement) == 0:
                raise ArithmeticError("second-stage tail cutoff retained no latent nodes")
            longitudinal_mass, longitudinal_first = longitudinal_moments(
                longitudinal, self.sigma, r2, self.longitudinal_order
            )
            perpendicular_mass = radial_polynomial_moment(
                float(kappa), perpendicular_radius, self.geometry.D - 1, self.power, self.radial_order
            )
            perpendicular_mean = radial_polynomial_moment(
                float(kappa), perpendicular_radius, self.geometry.D + 1, self.power, self.radial_order
            )
            node_mass = longitudinal_mass * perpendicular_mass
            stage_mass = float(np.sum(latent_weights * node_mass))
            numerator = axis * np.sum(latent_weights * longitudinal_first * perpendicular_mass)
            numerator += np.einsum(
                "n,n,n,ni->i", latent_weights, longitudinal_mass, perpendicular_mean, perpendicular
            )
            if not np.isfinite(stage_mass) or stage_mass <= 0:
                raise ArithmeticError("invalid second-stage YY population mass")
            outputs.append(center + numerator / stage_mass)
            second_masses.append(stage_mass)
            active_counts.append(int(np.sum(active)))
        outputs = np.asarray(outputs)
        deltas = outputs - center
        longitudinal = (deltas @ axis)[:, None] * axis[None, :]
        complementary = deltas - longitudinal
        multi = center + np.einsum("k,ki->i", self.weights["axial"], longitudinal)
        multi += np.einsum("k,ki->i", self.weights["complementary"], complementary)
        return {
            "center": center,
            "axis": axis,
            "first_mass": mass,
            "first_signal_over_sigma": signal / self.sigma,
            "scale_outputs": outputs,
            "single": outputs[self.single_index],
            "multi": multi,
            "second_masses": second_masses,
            "first_active_count": int(np.sum(radius_all <= self.c0 + self.tail_padding)),
            "second_active_counts": active_counts,
        }
