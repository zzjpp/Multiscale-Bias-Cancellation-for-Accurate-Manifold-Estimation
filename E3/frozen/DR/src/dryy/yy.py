from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .coefficients import yy_weights
from .kernels import effective_count, longitudinal_profile, polynomial_cutoff


class YYFailure(RuntimeError):
    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_yy_query(data, tree: cKDTree, geometry, truth, sigma: float, central_kappa: float, controls: dict) -> dict:
    """Empirical Eqs. (10)--(11), with one immutable common start."""
    truth = np.asarray(truth, dtype=float)
    center = geometry.preliminary(truth, sigma)
    power = int(controls["kernel_power"])
    r0 = float(controls["c0"]) * sigma
    first_indices = np.asarray(sorted(tree.query_ball_point(center, r0)), dtype=int)
    if len(first_indices) == 0:
        raise YYFailure("first_stage_low_support", support_count=0)
    first_vectors = np.asarray(data[first_indices] - center, dtype=float)
    first_weights = polynomial_cutoff(np.sum(first_vectors**2, axis=1), r0, power)
    first_ess = effective_count(first_weights)
    first_total = float(first_weights.sum())
    minimum = float(controls["minimum_effective_count"])
    if not np.isfinite(first_total) or first_total <= 0 or first_ess < minimum:
        raise YYFailure("first_stage_mass_failure", support_count=len(first_indices), effective_count=first_ess)
    first_offset = first_weights @ first_vectors / first_total
    signal = float(np.linalg.norm(first_offset) / sigma)
    if not np.isfinite(signal) or signal < float(controls["minimum_signal_over_sigma"]):
        raise YYFailure("first_stage_signal_failure", signal_over_sigma=signal)
    axis = first_offset / np.linalg.norm(first_offset)
    common_start_digest = _digest(center, first_offset, axis)

    relative = np.asarray(controls["relative_radii"], dtype=float)
    kappas = float(central_kappa) * relative
    radii = sigma * kappas
    weights_info = yy_weights(geometry.d, geometry.q, kappas, power)
    tolerance = float(controls.get("weight_residual_tolerance", 1e-10))
    if weights_info["max_abs_residual"] > tolerance:
        raise YYFailure("weight_constraint_failure", residual=weights_info["max_abs_residual"])
    r2 = float(controls["C2"]) * sigma * np.sqrt(np.log(1.0 / sigma))
    candidate_radius = float(np.hypot(r2, radii.max()))
    indices = np.asarray(sorted(tree.query_ball_point(center, candidate_radius)), dtype=int)
    if len(indices) == 0:
        raise YYFailure("second_stage_low_support", support_count=0)
    vectors = np.asarray(data[indices] - center, dtype=float)
    longitudinal = vectors @ axis
    perpendicular = vectors - longitudinal[:, None] * axis
    longitudinal_weights = longitudinal_profile(np.abs(longitudinal) / r2)
    perpendicular_squared = np.sum(perpendicular**2, axis=1)
    outputs = []
    diagnostics = []
    for kappa, radius in zip(kappas, radii):
        weights = longitudinal_weights * polynomial_cutoff(perpendicular_squared, radius, power)
        total = float(weights.sum())
        ess = effective_count(weights)
        stage = {
            "kappa": float(kappa), "r1": float(radius),
            "support_count": int(np.count_nonzero(weights)), "effective_count": float(ess),
            "common_start_digest": common_start_digest,
        }
        if not np.isfinite(total) or total <= 0 or ess < minimum:
            outputs.append(None)
            diagnostics.append({**stage, "status": "failed", "reason": "second_stage_mass_failure"})
        else:
            outputs.append(center + weights @ vectors / total)
            diagnostics.append({**stage, "status": "success", "reason": None})
    central_matches = np.flatnonzero(np.isclose(kappas, central_kappa, rtol=0.0, atol=1e-14))
    if len(central_matches) != 1:
        raise YYFailure("central_scale_missing")
    central_output = outputs[int(central_matches[0])]
    multi = None
    if all(value is not None for value in outputs):
        output_array = np.asarray(outputs)
        deltas = output_array - center
        axial_components = np.einsum("ki,i->k", deltas, axis)[:, None] * axis[None, :]
        complementary_components = deltas - axial_components
        multi = center + np.einsum("k,ki->i", weights_info["axial"], axial_components)
        multi += np.einsum("k,ki->i", weights_info["complementary"], complementary_components)
    return {
        "single": central_output,
        "single_radius_points": outputs,
        "multi": multi,
        "single_status": "success" if central_output is not None else "failed",
        "multi_status": "success" if multi is not None else "failed",
        "center": center,
        "axis": axis,
        "common_start_digest": common_start_digest,
        "first_support_count": int(np.count_nonzero(first_weights)),
        "first_effective_count": float(first_ess),
        "signal_over_sigma": signal,
        "kappas": kappas,
        "radii": radii,
        "weights": weights_info,
        "stages": diagnostics,
    }

