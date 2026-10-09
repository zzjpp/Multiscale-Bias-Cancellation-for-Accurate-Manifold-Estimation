from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .coefficients import yy_weights
from .kernels import effective_count, longitudinal_profile, polynomial_cutoff
from .yy import YYFailure


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_common_start(data, tree: cKDTree, geometry, truth, sigma: float, controls: dict) -> dict:
    """Compute Eq. (10) exactly once for a supplied preliminary query."""
    truth = np.asarray(truth, dtype=float)
    center = geometry.preliminary(truth, sigma)
    power = int(controls["kernel_power"])
    r0 = float(controls["c0"]) * sigma
    indices = np.asarray(sorted(tree.query_ball_point(center, r0)), dtype=int)
    if len(indices) == 0:
        raise YYFailure("first_stage_low_support", support_count=0)
    vectors = np.asarray(data[indices] - center, dtype=float)
    weights = polynomial_cutoff(np.sum(vectors**2, axis=1), r0, power)
    ess = effective_count(weights)
    total = float(weights.sum())
    minimum = float(controls["minimum_effective_count"])
    if not np.isfinite(total) or total <= 0 or ess < minimum:
        raise YYFailure("first_stage_mass_failure", support_count=len(indices), effective_count=ess)
    offset = weights @ vectors / total
    signal = float(np.linalg.norm(offset) / sigma)
    if not np.isfinite(signal) or signal < float(controls["minimum_signal_over_sigma"]):
        raise YYFailure("first_stage_signal_failure", signal_over_sigma=signal)
    axis = offset / np.linalg.norm(offset)
    return {
        "truth": truth, "center": center, "axis": axis,
        "digest": _digest(center, offset, axis),
        "first_support_count": int(np.count_nonzero(weights)),
        "first_effective_count": float(ess), "signal_over_sigma": signal,
    }


def evaluate_common_start(data, tree: cKDTree, geometry, start: dict, sigma: float, central_kappa: float, controls: dict) -> dict:
    """Evaluate Eq. (11) at three radii without changing the common start."""
    center = np.asarray(start["center"], dtype=float)
    axis = np.asarray(start["axis"], dtype=float)
    power = int(controls["kernel_power"])
    relative = np.asarray(controls["relative_radii"], dtype=float)
    kappas = float(central_kappa) * relative
    radii = sigma * kappas
    weights_info = yy_weights(geometry.d, geometry.q, kappas, power)
    tolerance = float(controls.get("weight_residual_tolerance", 1e-10))
    if weights_info["max_abs_residual"] > tolerance:
        raise YYFailure("weight_constraint_failure", residual=weights_info["max_abs_residual"])
    r2 = float(controls["C2"]) * sigma * np.sqrt(np.log(1.0 / sigma))
    indices = np.asarray(sorted(tree.query_ball_point(center, float(np.hypot(r2, radii.max())))), dtype=int)
    if len(indices) == 0:
        raise YYFailure("second_stage_low_support", support_count=0)
    vectors = np.asarray(data[indices] - center, dtype=float)
    longitudinal = vectors @ axis
    perpendicular = vectors - longitudinal[:, None] * axis
    longitudinal_weights = longitudinal_profile(np.abs(longitudinal) / r2)
    perpendicular_squared = np.sum(perpendicular**2, axis=1)
    minimum = float(controls["minimum_effective_count"])
    outputs, stages = [], []
    for kappa, radius in zip(kappas, radii):
        weights = longitudinal_weights * polynomial_cutoff(perpendicular_squared, radius, power)
        total = float(weights.sum())
        ess = effective_count(weights)
        stage = {
            "kappa": float(kappa), "r1": float(radius),
            "support_count": int(np.count_nonzero(weights)), "effective_count": float(ess),
            "common_start_digest": start["digest"],
        }
        if not np.isfinite(total) or total <= 0 or ess < minimum:
            outputs.append(None)
            stages.append({**stage, "status": "failed", "reason": "second_stage_mass_failure"})
        else:
            outputs.append(center + weights @ vectors / total)
            stages.append({**stage, "status": "success", "reason": None})
    center_index = np.flatnonzero(np.isclose(kappas, central_kappa, rtol=0.0, atol=1e-14))
    if len(center_index) != 1:
        raise YYFailure("central_scale_missing")
    single = outputs[int(center_index[0])]
    multi = None
    if all(value is not None for value in outputs):
        output_array = np.asarray(outputs)
        deltas = output_array - center
        axial = np.einsum("ki,i->k", deltas, axis)[:, None] * axis[None, :]
        complementary = deltas - axial
        multi = center + np.einsum("k,ki->i", weights_info["axial"], axial)
        multi += np.einsum("k,ki->i", weights_info["complementary"], complementary)
    return {
        "single": single, "single_radius_points": outputs, "multi": multi,
        "single_status": "success" if single is not None else "failed",
        "multi_status": "success" if multi is not None else "failed",
        "common_start_digest": start["digest"], "kappas": kappas, "radii": radii,
        "weights": weights_info, "stages": stages,
    }

