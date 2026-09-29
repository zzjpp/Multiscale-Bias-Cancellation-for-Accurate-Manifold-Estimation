from __future__ import annotations

import numpy as np

from .config import CONFIG
from .records import MatchResult, OperatorFailure, array_hash, readonly


FAILURE_NAMES = {"insufficient_mass", "low_support", "eigengap_failure", "normal_transversality_failure", "jacobian_condition_failure", "newton_no_descent", "newton_max_iter", "search_radius_exceeded", "residual_failure", "reference_branch_failure", "target_branch_failure", "nonfinite", "other"}


def solve_fiber(operator, reference, basis, search_bandwidth, controls=None) -> MatchResult:
    cfg = dict(CONFIG["matching"])
    if controls:
        cfg.update(controls)
    z, E = readonly(reference), readonly(basis)
    if E.ndim != 2 or E.shape[0] != z.size or not np.allclose(E.T @ E, np.eye(E.shape[1]), atol=1e-12, rtol=0):
        raise ValueError("orthonormal fiber basis required")
    coordinate = np.zeros(E.shape[1])
    trace = []
    # Theory: the unique normal-fiber root is sought in an O(h^2)
    # neighbourhood of the reference point, not an O(h) neighbourhood.
    radius = cfg["search_radius_factor"] * float(search_bandwidth) ** 2
    basis_digest = array_hash(E)

    def failure(reason, iteration, **detail):
        if reason not in FAILURE_NAMES:
            reason = "other"
        return MatchResult(reason, None, coordinate.copy(), E, iteration, {"controls": cfg, "radius": radius, "basis_hash": basis_digest, **detail}, trace)

    def inspect(value):
        if np.linalg.norm(value) > radius:
            raise OperatorFailure("search_radius_exceeded", failure_stage="root")
        state = operator.evaluate(z + E @ value)
        if state.support_count < cfg["minimum_support"]:
            raise OperatorFailure("low_support", failure_stage="root", support_count=state.support_count)
        if not np.isfinite(state.normalized_mass) or state.normalized_mass < cfg["normalized_mass_min"]:
            raise OperatorFailure("insufficient_mass", failure_stage="root", normalized_mass=state.normalized_mass)
        if not np.isfinite(state.normalized_eigengap) or state.normalized_eigengap < cfg["normalized_eigengap_min"]:
            raise OperatorFailure("eigengap_failure", failure_stage="root", eigengap=state.normalized_eigengap)
        discrepancy = float(np.linalg.norm(E @ E.T - state.Q, 2))
        if not np.isfinite(discrepancy) or discrepancy > cfg["transversality_max"]:
            raise OperatorFailure("normal_transversality_failure", failure_stage="root", normal_space_discrepancy=discrepancy)
        projected = E.T @ state.residual
        jacobian = operator.fiber_jacobian(state, E)
        if not np.isfinite(jacobian).all() or not np.isfinite(projected).all():
            raise OperatorFailure("nonfinite", failure_stage="jacobian")
        condition = float(np.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > cfg["jacobian_condition_max"]:
            raise OperatorFailure("jacobian_condition_failure", failure_stage="jacobian", jacobian_condition=condition)
        diagnostics = {"projected_residual": float(np.linalg.norm(projected)), "full_residual": float(np.linalg.norm(state.residual)), "normalized_mass": state.normalized_mass, "support_count": state.support_count, "effective_support": state.effective_support, "eigengap": state.normalized_eigengap, "normal_space_discrepancy": discrepancy, "jacobian_condition": condition}
        trace.append({"fiber_coordinate": value.copy(), "basis_hash": basis_digest, **diagnostics})
        return state, projected, jacobian, diagnostics

    for iteration in range(cfg["max_newton_iterations"] + 1):
        try:
            state, residual, jacobian, diagnostic = inspect(coordinate)
        except OperatorFailure as exc:
            return failure(exc.reason, iteration, **exc.diagnostics)
        if diagnostic["projected_residual"] <= cfg["projected_residual_tolerance"]:
            if diagnostic["full_residual"] > cfg["full_residual_tolerance"]:
                return failure("residual_failure", iteration, **diagnostic)
            return MatchResult("success", z + E @ coordinate, coordinate.copy(), E, iteration, {"controls": cfg, "radius": radius, "basis_hash": basis_digest, **diagnostic}, trace)
        if iteration == cfg["max_newton_iterations"]:
            return failure("newton_max_iter", iteration, **diagnostic)
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return failure("jacobian_condition_failure", iteration, **diagnostic)
        reasons = []
        accepted = False
        for backtrack in range(cfg["max_backtracks"]):
            alpha = 0.5**backtrack
            candidate = coordinate + alpha * step
            try:
                _, trial_residual, _, _ = inspect(candidate)
            except OperatorFailure as exc:
                reasons.append(exc.reason)
                continue
            if np.linalg.norm(trial_residual) <= (1 - cfg["armijo_factor"] * alpha) * np.linalg.norm(residual):
                coordinate = candidate
                accepted = True
                break
            reasons.append("armijo")
        if not accepted:
            if reasons and len(set(reasons)) == 1 and reasons[0] != "armijo":
                return failure(reasons[0], iteration, trial_failures=reasons, **diagnostic)
            return failure("newton_no_descent", iteration, trial_failures=reasons, **diagnostic)
    raise AssertionError("unreachable")
