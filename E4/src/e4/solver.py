from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import CONFIG
from .operator import OperatorFailure

FAILURE_NAMES = set(CONFIG["failure_taxonomy"])


@dataclass
class MatchResult:
    status: str
    point: np.ndarray | None
    coordinate: np.ndarray
    iterations: int
    diagnostics: dict
    trace: list


def solve_fiber(operator, reference, basis, search_bandwidth, controls=None):
    cfg = dict(CONFIG["matching"])
    if controls:
        cfg.update(controls)
    z = np.asarray(reference, dtype=float).copy()
    E = np.asarray(basis, dtype=float).copy()
    if E.shape != (3, 2) or not np.allclose(E.T @ E, np.eye(2), atol=1e-11, rtol=0):
        raise ValueError("orthonormal 3x2 fiber basis required")
    coordinate = np.zeros(2)
    # The theoretical normal-fiber chart is local on the O(h^2) scale.
    radius = float(cfg["search_radius_factor"] * search_bandwidth**2)
    trace = []

    def fail(reason, iteration, **details):
        return MatchResult(reason if reason in FAILURE_NAMES else "other", None, coordinate.copy(), iteration, {"radius": radius, **details}, trace)

    def inspect(value):
        if np.linalg.norm(value) > radius:
            raise OperatorFailure("search_radius_exceeded", failure_stage="root")
        state = operator.evaluate(z + E @ value)
        if state.support_count < cfg["minimum_support"]:
            raise OperatorFailure("low_support", support_count=state.support_count)
        if state.normalized_mass < cfg["normalized_mass_min"]:
            raise OperatorFailure("insufficient_mass", normalized_mass=state.normalized_mass)
        if state.normalized_eigengap < cfg["normalized_eigengap_min"]:
            raise OperatorFailure("eigengap_failure", eigengap=state.normalized_eigengap)
        discrepancy = float(np.linalg.norm(E @ E.T - state.Q, 2))
        if discrepancy > cfg["transversality_max"]:
            raise OperatorFailure("normal_transversality_failure", normal_space_discrepancy=discrepancy)
        projected = E.T @ state.residual
        jacobian = operator.fiber_jacobian(state, E)
        condition = float(np.linalg.cond(jacobian))
        if not np.isfinite(condition) or condition > cfg["jacobian_condition_max"]:
            raise OperatorFailure("jacobian_condition_failure", jacobian_condition=condition)
        diag = {"projected_residual": float(np.linalg.norm(projected)), "full_residual": float(np.linalg.norm(state.residual)), "normalized_mass": state.normalized_mass, "support_count": state.support_count, "effective_support": state.effective_support, "eigengap": state.normalized_eigengap, "normal_space_discrepancy": discrepancy, "jacobian_condition": condition}
        return state, projected, jacobian, diag

    for iteration in range(cfg["max_newton_iterations"] + 1):
        try:
            state, residual, jacobian, diag = inspect(coordinate)
        except OperatorFailure as exc:
            return fail(exc.reason, iteration, **exc.diagnostics)
        trace.append({"coordinate": coordinate.copy(), **diag})
        if diag["projected_residual"] <= cfg["projected_residual_tolerance"]:
            if diag["full_residual"] > cfg["full_residual_tolerance"]:
                return fail("residual_failure", iteration, **diag)
            return MatchResult("success", z + E @ coordinate, coordinate.copy(), iteration, diag, trace)
        if iteration == cfg["max_newton_iterations"]:
            return fail("newton_max_iter", iteration, **diag)
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return fail("jacobian_condition_failure", iteration, **diag)
        accepted = False
        reasons = []
        for backtrack in range(cfg["max_backtracks"]):
            alpha = 0.5**backtrack
            candidate = coordinate + alpha * step
            try:
                _, trial, _, _ = inspect(candidate)
            except OperatorFailure as exc:
                reasons.append(exc.reason)
                continue
            if np.linalg.norm(trial) <= (1 - cfg["armijo_factor"] * alpha) * np.linalg.norm(residual):
                coordinate = candidate
                accepted = True
                break
            reasons.append("armijo")
        if not accepted:
            return fail(reasons[0] if reasons and len(set(reasons)) == 1 and reasons[0] != "armijo" else "newton_no_descent", iteration, trial_failures=reasons, **diag)
    raise AssertionError("unreachable")
