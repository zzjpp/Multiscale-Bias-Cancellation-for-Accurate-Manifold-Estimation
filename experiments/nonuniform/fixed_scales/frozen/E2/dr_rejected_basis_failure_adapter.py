"""Retain a finite basis rejected by the original solver as a query failure.

This is exception plumbing, NOT a basis repair or an alternative solver.
All successful returns and existing DRFailure exceptions come from the original
function unchanged. Only its exact orthogonality ValueError can be translated;
wrong dimensions, nonfinite bases and unrelated exceptions still stop the run.
The original atol=1e-10/rtol=0 decision is independently confirmed, never relaxed.
"""
from __future__ import annotations

import hashlib
import numpy as np

MESSAGE = "fiber basis must be finite and orthonormal"
REASON = "pca_basis_orthogonality_failure"
ATOL = 1e-10
RTOL = 0.0


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def wrap(original, xp, failure_type, observe):
    """Use the original check first; fail closed on unexpected rejection modes."""
    calls = 0
    def solve(operator, origin, basis, search_h, solver):
        nonlocal calls
        calls += 1
        try:
            return original(operator, origin, basis, search_h, solver)
        except ValueError as exc:
            if str(exc) != MESSAGE:
                raise
            device = xp.asarray(basis, dtype=xp.float64)
            D, q = operator.geometry.D, operator.geometry.q
            if device.shape != (D, q) or not bool(xp.all(xp.isfinite(device))):
                raise
            gram = device.T @ device
            if bool(xp.allclose(gram, xp.eye(q), rtol=RTOL, atol=ATOL)):
                raise AssertionError("Original rejection disagrees with frozen Gram check") from exc
            host = xp.asnumpy(device) if hasattr(xp, "asnumpy") else np.asarray(device)
            host_origin = xp.asnumpy(xp.asarray(origin)) if hasattr(xp, "asnumpy") else np.asarray(origin)
            deviation = xp.max(xp.abs(gram - xp.eye(q)))
            record = dict(failure_stage="pca_basis_validation", original_exception=MESSAGE,
                original_basis_atol=ATOL, original_basis_rtol=RTOL,
                rejection_index=None, solve_call_index=calls, D=D, q=q,
                operator_h=float(operator.h), search_h=float(search_h),
                gram_max_abs_error=float(deviation), rejected_basis=host.tolist(),
                rejected_basis_sha256=array_sha(host), fiber_origin=host_origin.tolist(),
                fiber_origin_sha256=array_sha(host_origin), basis_repaired=False,
                tolerance_relaxed=False, root_evaluation_started=False,
                CPU_basis_substituted=False, query_removed=False)
            record["rejection_index"] = observe(record)
            raise failure_type(REASON, **record) from exc
    return solve
