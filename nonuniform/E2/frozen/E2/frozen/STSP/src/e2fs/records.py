from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import numpy as np


def readonly(value):
    arr = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape)


def array_hash(value) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


class OperatorFailure(RuntimeError):
    def __init__(self, reason: str, **diagnostics):
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(reason)


@dataclass
class IntegralDiagnostic:
    primitive: str
    nodes: int
    estimated_error: float | None = None
    subintervals: int | None = None
    status: str = "fixed_rule_not_refined"
    details: dict = field(default_factory=dict)


@dataclass
class STSPState:
    center: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    residual: np.ndarray
    normalized_mass: float
    normalized_eigengap: float
    support_count: int
    effective_support: float
    support_vectors: np.ndarray
    derivative_weights: np.ndarray


@dataclass
class MatchResult:
    status: str
    point: np.ndarray | None
    fiber_coordinate: np.ndarray
    basis: np.ndarray
    iterations: int
    diagnostics: dict
    trace: list
