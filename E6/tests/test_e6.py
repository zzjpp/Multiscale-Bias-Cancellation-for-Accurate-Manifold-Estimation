import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from e6_misspec.coefficients import dr_coefficient, two_scale_weights, yy_weights


def test_dr_true_residual_matches_closed_form():
    sigma, h1, h2, d = 0.05, 0.15, 0.30, 1
    true = np.array([dr_coefficient(d, h, sigma) for h in (h1, h2)])
    for delta in (-0.3, -0.1, 0.0, 0.2, 0.3):
        sigma_hat = (1+delta)*sigma
        assumed = np.array([dr_coefficient(d, h, sigma_hat) for h in (h1, h2)])
        observed = two_scale_weights(assumed)@true
        expected = 0.5*d*(sigma**2-sigma_hat**2)
        assert abs(observed-expected) < 1e-15


def test_yy_constraints_use_sigma_only_through_kappa():
    weights = yy_weights(1, 2, [0.20, 0.50, 0.80], 0.05)
    assert np.max(abs(weights["axial_constraint_residual"])) < 1e-12
    assert np.max(abs(weights["complementary_constraint_residual"])) < 1e-12


def test_frozen_protocol():
    config = json.loads((ROOT/"config.json").read_text())
    assert config["true_sigma"] == 0.05
    assert config["dr"] == {"h1": 0.15, "h2": 0.30}
    assert config["query_count"] == 256
    assert config["deltas"] == [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]


def test_population_module_imports():
    from e6_misspec.population import radial_polynomial_moment
    assert callable(radial_polynomial_moment)
