from __future__ import annotations

import inspect

import numpy as np

from e3.coefficients import stsp_coefficient, two_scale_weights
from e3.config import CONFIG
from e3.geometry import BY_KEY, GEOMETRIES, noisy_sample, seed_lineage
from e3.metrics import reliable
from e3.operator import EmpiricalSTSP
from e3.pipelines import estimate_normal


EXPECTED_GRID = [
    0.06, 0.075, 0.09, 0.11, 0.135, 0.165, 0.2, 0.245,
    0.3, 0.37, 0.45, 0.55, 0.67, 0.82, 1.0, 1.2, 1.4,
    1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0,
]


def test_fixed_design_and_blind_enf_api():
    assert CONFIG["n"] == 12000
    assert CONFIG["sigma"] == 0.015
    assert CONFIG["bandwidths_h2"] == EXPECTED_GRID
    assert CONFIG["h1_ratio"] == 0.5
    assert len(GEOMETRIES) == 5
    assert tuple(inspect.signature(estimate_normal).parameters) == ("operator", "point")


def test_all_weights_cancel_current_joint_coefficient():
    for geometry in GEOMETRIES:
        for h in EXPECTED_GRID:
            coefficient = stsp_coefficient(geometry.d, geometry.D, h, CONFIG["sigma"])
            assert coefficient["status"] == "converged"
            result = two_scale_weights(geometry.d, geometry.D, h / 2, h, CONFIG["sigma"])
            assert result["status"] == "converged"
            assert abs(np.sum(result["weights"]) - 1) < 1e-12
            assert abs(result["weights"] @ np.array([result["c1"], result["c2"]])) < 1e-12


def test_geometry_and_normal_dimension_contracts():
    for geometry in GEOMETRIES:
        sample = geometry.sample_clean(np.random.default_rng(7), 128)
        assert sample.shape == (128, geometry.D)
        np.testing.assert_allclose(geometry.distance(sample), 0, atol=3e-15)
    geometry = BY_KEY["sphere2_r10"]
    data = geometry.sample_clean(np.random.default_rng(3), CONFIG["n"])
    state = EmpiricalSTSP(data, geometry, 0.4).evaluate(geometry.query_grid(256)[0])
    assert state.eigenvectors[:, : geometry.q].shape == (10, 8)


def test_seed_lineage_is_deterministic_and_shared_across_scales():
    geometry = BY_KEY["sphere3_r4"]
    first = noisy_sample(geometry, 4, n=128)
    second = noisy_sample(geometry, 4, n=128)
    np.testing.assert_array_equal(first[2], second[2])
    assert first[3] == second[3]
    assert seed_lineage(geometry, 4)["clean"] != seed_lineage(geometry, 4)["noise"]


def test_reliability_rule():
    assert reliable(0.01, 18)
    assert not reliable(0.0100001, 20)
    assert not reliable(0.0, 17)
