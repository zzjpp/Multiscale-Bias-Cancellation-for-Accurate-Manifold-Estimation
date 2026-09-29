from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from dryy.coefficients import dr_weights, yy_weights
from dryy.config import load_config
from dryy.dr import EmpiricalDR
from dryy.geometry import geometries_from_config, noisy_sample
from dryy.yy_common import evaluate_common_start, prepare_common_start


def test_fixed_design_axes():
    config, _ = load_config("e3")
    assert config["n"] == 12000
    assert config["sigma"] == 0.015
    assert config["dr"]["h1_ratio"] == 0.5
    assert min(config["dr"]["bandwidths_h2"]) == 0.06
    assert max(config["dr"]["bandwidths_h2"]) == 5.0
    assert config["yy"]["c0"] == 24.0
    r1 = np.asarray(config["yy"]["central_kappas"]) * config["sigma"]
    np.testing.assert_allclose(r1, config["dr"]["bandwidths_h2"], rtol=0, atol=1e-12)


def test_dr_weight_constraints_on_release_grid():
    config, _ = load_config("e3")
    for h in config["dr"]["bandwidths_h2"]:
        result = dr_weights(h / 2, h, config["sigma"])
        assert result["max_abs_residual"] < 1e-10
        assert abs(result["weights"].sum() - 1.0) < 1e-12


def test_yy_directional_constraints_all_geometries_and_scales():
    config, _ = load_config("e3")
    for geometry in geometries_from_config(config):
        for central in config["yy"]["central_kappas"]:
            kappas = central * np.asarray(config["yy"]["relative_radii"])
            result = yy_weights(geometry.d, geometry.q, kappas, config["yy"]["kernel_power"])
            assert result["max_abs_residual"] < 1e-10
            if geometry.q == 1:
                assert np.array_equal(result["axial"], result["complementary"])


def test_dr_analytic_directional_derivative():
    config, _ = load_config("e3")
    geometry = geometries_from_config(config)[0]
    observations, _ = noisy_sample(config, geometry, 0, config["n"])
    controls = dict(config["dr"])
    controls.setdefault("relative_omitted_weight_bound_max", 0.0)
    operator = EmpiricalDR(observations, geometry, 0.2, controls)
    point = geometry.query_grid(16, config["query_seed"])[0]
    state = operator.evaluate(point)
    direction = geometry.normal_basis(point)[:, 0]
    analytic = operator.directional_residual(state, direction)
    epsilon = 1e-6
    finite = (
        operator.evaluate(point + epsilon * direction).residual
        - operator.evaluate(point - epsilon * direction).residual
    ) / (2 * epsilon)
    assert np.allclose(analytic, finite, rtol=2e-4, atol=2e-6)


def test_yy_common_start_is_identical_across_radius_grid():
    config, _ = load_config("e3")
    geometry = geometries_from_config(config)[0]
    observations, _ = noisy_sample(config, geometry, 0, config["n"])
    tree = cKDTree(observations)
    controls = dict(config["yy"])
    controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
    truth = geometry.query_grid(16, config["query_seed"])[0]
    start = prepare_common_start(observations, tree, geometry, truth, config["sigma"], controls)
    digests = []
    for central in config["yy"]["central_kappas"]:
        result = evaluate_common_start(
            observations, tree, geometry, start, config["sigma"], central, controls
        )
        digests.append(result["common_start_digest"])
        assert all(stage["common_start_digest"] == start["digest"] for stage in result["stages"])
    assert len(set(digests)) == 1
