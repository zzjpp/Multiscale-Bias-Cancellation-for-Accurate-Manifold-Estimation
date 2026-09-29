from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e56v2.experiment import finite_conditioning_gate, finite_seed, weights_for


ROOT = Path(__file__).resolve().parents[1]
POPULATION = json.loads((ROOT / "config" / "population.json").read_text(encoding="utf-8"))
FINITE = json.loads((ROOT / "config" / "finite_sample.json").read_text(encoding="utf-8"))


def test_release_contains_exactly_the_two_figure_designs():
    assert POPULATION["sigma_grid"] == [0.03, 0.025, 0.02, 0.016, 0.0125, 0.01, 0.008]
    assert POPULATION["query_count"] == 64
    assert POPULATION["formal_evidence"] is True
    assert FINITE["sample_sizes"] == [240000, 480000, 960000]
    assert FINITE["repeats"] == 20
    assert FINITE["query_count"] == 256
    assert FINITE["formal_evidence"] is True


def test_four_method_contract_and_weight_constraints():
    expected = ["Single", "Scalar-A", "Scalar-C", "Directional"]
    assert POPULATION["methods"] == FINITE["methods"] == expected
    for config in (POPULATION, FINITE):
        weights = weights_for(config)
        assert weights["max_abs_residual"] < 1e-10
        assert np.isfinite(weights["W"])


def test_finite_paper_protocol_and_revised_gate_preserve_provenance():
    provenance = FINITE["provenance"]
    assert provenance["legacy_declared_c0"] == 5.0
    assert provenance["legacy_applied_c0"] == FINITE["c0"] == 3.0
    assert FINITE["legacy_failed_gate"]["pass"] is False
    weights = weights_for(FINITE)
    gate = finite_conditioning_gate(FINITE, weights)
    assert gate["pass"] is True
    assert gate["observed"] == weights["complementary_condition"]
    assert gate["observed"] <= gate["threshold"]
    original_gate = {**FINITE, "conditioning_gate": {
        **FINITE["conditioning_gate"], "threshold": 50.0,
    }}
    assert finite_conditioning_gate(original_gate, weights)["pass"] is False


def test_finite_seed_lineage_is_frozen_and_unique():
    values = {
        finite_seed(FINITE, sample_size, repeat)
        for sample_size in FINITE["sample_sizes"]
        for repeat in range(FINITE["repeats"])
    }
    assert len(values) == 60
    assert min(values) == FINITE["master_seed"]
