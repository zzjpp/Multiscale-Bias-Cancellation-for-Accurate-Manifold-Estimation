from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from e56v2.experiment import finite_seed, finite_weight_residual_gate, weights_for

ROOT = Path(__file__).resolve().parents[1]
POPULATION = json.loads((ROOT / "config" / "population.json").read_text(encoding="utf-8"))
FINITE = json.loads((ROOT / "config" / "finite_sample.json").read_text(encoding="utf-8"))
PROVENANCE = json.loads((ROOT / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_scope_matches_paper_figures_6_and_22():
    assert POPULATION["paper_figure"] == 22
    assert POPULATION["formal_evidence"] is True
    assert FINITE["paper_figure"] == 6
    assert FINITE["formal_evidence"] is False
    assert FINITE["evidence_class"] == "paper-used held-out result"
    assert set(PROVENANCE["paper_figures"]) == {"6", "22"}


def test_frozen_paper_designs():
    assert POPULATION["sigma_grid"] == [0.03, 0.025, 0.02, 0.016, 0.0125, 0.01, 0.008]
    assert POPULATION["query_count"] == 64
    assert FINITE["sample_sizes"] == [240000, 480000, 960000]
    assert FINITE["repeats"] == 20
    assert FINITE["query_count"] == 256
    assert FINITE["kappas"] == [5.0, 6.0, 12.0]


def test_four_method_contract_and_weight_constraints():
    expected = ["Single", "Scalar-A", "Scalar-C", "Directional"]
    assert POPULATION["methods"] == FINITE["methods"] == expected
    for config in (POPULATION, FINITE):
        weights = weights_for(config)
        assert weights["max_abs_residual"] < 1e-10
        assert np.isfinite(weights["W"])


def test_finite_evidence_provenance_is_not_upgraded():
    assert FINITE["provenance"]["legacy_applied_c0"] == FINITE["c0"] == 3.0
    assert "selection_source" in FINITE["provenance"]
    assert FINITE["legacy_failed_gate"]["pass"] is False
    assert FINITE["legacy_failed_gate"]["observed"] > FINITE["legacy_failed_gate"]["threshold"]
    weights = weights_for(FINITE)
    replay_gate = finite_weight_residual_gate(FINITE, weights)
    assert replay_gate["pass"] is True
    assert replay_gate["complementary_condition_diagnostic"] == weights["complementary_condition"]
    assert "threshold" not in FINITE["weight_residual_gate"]


def test_finite_gate_checks_residual_without_condition_number_limit():
    weights = weights_for(FINITE)
    large_condition = {**weights, "complementary_condition": 1e12}
    assert finite_weight_residual_gate(FINITE, large_condition)["pass"] is True
    bad_residual = {**weights, "max_abs_residual": 1e-4}
    assert finite_weight_residual_gate(FINITE, bad_residual)["pass"] is False


def test_reference_inventory_and_hashes_are_exact():
    expected = PROVENANCE["reference_files"]
    observed = {path.name for path in (ROOT / "paper_figures").iterdir() if path.is_file()}
    assert observed == set(expected)
    for name, digest in expected.items():
        assert sha256(ROOT / "paper_figures" / name) == digest


def test_finite_seed_lineage_is_frozen_and_unique():
    seeds = {
        finite_seed(FINITE, sample_size, repeat)
        for sample_size in FINITE["sample_sizes"]
        for repeat in range(FINITE["repeats"])
    }
    assert len(seeds) == 60
    assert min(seeds) == FINITE["master_seed"]
