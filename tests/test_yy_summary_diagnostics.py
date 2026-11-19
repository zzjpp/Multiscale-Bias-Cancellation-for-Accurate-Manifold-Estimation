"""Coverage diagnostics must not be advertised as Hausdorff certificates."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/yy_directional_ablation/src"))
from e56v2.e5_ablation import METHODS
from e56v2.e5_empirical import retained_fill_distance
from e56v2.experiment import summarize
from e56v2.geometry_e5 import arclength_theta, gamma4

DIAGNOSTIC = "sampled_fill_geometric_max_diagnostic"


def metric_rows(values):
    return {
        method: [
            {
                "geometric": value,
                "axial_normal": 2 * value,
                "complementary_normal": 3 * value,
                "tangential": 4 * value,
            }
            for value in values
        ]
        for method in METHODS
    }


def test_tangential_drift_defeats_the_old_max_and_sum_with_geometric_error():
    # Synthetic unit-circle example, not an estimator run: four well-spaced
    # bases correspond to four distinct outputs on a short arc of the circle.
    theta = np.arange(4) * np.pi / 2
    phi = np.linspace(-0.2, 0.2, 4)
    bases = np.column_stack((np.cos(theta), np.sin(theta)))
    outputs = np.column_stack((np.cos(phi), np.sin(phi)))
    true_base_fill = 2 * np.sin(np.pi / 8)
    true_cloud_hausdorff = 2 * np.cos(0.2 / 2)
    errors = outputs - bases
    tangents = np.column_stack((-np.sin(theta), np.cos(theta)))
    circle_metrics = [
        {
            "geometric": 0.0,
            "axial_normal": abs(float(error @ normal)),
            "complementary_normal": 0.0,
            "tangential": abs(float(error @ tangent)),
        }
        for error, normal, tangent in zip(errors, bases, tangents)
    ]
    summary = summarize({method: circle_metrics for method in METHODS}, true_base_fill)
    for item in summary.values():
        assert "finite_cloud_hausdorff_upper" not in item
        assert item[DIAGNOSTIC] == true_base_fill
        assert item[DIAGNOSTIC] < true_cloud_hausdorff
        assert true_base_fill + item["geometric_max"] < true_cloud_hausdorff
    paired_displacement = np.max(np.linalg.norm(outputs - bases, axis=1))
    assert true_cloud_hausdorff <= true_base_fill + paired_displacement


@pytest.mark.parametrize("fill", [None, 0.125])
def test_valid_error_summaries_are_unchanged(fill):
    values = [0.125, 0.25, 0.5]
    result = summarize(metric_rows(values), fill)
    for item in result.values():
        assert item["retained_count"] == 3
        for name, factor in [
            ("geometric", 1),
            ("axial_normal", 2),
            ("complementary_normal", 3),
            ("tangential", 4),
        ]:
            assert item[name + "_mean"] == np.mean(np.asarray(values) * factor)
            assert item[name + "_max"] == 0.5 * factor
        assert item[DIAGNOSTIC] == (None if fill is None else 0.5)
        assert "finite_cloud_hausdorff_upper" not in item


@pytest.mark.parametrize("fill", [None, 0.125])
def test_empty_queries_remain_undefined(fill):
    for item in summarize(metric_rows([]), fill).values():
        assert item["retained_count"] == 0 and item[DIAGNOSTIC] is None
        assert item["geometric_mean"] is None and item["geometric_max"] is None


def test_dense_grid_fill_estimate_is_not_a_continuous_covering_certificate():
    retained = gamma4(arclength_theta(np.arange(8) / 8))
    assert retained_fill_distance(retained, dense_count=8) == 0.0
    between = gamma4(arclength_theta(0.5 / 8))
    assert np.min(np.linalg.norm(retained - between, axis=1)) > 0.01


def test_source_amendment_changes_only_labels_schemas_and_documentation():
    record = json.loads((ROOT / "docs/SCIENTIFIC_PRESERVATION.json").read_bytes())
    amendment = next(
        item
        for item in record["subsequent_amendments"]
        if item["current_path"].endswith("e56v2/experiment.py")
    )
    tree = ast.parse((ROOT / amendment["current_path"]).read_text("utf8"))

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            old_names = {
                DIAGNOSTIC: "finite_cloud_hausdorff_upper",
                "E5_RELEASE_POPULATION_CELL_V2": "E5_RELEASE_POPULATION_CELL_V1",
                "E5_RELEASE_FINITE_PAPER_PROTOCOL_CELL_V3": "E5_RELEASE_FINITE_PAPER_PROTOCOL_CELL_V2",
            }
            if isinstance(node.value, str) and node.value in old_names:
                node.value = old_names[node.value]
            return node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]
            return node

    digest = hashlib.sha256(
        ast.dump(Normalize().visit(tree), include_attributes=False).encode("utf8")
    ).hexdigest()
    assert digest == amendment["original_normalized_ast_sha256"]
