"""Figure-only E6 replay; no fitting, gate relaxation or formal publication.

The deviation is measured against the output using the correct sigma, NOT
against the true manifold. All 54 cells and the two original geometries stay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

METHODS = ("DR", "STSP", "YY")
GEOMETRIES = ("circle_R2", "variable_curve_R3")
DELTAS = [-.3, -.2, -.1, -.05, 0., .05, .1, .2, .3]
SOURCE_SHA = "891dc684715676ad0d20b1d9db632a0ef14c7b9c9f704a002e204486ac4a5dae"
CONFIG_SHA = "cc682740d943810a0585c303b80918960244d99815659ad68859b96b58e15fd6"
PREVIOUS_SHA = "51ab641ffb3bd3eb5195a7ce4983552a69a95d7fea7c55a9cc7952b8c3c89b1c"
PREVIOUS_AUDIT_SHA = "a5a2a297df63d6bc5ce615fdca83618ae5c461188b1de58ffe5ea0133c5a82d1"
TEMPLATE_SHA = "29f91d521d85d6e52ffe7824379c25b12c7f066a334fe94d473acd86192ff39c"
STEMS = ("e6_dr_noise_misspecification_main", "e6_stsp_noise_misspecification_supplement",
         "e6_yy_noise_misspecification_supplement")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(left, right):
    try:
        return math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=2e-14)
    except (ValueError, TypeError):
        return False


def require_audit(report):
    expected = {"schema": "E6_PCA_EXTERNAL_RELEASE_AUDIT_V1", "status": "PASS",
                "complete_query_packet_count": 8, "expected_query_packet_count": 8,
                "audited_query_records": 2048, "recomputed_delta_query_rows": 18432,
                "preserved_failed_query_rows": 0, "errors": [], "formal_publication_permitted": False}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("E6 needs the complete unchanged numerical gates and eight query packets")
    gates = report.get("original_numerical_gates", {})
    checks = {"weight_constraints", "exact_zero_at_correct_sigma", "frozen_fit_digest_invariant_across_delta",
              "all_256_queries_resolved", "fine_coarse_resolution"}
    if gates.get("status") != "PASS" or set(gates.get("checks", {})) != checks or any(
            v is not True for v in gates["checks"].values()):
        raise ValueError("Original E6 numerical gate did not PASS")
    taxonomy = report.get("failure_taxonomy", {})
    expected_taxonomy = {f"{m}/{g}/{r}" for m in ("DR", "STSP") for g in GEOMETRIES for r in ("fine", "coarse")}
    if set(taxonomy) != expected_taxonomy or any(taxonomy.values()):
        raise ValueError("Failed query evidence must not be plotted as a complete release")


def validate_payload(payload, previous, summary):
    config = payload["config"]
    if (config != previous["config"] or config["deltas"] != DELTAS or config["query_count"] != 256
            or config["true_sigma"] != .05 or config["query_phase"] != .37
            or config["dr"] != {"h1": .15, "h2": .3}
            or config["stsp"] != {"h1": .15, "h2": .3, "kernel_power": 6}
            or set(payload["methods"]) != set(METHODS)):
        raise ValueError("Frozen E6 setting changed")
    provenance = payload.get("pca_revision_provenance", {})
    wanted = {"source_manifest_sha256": SOURCE_SHA, "matching_kernel": "K(t)=(1-t)_+^6",
              "pca_radius": "h1", "covariance": "centered_kernel_weighted", "normal_api_poisoned": True,
              "DR_STSP_fits_recomputed": True, "YY_recomputed": False}
    if any(provenance.get(k) != v for k, v in wanted.items()):
        raise ValueError("PCA/unchanged YY provenance changed")
    expected_rows = {}
    for method in METHODS:
        if set(payload["methods"][method]) != set(GEOMETRIES):
            raise ValueError("A geometry was removed or substituted")
        for geometry in GEOMETRIES:
            block = payload["methods"][method][geometry]
            coefficients, metrics = block["coefficients"], block["geometry"]
            if (len(coefficients) != 9 or len(metrics) != 9
                    or [r["delta"] for r in coefficients] != DELTAS
                    or [r["delta"] for r in metrics] != DELTAS
                    or len({r["fit_digest"] for r in metrics}) != 1):
                raise ValueError("Incomplete grid or fitted clouds changed across delta")
            if method == "YY":
                old = dict(block)
                receipt = old.pop("unchanged_YY_provenance", None)
                expected = {"previous_results_sha256": PREVIOUS_SHA, "previous_audit_sha256": PREVIOUS_AUDIT_SHA,
                            "reason": "YY same-point estimator and weights unchanged; not a PCA correspondence method"}
                if old != previous["methods"][method][geometry] or receipt != expected:
                    raise ValueError("YY was changed or relabeled as new PCA output")
            for i, (coefficient, metric) in enumerate(zip(coefficients, metrics)):
                if metric["query_count"] != 256 or metric["resolved_queries"] != 256:
                    raise ValueError("Original complete-query rule changed")
                for key in ("delta_average", "delta_infinity", "coarse_delta_infinity", "fine_coarse_abs_difference"):
                    if not math.isfinite(metric[key]) or metric[key] < 0:
                        raise ValueError("Invalid numerical deviation")
                if metric["delta_average"] > metric["delta_infinity"] + 2e-14:
                    raise ValueError("Average exceeds maximum")
                if i == 4 and metric["delta_infinity"] > config["gates"]["delta_zero_tolerance"]:
                    raise ValueError("Correct-sigma deviation is not zero")
                residual = coefficient["maximum_directional_residual" if method == "YY" else "true_cancellation_residual"]
                if not math.isfinite(residual) or residual < 0:
                    raise ValueError("Invalid cancellation diagnostic")
                if method != "YY" and (metric["reference_pca_radius"] != .15
                                        or metric["matching_pca_kernel"] != "K(t)=(1-t)_+^6"):
                    raise ValueError("Matching radius or kernel changed")
                expected_rows[method, geometry, DELTAS[i]] = coefficient, metric, previous["methods"][method][geometry]["geometry"][i]
    seen = set()
    for row in summary:
        key = row["method"], row["geometry"], float(row["delta"])
        if key not in expected_rows or key in seen:
            raise ValueError("Extra, duplicate or relabeled summary cell")
        seen.add(key)
        coefficient, metric, old = expected_rows[key]
        fields = {"sigma_hat": coefficient["sigma_hat"], "delta_infinity": metric["delta_infinity"],
                  "delta_average": metric["delta_average"], "old_delta_infinity": old["delta_infinity"],
                  "old_delta_average": old["delta_average"], "fine_coarse_abs_difference": metric["fine_coarse_abs_difference"],
                  "change_in_delta_infinity": metric["delta_infinity"] - old["delta_infinity"], "resolved_queries": 256}
        fields.update({k: coefficient.get(k, "") for k in ("axial_residual", "complementary_alpha_residual",
                      "complementary_beta_residual", "maximum_directional_residual")})
        fields["coefficient_residual"] = coefficient.get("true_cancellation_residual", "")
        if any(row.get(k) != "" if v == "" else not close(row.get(k), v) for k, v in fields.items()):
            raise ValueError("Summary differs from audited plotted values")
        if row["fits_recomputed"] != str(row["method"] != "YY"):
            raise ValueError("Unchanged YY summary provenance changed")
    if seen != set(expected_rows):
        raise ValueError("Missing E6 summary cells")


def verify_contract(root):
    contract = read(root / "FIGURE_INPUTS.json")
    required = {"reproduce_figures.py", "original_plotting.py", "data/AUDIT.json", "data/TABLE_PREPARATION.json",
                "data/config.json", "data/previous_results.json", "data/previous_audit.json",
                "data/validated_summary_with_previous.csv", "data/per_delta_query_outputs.csv", "results/e6_results.json"}
    if (contract.get("schema") != "E6_PCA_FIGURE_INPUTS_V1" or contract.get("formal_publication_permitted") is not False
            or contract.get("all_studies_complete") is not False or not required.issubset(contract.get("files", {}))):
        raise ValueError("Incomplete isolated E6 figure inputs")
    for relative, digest in contract["files"].items():
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or sha(path) != digest):
            raise ValueError("Figure source/data changed or escaped stage: " + relative)
    if sha(root / "original_plotting.py") != TEMPLATE_SHA or sha(root / "data/config.json") != CONFIG_SHA:
        raise ValueError("Original plot template or scientific config changed")
    if sha(root / "data/previous_results.json") != PREVIOUS_SHA or sha(root / "data/previous_audit.json") != PREVIOUS_AUDIT_SHA:
        raise ValueError("Original YY provenance evidence changed")
    report = read(root / "data/AUDIT.json")
    require_audit(report)
    if report["sha256_receipts"].get("project/E6/results/e6_results.json") != sha(root / "results/e6_results.json"):
        raise ValueError("Plotted payload is not the audited payload")
    preparation = read(root / "data/TABLE_PREPARATION.json")
    if (preparation.get("row_count") != 54 or preparation.get("per_delta_query_rows") != 18432
            or preparation.get("source_audit_sha256") != sha(root / "data/AUDIT.json")
            or preparation.get("formal_publication_permitted") is not False):
        raise ValueError("E6 preparation receipt changed")
    payload, previous = read(root / "results/e6_results.json"), read(root / "data/previous_results.json")
    if payload["config"] != read(root / "data/config.json"):
        raise ValueError("Embedded config does not match frozen file")
    with (root / "data/validated_summary_with_previous.csv").open(newline="", encoding="utf-8") as handle:
        validate_payload(payload, previous, list(csv.DictReader(handle)))
    return contract


def render(root):
    root = Path(root).resolve()
    verify_contract(root)
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib as mpl
    import numpy as np
    environment = {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": mpl.__version__}
    if not environment["python"].startswith("3.12.") or environment["numpy"] != "2.4.4" or environment["matplotlib"] != "3.11.1":
        raise ValueError("Use figure-only Python 3.12 / NumPy 2.4.4 / Matplotlib 3.11.1, not the scientific fitting environment")
    mpl.rcdefaults()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    spec = importlib.util.spec_from_file_location("e6_original_plot_template", root / "original_plotting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(root)  # byte-for-byte original presentation code; no population imports
    expected = {f"{stem}.{suffix}" for stem in STEMS for suffix in ("png", "pdf")}
    if {p.name for p in (root / "figures").glob("*") if p.suffix in (".pdf", ".png")} != expected:
        raise ValueError("E6 must export exactly the original three figure pairs")
    report = {"status": "ISOLATED_FIGURES_GENERATED_VISUAL_REVIEW_PENDING", "logical_figures": 3, "export_count": 6,
              "outputs": {name: sha(root / "figures" / name) for name in sorted(expected)}, "environment": environment,
              "original_template_sha256": TEMPLATE_SHA, "row_count": 54, "per_delta_query_rows": 18432,
              "metric": "Deviation from correct-sigma output, not distance to the true manifold",
              "historical_YY_recomputed": False, "formal_publication_permitted": False,
              "all_studies_complete": False, "visual_audit": "PENDING", "method_level_runtime_recorded": False,
              "input_contract_sha256": sha(root / "FIGURE_INPUTS.json")}
    (root / "figures/FIGURE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(render(parser.parse_args().root), indent=2))
