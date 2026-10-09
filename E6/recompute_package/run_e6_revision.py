"""Recompute affected E6 fits; retain explicitly unchanged YY provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT/"project/E6"
sys.path.insert(0, str(PACKAGE/"src"))

from e6_misspec import pipeline
from e6_misspec.geometry import Geometry
from e6_misspec.plotting import build


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((PACKAGE/"config.json").read_text())
    baseline = ROOT/"inputs/e6_previous_results.json"
    previous = json.loads(baseline.read_text())
    old_audit = json.loads((ROOT/"inputs/e6_previous_audit.json").read_text())
    assert old_audit["status"] == "PASS" and previous["config"] == config
    assert config["stsp"]["kernel_power"] == 6
    assert config["query_count"] == 256
    pipeline.freeze_hashes(PACKAGE)
    source_hash = sha(PACKAGE/"SOURCE_SHA256.txt")
    results = PACKAGE/"results"
    results.mkdir(exist_ok=True)
    marker = results/"REVISION_SOURCE.json"
    if marker.exists():
        assert json.loads(marker.read_text())["source_manifest_sha256"] == source_hash
    else:
        pipeline._write_json(marker, {"source_manifest_sha256": source_hash,
                                      "config_sha256": sha(PACKAGE/"config.json")})
    cache = results/"checkpoint_blocks"
    cache.mkdir(exist_ok=True)
    for key, block in previous["methods"]["YY"].items():
        path = cache/("yy_"+key+".json")
        reused = dict(block, unchanged_YY_provenance={
            "previous_results_sha256": sha(baseline),
            "previous_audit_sha256": sha(ROOT/"inputs/e6_previous_audit.json"),
            "reason": "YY same-point estimator and weights unchanged; not a PCA correspondence method",
        })
        if path.exists():
            assert json.loads(path.read_text()) == reused
        else:
            pipeline._write_json(path, reused)
    def forbidden(*args, **kwargs):
        raise AssertionError("non-Oracle E6 consulted true normal")
    Geometry.normal_basis = forbidden
    payload = pipeline.run(PACKAGE)
    payload["pca_revision_provenance"] = {
        "source_manifest_sha256": source_hash, "launcher_sha256": sha(__file__),
        "matching_kernel": "K(t)=(1-t)_+^6", "pca_radius": "h1",
        "covariance": "centered_kernel_weighted", "normal_api_poisoned": True,
        "DR_STSP_fits_recomputed": True, "YY_recomputed": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "backend": "cpu deterministic quadrature",
                        "threads": {name: os.environ.get(name) for name in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
    }
    pipeline._write_json(results/"e6_results.json", payload)
    base_audit = pipeline.audit(PACKAGE, payload)
    checked = 0
    for path in sorted(cache.glob("*_queries.json")):
        packet = json.loads(path.read_text())
        assert not any(packet["failure_mask"]), path
        assert packet["matching_pca_kernel"] == "K(t)=(1-t)_+^6"
        assert packet["reference_pca_radius"] == .15
        assert len(packet["query_records"]) == 256
        for i, row in enumerate(packet["query_records"]):
            assert row["query_id"] == i and row["status"] == "success"
            assert row["pca_radius"] == .15
            basis = np.asarray(row["reference_basis"])
            np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=2e-13)
            np.testing.assert_array_equal(row["t1"], np.zeros(basis.shape[1]))
            np.testing.assert_allclose(
                np.asarray(row["reference_point"])+basis@np.asarray(row["t2"]),
                packet["coarse_points"][i], rtol=0, atol=2e-14,
            )
            assert row["projected_coarse_residual_norm"] <= 1e-10
            checked += 1
    assert checked == 2*2*2*256
    report = {"status": "PASS" if base_audit["status"] == "PASS" else "FAIL",
              "base_audit": base_audit, "audited_query_records": checked,
              "source_manifest_sha256": source_hash, "matching_pca_protocol": "PASS",
              "YY_unchanged_source": sha(baseline)}
    pipeline._write_json(results/"PCA_FIBER_RELEASE_AUDIT.json", report)
    assert report["status"] == "PASS", report
    build(PACKAGE)
    print("E6_REVISED_RELEASE_AUDIT_PASS", checked, flush=True)


if __name__ == "__main__":
    main()
