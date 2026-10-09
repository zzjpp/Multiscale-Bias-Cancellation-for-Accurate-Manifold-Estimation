import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from e6_misspec.geometry import GEOMETRIES
from e6_misspec.population import fit_pair


@pytest.mark.parametrize("method", ["STSP", "DR"])
def test_matching_never_calls_true_normal(monkeypatch, method):
    geometry = GEOMETRIES[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("true normal consulted by non-Oracle fit")

    monkeypatch.setattr(type(geometry), "normal_basis", forbidden)
    queries = geometry.queries(256, .37)[:1]
    fits = fit_pair(
        geometry, method, .15, .3, .05, queries,
        {"dr_latent_count": 256, "stsp_latent_count": 256, "stsp_radial_order": 48},
    )
    assert not fits["failure_mask"].any(), fits["failures"]
    record = fits["query_records"][0]
    assert record["pca_radius"] == .15
    assert record["matching_kernel"] == "K(t)=(1-t)_+^6"
    basis = record["reference_basis"]
    assert basis.shape == (geometry.D, geometry.q)
    np.testing.assert_allclose(basis.T@basis, np.eye(geometry.q), atol=1e-13)
    np.testing.assert_array_equal(record["t1"], np.zeros(geometry.q))
    np.testing.assert_allclose(
        fits["coarse_points"][0], record["reference_point"]+basis@record["t2"],
        atol=2e-14, rtol=0,
    )
    assert record["projected_coarse_residual_norm"] <= 1e-10
