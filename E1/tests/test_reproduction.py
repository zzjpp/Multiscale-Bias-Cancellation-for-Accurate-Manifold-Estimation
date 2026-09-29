from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e1_simulation.plot_population import (  # noqa: E402
    GEOMETRIES,
    METHODS,
    group_rows,
    load_rows,
    reproduce,
    slope,
    validate_rows,
)


DATA = ROOT / "data" / "e1_population_errors.csv"
PROVENANCE = ROOT / "data" / "provenance.json"
CONFIG = ROOT / "config.json"


def test_data_contract_and_registered_slopes() -> None:
    rows = load_rows(DATA)
    validate_rows(rows)
    assert len(rows) == 115
    grouped = group_rows(rows)
    expected = json.loads(PROVENANCE.read_text(encoding="utf-8"))["expected_slopes"]
    registered = {(r["method"], r["geometry"], r["output"]): r["slope"] for r in expected}
    for method in METHODS:
        for geometry in GEOMETRIES:
            for output in ("single", "multi"):
                actual = slope(grouped[(method, geometry)], output)
                assert np.isclose(actual, registered[(method, geometry, output)], rtol=0.0, atol=1e-12)


def test_machine_readable_config_matches_registered_data() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert config["seed"] == 20260926
    assert config["protocol"] == provenance["protocol"]
    assert config["data_sha256"] == provenance["data_sha256"]


def test_end_to_end_figure_reproduction(tmp_path: Path) -> None:
    report = reproduce(DATA, PROVENANCE, tmp_path)
    assert report["status"] == "PASS"
    assert report["rows"] == 115
    assert len(report["output_sha256"]) == 12
    assert len(list(tmp_path.glob("*.pdf"))) == 6
    assert len(list(tmp_path.glob("*.png"))) == 6
    assert (tmp_path / "FIGURE_MANIFEST.json").is_file()
    assert all(path.stat().st_size > 10_000 for path in tmp_path.glob("*.pdf"))

