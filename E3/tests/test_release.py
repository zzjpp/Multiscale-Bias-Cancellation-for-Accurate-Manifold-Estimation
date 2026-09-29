from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / "config" / "e3_n12000.json").read_text(encoding="utf-8"))
STSP = json.loads((ROOT / "methods" / "stsp" / "E3_CONFIG.json").read_text(encoding="utf-8"))
DRYY = json.loads((ROOT / "methods" / "dr_yy" / "configs" / "e3_dr_yy_frozen.json").read_text(encoding="utf-8"))


def test_single_fixed_sample_size():
    assert DESIGN["n"] == STSP["n"] == DRYY["n"] == 12000


def test_common_design_axes():
    assert DESIGN["sigma"] == STSP["sigma"] == DRYY["sigma"] == 0.015
    assert DESIGN["h_grid"] == STSP["bandwidths_h2"] == DRYY["dr"]["bandwidths_h2"]
    assert min(DESIGN["h_grid"]) == 0.06
    assert max(DESIGN["h_grid"]) == 5.0
    assert DESIGN["geometries"] == [row["key"] for row in STSP["geometries"]]
    assert DESIGN["geometries"] == [row["key"] for row in DRYY["geometries"]]


def test_method_specific_scale_contracts():
    assert STSP["h1_ratio"] == DRYY["dr"]["h1_ratio"] == 0.5
    assert DRYY["yy"]["c0"] == 24.0
    r1 = [value * DRYY["sigma"] for value in DRYY["yy"]["central_kappas"]]
    assert len(r1) == len(DESIGN["h_grid"])
    assert all(abs(left - right) < 1e-12 for left, right in zip(r1, DESIGN["h_grid"]))


def test_release_has_no_backup_sources():
    names = [path.name for path in ROOT.rglob("*")]
    assert not any(".pre_" in name for name in names)
