from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import publish_paper_figures

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_publisher_writes_only_frozen_figure_sources(tmp_path, monkeypatch):
    output = tmp_path / "figures"
    monkeypatch.setattr(publish_paper_figures, "OUTPUT", output)
    publish_paper_figures.main()
    expected = PROVENANCE["reference_files"]
    observed = {path.name for path in output.iterdir() if path.is_file()}
    assert observed == set(expected) | {"FIGURE_MANIFEST.json"}
    for name, digest in expected.items():
        assert sha256(output / name) == digest
    manifest = json.loads((output / "FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert len(manifest["figures"]) == 4