from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_release import validate


def test_public_release_contract() -> None:
    report = validate(ROOT)
    assert report["status"] == "PASS", report["errors"]
    assert report["experiments"] == ["E1", "E2", "E3", "E4", "E5", "E6"]

