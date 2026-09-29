from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from e2plots import load_release_rows, load_style


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_data_contract() -> None:
    style = load_style(ROOT)
    rows = load_release_rows(ROOT, style)
    assert len(rows) == 480
    assert sorted({row["method"] for row in rows}) == ["DR", "STSP", "YY"]
    assert sorted({row["metric"] for row in rows}) == ["E_avg", "E_infty"]
    assert sorted({row["scale"] for row in rows}) == style["sample_sizes"]


def test_entrypoint_renders_exactly_six_logical_figures(tmp_path: Path) -> None:
    output = tmp_path / "results"
    completed = subprocess.run(
        [sys.executable, "reproduce.py", "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["logical_figures"] == 6
    assert report["exports"] == 12
    assert len(list(output.glob("*.png"))) == 6
    assert len(list(output.glob("*.pdf"))) == 6
    manifest = json.loads((output / "OUTPUT_SHA256.json").read_text(encoding="utf-8"))
    assert manifest["logical_figures"] == 6
    assert len(manifest["files"]) == 12

