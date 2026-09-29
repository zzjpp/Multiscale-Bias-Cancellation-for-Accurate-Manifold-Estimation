from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke(tmp_path: Path) -> None:
    output = tmp_path / "smoke-output"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--smoke", "--output-dir", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "Wrote" in completed.stdout
    assert (output / "e4_enf_vs_nn_equal_density.png").stat().st_size > 10_000
    assert (output / "e4_enf_vs_nn_equal_density.svg").stat().st_size > 10_000
    audit = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["smoke"] is True
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["method"] for row in rows} == {"Oracle", "ENF", "NN"}
