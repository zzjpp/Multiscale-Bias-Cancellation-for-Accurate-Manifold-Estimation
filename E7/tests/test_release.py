from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e7_release import build_all


def test_frozen_aggregate_shapes() -> None:
    for name in ("stsp", "dr", "yy"):
        path = ROOT / "data" / f"e7_{name}_aggregated.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 56


def test_end_to_end_plot_smoke(tmp_path: Path) -> None:
    outputs = build_all(ROOT, tmp_path)
    assert [path.name for path in outputs] == [
        "e7_stsp_maximum_average_errors.pdf",
        "e7_stsp_maximum_average_errors.png",
        "e7_dr_maximum_average_errors.pdf",
        "e7_dr_maximum_average_errors.png",
        "e7_yy_maximum_average_errors.pdf",
        "e7_yy_maximum_average_errors.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in outputs)
