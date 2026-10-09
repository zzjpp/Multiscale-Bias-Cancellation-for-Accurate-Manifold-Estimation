from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e6_misspec.plotting import build


def test_build_generates_exactly_three_figure_pairs(tmp_path):
    (tmp_path / "results").mkdir()
    shutil.copy2(ROOT / "results" / "e6_results.json", tmp_path / "results" / "e6_results.json")
    build(tmp_path)
    observed = sorted(path.name for path in (tmp_path / "figures").iterdir())
    expected = sorted([
        "e6_dr_noise_misspecification_main.pdf",
        "e6_dr_noise_misspecification_main.png",
        "e6_stsp_noise_misspecification_supplement.pdf",
        "e6_stsp_noise_misspecification_supplement.png",
        "e6_yy_noise_misspecification_supplement.pdf",
        "e6_yy_noise_misspecification_supplement.png",
    ])
    assert observed == expected
