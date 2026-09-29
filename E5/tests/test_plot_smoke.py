from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_two_figures import finite_figure, population_figure


ROOT = Path(__file__).resolve().parents[1]
POPULATION = json.loads((ROOT / "config" / "population.json").read_text(encoding="utf-8"))
FINITE = json.loads((ROOT / "config" / "finite_sample.json").read_text(encoding="utf-8"))
METHODS = FINITE["methods"]


def test_both_figure_builders_write_png_and_pdf(tmp_path):
    finite_rows = []
    for metric in ("E_inf", "E_avg"):
        for sample_size in FINITE["sample_sizes"]:
            for index, method in enumerate(METHODS, start=1):
                median = index / sample_size**0.25
                finite_rows.append({
                    "metric": metric, "sample_size": sample_size, "method": method,
                    "count": FINITE["repeats"], "median": median,
                    "q1": 0.9 * median, "q3": 1.1 * median,
                })
    finite_figure(finite_rows, tmp_path, FINITE)
    assert (tmp_path / "figure_e5_finite_sample_dual_metrics.png").is_file()
    assert (tmp_path / "figure_e5_finite_sample_dual_metrics.pdf").is_file()

    packet_dir = tmp_path / "results" / "population" / "packets"
    packet_dir.mkdir(parents=True)
    for sigma in POPULATION["sigma_grid"]:
        summaries = {}
        for method in METHODS:
            power = 4 if method == "Directional" else 2
            summaries[method] = {
                "geometric_max": sigma**power,
                "axial_normal_max": 0.8 * sigma**power,
                "complementary_normal_max": 0.6 * sigma**power,
            }
        payload = {"result": {"sigma": sigma, "summaries": summaries}}
        (packet_dir / f"population_sigma_{sigma:.6f}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    slopes = population_figure(tmp_path / "results", tmp_path, POPULATION)
    assert abs(slopes["Single"]["geometric_max"] - 2.0) < 1e-10
    assert abs(slopes["Directional"]["geometric_max"] - 4.0) < 1e-10
    assert (tmp_path / "figure_e5_population_directional_ablation.png").is_file()
    assert (tmp_path / "figure_e5_population_directional_ablation.pdf").is_file()
