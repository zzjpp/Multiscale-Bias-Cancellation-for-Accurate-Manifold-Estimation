"""Actual archived statistics and negative cases; never modifies original evidence."""
import json
from pathlib import Path
import shutil

import pytest

from scripts.verify_historical_finite import audit, check_selected_svg

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def checked():
    return audit()


def test_recovered_grid_and_actual_figure_statistics(checked):
    assert checked['packets']==60
    assert checked['archived_estimator_query_outcomes']==61440
    assert checked['summary_rows']==24
    assert checked['total_query_failures']==0
    assert checked['selected_plot_check']['median_values']==24
    assert checked['selected_plot_check']['quartile_boundary_values']==48


def test_missing_packet_grid_is_rejected(tmp_path):
    (tmp_path/'packets').mkdir()
    with pytest.raises(ValueError,match='packet grid'):
        audit(data_root=tmp_path)


def test_changed_packet_bytes_are_rejected(tmp_path):
    source=ROOT/'data/finite'
    (tmp_path/'packets').mkdir()
    shutil.copyfile(source/'original_config.json',tmp_path/'original_config.json')
    for p in (source/'packets').glob('e5_directional_n*.json'):
        shutil.copyfile(p,tmp_path/'packets'/p.name)
    changed=tmp_path/'packets/e5_directional_n0240000_r000.json'
    data=changed.read_bytes()
    changed.write_bytes(b' '+data[1:])
    with pytest.raises(ValueError,match='Changed historical E5 packet'):
        audit(data_root=tmp_path)


def test_incorrect_median_is_rejected_by_actual_svg(checked):
    summary=[dict(row) for row in checked['summaries']]
    summary[0]['median']*=1.01
    with pytest.raises(ValueError,match='SVG disagrees'):
        check_selected_svg(summary,ROOT/'paper_figures/figure_e5_finite_sample_dual_metrics.svg')


def test_incorrect_iqr_is_rejected_by_actual_svg(checked):
    summary=[dict(row) for row in checked['summaries']]
    summary[1]['q1']*=.99
    with pytest.raises(ValueError,match='SVG disagrees'):
        check_selected_svg(summary,ROOT/'paper_figures/figure_e5_finite_sample_dual_metrics.svg')
