"""Check archived paper evidence, including all recovered E5 finite packets."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_manuscript_result_alignment():
    spec=importlib.util.spec_from_file_location('paper_audit',ROOT/'verify_paper.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.audit()
    assert report['matched_claims']>=30
    assert report['unresolved']==[]
    assert report['E5_finite']['packets']==60
    assert report['E5_finite']['selected_plot_check']['status']=='MATCH'
    assert not (ROOT/'E6').exists() and not (ROOT/'E7').exists()
