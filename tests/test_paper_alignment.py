"""Check actual archived paper evidence, including the declared missing data."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_manuscript_result_alignment():
    spec=importlib.util.spec_from_file_location('paper_audit',ROOT/'verify_paper.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.audit()
    assert report['matched_claims']>=30
    assert report['unresolved'][0]['status']=='MISSING_LOCAL_RAW_PACKETS'
    assert report['unresolved'][0]['expected']==60
    assert not (ROOT/'E6').exists() and not (ROOT/'E7').exists()
