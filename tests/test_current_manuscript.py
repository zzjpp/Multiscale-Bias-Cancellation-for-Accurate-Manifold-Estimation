"""Checks for the actual current figure mapping and preserved scientific source bytes."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_every_current_main_and_supplement_figure_exists():
    figures=json.loads((ROOT/'docs/FIGURE_MAP.json').read_bytes())['figures']
    assert {x['figure'] for x in figures}=={f'Figure {i}' for i in range(1,9)}|{f'Figure S{i}' for i in range(1,19)}
    for row in figures:
        assert all((ROOT/p).is_file() for p in row['files'])

def test_shared_fit_weight_control_bindings():
    base=ROOT/'E1/weight_control'
    for row in json.loads((base/'SOURCE_BINDING.json').read_bytes())['files']:
        p=base/row['copy']
        assert p.stat().st_size==row['size']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    with (base/'analysis/SLOPES.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==24
    assert all(abs(float(r['slope_all_7'])-2)<.001 for r in rows if r['method']=='naive')

def test_current_cp_evidence_has_all_shared_queries():
    base=ROOT/'E4/correspondence'
    a=json.loads((base/'finite_summary/AUDIT.json').read_bytes())
    assert a['status']=='PASS' and a['CP_points_checked']==320 and a['CP_is_not_discrete_NN']
    for name,digest in a['packet_sha256'].items():
        assert hashlib.sha256((base/'finite_sample'/name).read_bytes()).hexdigest()==digest

def test_nonuniform_original_producer_contract_is_complete():
    base=ROOT/'nonuniform/E2'
    j=json.loads((base/'PRODUCER_MANIFEST.json').read_bytes())
    for name,item in j['members'].items():
        p=base/name
        assert p.stat().st_size==item['size']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    original=json.loads((base/'data/ORIGINAL_PRODUCER_MANIFEST.json').read_bytes())
    for name,item in original['members'].items():
        if name=='README.md':continue  # Public documentation, explicitly versioned separately.
        p=base/name
        assert p.stat().st_size==item['size']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']

def test_e2_population_overlay_stays_withdrawn():
    assert not (ROOT/'E2/population_reference').exists()
    assert not (ROOT/'E2/data/population_baselines.csv').exists()
    assert 'population-baselines' not in (ROOT/'E2/reproduce.py').read_text(encoding='utf8')
