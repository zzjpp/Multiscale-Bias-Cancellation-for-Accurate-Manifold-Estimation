"""Offline interface/negative checks and actual tiny CPU computations."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')

def invoke(entry,*args):
    return subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/entry),*args],cwd=ROOT,
                          env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=180)

@pytest.mark.parametrize('exp',['E1','E2','E3','E4','E5','E6'])
def test_documented_entrypoint(exp):
    result=invoke(exp+'/reproduce.py','--help')
    assert result.returncode==0,(result.stdout,result.stderr)
    assert (ROOT/exp/'README.md').is_file()

@pytest.mark.parametrize('exp',['E1','E2','E3'])
@pytest.mark.parametrize('method',['STSP','DR','YY'])
def test_actual_cpu_smoke(exp,method):
    entry=exp+('/smoke.py' if exp=='E1' else '/run_fits.py')
    args=['--method',method]+([] if exp=='E1' else ['--smoke'])
    result=invoke(entry,*args)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert 'smoke' in result.stdout.lower()

@pytest.mark.parametrize('exp',['E4','E6'])
def test_actual_controlled_pipeline(exp):
    result=invoke(exp+'/smoke.py')
    assert result.returncode==0,(result.stdout,result.stderr)

@pytest.mark.parametrize('exp',['E2','E3'])
def test_complete_saved_table(exp):
    result=invoke(exp+'/reproduce.py','--verify-only')
    assert result.returncode==0,(result.stdout,result.stderr)

@pytest.mark.parametrize('exp',['E2','E3'])
def test_incomplete_table_rejected(exp,tmp_path):
    table=tmp_path/'incomplete.csv'
    lines=(ROOT/exp/'data/summary.csv').read_text(encoding='utf-8-sig').splitlines()
    table.write_text('\n'.join(lines[:2]),encoding='utf-8')
    assert invoke(exp+'/reproduce.py','--table',str(table),'--verify-only').returncode!=0

@pytest.mark.parametrize('exp',['E2','E3'])
def test_partial_fit_grid_rejected(exp,tmp_path):
    assert invoke(exp+'/evaluate_fits.py','--input',str(tmp_path),'--output',str(tmp_path/'bad.csv')).returncode!=0
    assert not (tmp_path/'bad.csv').exists()

def test_frozen_common_queries_and_sources():
    result=invoke('E4/reproduce.py','--mode','verify')
    assert result.returncode==0,(result.stdout,result.stderr)

def test_all_sources_parse():
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))

def test_selected_version_and_no_standalone_E7():
    assert not (ROOT/'E7').exists()
    rows=(ROOT/'E3/data/summary.csv').read_text(encoding='utf-8-sig')
    assert 'Single_0p4r1' in rows and 'Single_0p8r1' not in rows
    assert 'corrugated_torus_r3' in rows

def test_distribution_manifest():
    spec=importlib.util.spec_from_file_location('release_validation',ROOT/'validate_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.validate()['verified_files']>300

def test_changed_member_rejected(tmp_path):
    member=tmp_path/'fake.py';member.write_text('changed',encoding='utf-8')
    (tmp_path/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(schema='PAPER62_PUBLIC_SOURCE_AND_DATA_V1',
        files={'fake.py':{'sha256':'0'*64,'size':7}})),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('release_validation_negative',ROOT/'validate_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError): module.validate(tmp_path)
