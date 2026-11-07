"""Current main + supplementary experimental figures from saved results, no fitting."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from validate_release import validate
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    validate(ROOT);out.mkdir(parents=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    commands=[('E1/reproduce.py',out/'E1',[]),('E1/weight_control/render_and_report.py',out/'E1_weight_control',['--figure-only']),('E2/reproduce.py',out/'E2',[]),('E3/reproduce.py',out/'E3',[]),('E4/reproduce.py',out/'E4',[]),('E4/correspondence/replay_counterexample.py',out/'counterexample',[]),('nonuniform/E1/reproduce.py',out/'nonuniform_E1',[]),('nonuniform/E2/reproduce.py',out/'nonuniform_E2',[])]
    for script,destination,args in commands:
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/script),'--output',str(destination),*args],cwd=ROOT,env=env,check=True)
    provenance=json.loads((ROOT/'E5/data/PROVENANCE.json').read_bytes())['reference_files'];(out/'E5').mkdir()
    for name,digest in provenance.items():
        src=ROOT/'E5/paper_figures'/name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError('Changed selected E5 export')
        shutil.copyfile(src,out/'E5'/name)
    # Only the geometry/objective counterexample is in Figure S13.
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ['.png','.pdf','.svg']}
    paper_files={n:h for n,h in files.items() if 'P1_CP_branch_switch' not in n}
    assert len(paper_files)==66,(len(paper_files),list(paper_files))
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(dict(schema='CURRENT_PAPER_SAVED_RESULTS_V2',outputs=paper_files,additional_diagnostics={n:h for n,h in files.items() if n not in paper_files},logical_experimental_plots=33,new_fits=False),indent=2),encoding='utf8')
    print(json.dumps(dict(experimental_plots=33,paper_exports=len(paper_files),additional_diagnostic_exports=len(files)-len(paper_files))))
if __name__=='__main__':main()
