"""Replay only the final arxiv_final.pdf experimental figures from saved results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from validate_release import validate
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    validate(ROOT)
    out.mkdir(parents=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    commands=[('E1/reproduce.py','E1',[]),('E1/weight_control/render_and_report.py','E1_weight_control',['--figure-only']),
        ('E2/reproduce.py','E2',[]),('E3/reproduce.py','E3',[]),('E4/reproduce.py','E4',[]),
        ('E4/correspondence/replay_counterexample.py','counterexample',[]),
        ('nonuniform/E1/reproduce.py','nonuniform_E1',[]),('nonuniform/E2/reproduce.py','nonuniform_E2',[])]
    for script,destination,args in commands:
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/script),'--output',str(out/destination),*args],cwd=ROOT,env=env,check=True)
    provenance=json.loads((ROOT/'E5/data/PROVENANCE.json').read_bytes())['reference_files']
    (out/'E5').mkdir()
    for name,digest in provenance.items():
        src=ROOT/'E5/paper_figures'/name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError('Changed selected E5 export')
        shutil.copyfile(src,out/'E5'/name)
    mapping=json.loads((ROOT/'docs/FIGURE_MAP.json').read_bytes())
    selected=[p for row in mapping['figures'] if row['experiment']!='schematic' for p in row['files']]
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in {'.png','.pdf','.svg'}}
    if len(files)!=mapping['experimental_exports'] or {Path(n).name for n in files}!={Path(n).name for n in selected}:
        raise ValueError('Replay export inventory differs from the final manuscript figure map')
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(dict(schema='ARXIV_FINAL_SAVED_RESULTS_V1',
        manuscript_sha256=mapping['manuscript_sha256'],outputs=files,logical_experimental_plots=mapping['logical_experimental_plots'],
        new_fits=False,evidence='Saved-data rendering; E5 finite exports are validated copies, not raw-statistic reconstruction.'),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(experimental_plots=mapping['logical_experimental_plots'],paper_exports=len(files),new_fits=False)))

if __name__=='__main__':main()
