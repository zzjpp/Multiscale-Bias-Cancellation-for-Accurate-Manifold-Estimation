"""Fresh expectation calculations using the original nonuniform fitting body."""
import argparse
from pathlib import Path
import study

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',choices=study.CONFIG['geometry_order'],default='circle_r2')
    p.add_argument('--sigma',type=float,default=.008)
    p.add_argument('--resolution',choices=['production','reference'],default='production')
    p.add_argument('--backend',choices=['cpu','gpu_cupy'],default='cpu')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never replace selected results')
    result=study.run_cell(a.geometry,a.sigma,a.resolution,a.backend,
        query_ids=[17] if a.smoke else None,reduced=a.smoke)
    study.write_fresh(a.output,result)
if __name__=='__main__':main()
