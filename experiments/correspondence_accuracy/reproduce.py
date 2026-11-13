"""Current-paper E4 local CP/ENF/NN/Oracle figure; no fitting."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'new_figures')
    p.add_argument('--mode',choices=['plot','verify'],default='plot')
    a=p.parse_args()
    if a.mode=='verify':
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'legacy_reproduce.py'),'--mode','verify'],check=True)
    else:
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'correspondence/plot_finite.py'),'--main-only','--output',str(a.output.resolve())],check=True)
if __name__=='__main__':main()
