"""Replay the stored deterministic counterexample, without solving roots again."""
import argparse
from pathlib import Path
from reproduce_plots import counter_only
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    counter_only(a.output.resolve())
