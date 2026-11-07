"""The two supplement E1 nonuniform figures from already audited population values."""
import argparse
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    source=ROOT.parent.parent/'E1/presentation.py'
    spec=importlib.util.spec_from_file_location('paper_nonuniform_e1_presentation',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,rows=module.read_inputs()
    module.plt.rcParams.update({'font.size':15,'axes.titlesize':17,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':16,'pdf.fonttype':42,'ps.fonttype':42})
    figures=[module.six_panel(rows,'STSP',metric,out,True) for metric in ['average','infinity']]
    (out/'MANIFEST.json').write_text(json.dumps(dict(figures=figures,new_fits=False,new_integrations=False),indent=2),encoding='utf8')
    print(json.dumps(dict(figures=2,new_fits=False)))
if __name__=='__main__':main()
