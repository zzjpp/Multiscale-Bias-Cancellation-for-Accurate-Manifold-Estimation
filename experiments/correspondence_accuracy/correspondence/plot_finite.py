"""Three-panel E4 extension and Eavg/Ebase supplement from actual ten-repeat data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
import numpy as np

ROOT=Path(__file__).resolve().parent
COLORS={'NN':'#D55E00','ENF':'#3B6EA8','CP':'#8E44AD','Oracle':'#666666'}
STYLES={'NN':'-','ENF':'-','CP':'-.','Oracle':'--'}
LABELS={'NN':'Discrete NN','ENF':'ENF','CP':'Continuous CP','Oracle':'Oracle'}
COUNTS=[32,36,40,44,48,52,56,60,64,80,96,112,128,256,512]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(folder=None):
    folder=folder or ROOT/'finite_summary'
    audit=json.loads((folder/'AUDIT.json').read_bytes());assert audit['status']=='PASS'
    source=folder/'SUMMARY.csv';assert sha(source)==audit['summary_sha256']
    return list(csv.DictReader(source.open(encoding='utf8')))

def series(table,method,key):
    if method=='NN':rr=[next(x for x in table if x['method']==method and int(x['candidate_count'])==m) for m in COUNTS]
    else:rr=[next(x for x in table if x['method']==method)]*len(COUNTS)
    return [[float(x[key+'_'+q]) for x in rr] for q in ['median','q1','q3']]

def panel(ax,table,key,label,scale='log',percent=False):
    x=np.array(COUNTS,dtype=float)
    for method in ['NN','ENF','CP','Oracle']:
        y,low,high=map(np.array,series(table,method,key))
        if percent:y*=100;low*=100;high*=100
        ax.plot(x,y,c=COLORS[method],ls=STYLES[method],lw=1.7,
                marker='o' if method=='NN' else None,ms=4,label=LABELS[method],zorder=4)
        ax.fill_between(x,low,high,color=COLORS[method],alpha=.10,linewidth=0,zorder=1)
    ax.set_xscale('log',base=2);ax.set_xlim(30,540)
    ax.xaxis.set_major_locator(FixedLocator([32,64,128,256,512]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x',rotation=35)
    if scale=='symlog':
        ax.set_yscale('symlog',linthresh=1e-7)
        ax.set_ylim(bottom=0)
    elif scale=='log':ax.set_yscale('log')
    ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)

def save(fig,out,stem):
    fig.savefig(out/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures')
    p.add_argument('--main-only',action='store_true',help='Render only the three-panel main figure')
    p.add_argument('--summary',type=Path,help='Explicit fresh evaluated summary directory')
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise ValueError('Use a fresh isolated figure directory')
    out.mkdir(parents=True);table=rows(a.summary)
    plt.rcParams.update({'font.size':13,'axes.titlesize':14,'legend.fontsize':13,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15.0,4.7),sharex=True)
    panel(axes[0],table,'E_infinity',r'Final error $E_\infty$')
    panel(axes[1],table,'E_match',r'Matching error $E_{\mathrm{match}}$',scale='symlog')
    panel(axes[2],table,'total_seconds','Total computation time (s)')
    for ax,title in zip(axes,['(A) Final maximum\ngeometric error','(B) Direct correspondence\nerror','(C) Total elapsed\ntime']):ax.set_title(title,pad=10)
    fig.legend(handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m],lw=1.7,label='Local CP' if m=='CP' else LABELS[m]) for m in ['NN','ENF','CP','Oracle']],
               loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.52,.995))
    fig.supxlabel(r'NN candidate count $M_{\mathrm{match}}$',y=.01,fontsize=16)
    fig.subplots_adjust(top=.75,left=.066,right=.99,bottom=.24,wspace=.32)
    save(fig,out,'e4_nn_density_accuracy')
    manifest=dict(status='ACTUAL_TEN_REPEAT_DATA',finite_summary_sha256=sha((a.summary or ROOT/'finite_summary')/'SUMMARY.csv'),
        renderer_sha256=sha(__file__),exports={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']})
    (out/'FINITE_PLOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(out)

if __name__=='__main__':main()
