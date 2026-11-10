"""Six E2 figure pairs from the final six-manifold table; no fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'plotting'))
import style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

def read_rows(table=None):
    with (table or ROOT/'data/summary.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    expected={(m,g,e,t,n) for m in style.METHODS for g,_ in style.GEOMETRIES
              for e in (('Single','Multi') if m=='YY' else ('Single_h1','Single_h2','Multi'))
              for t in style.METRICS for n in style.N}
    seen=set()
    for r in rows:
        value=float(r['scale'])
        if not value.is_integer(): raise ValueError('Nonintegral E2 sample size')
        key=(r['method'],r['geometry'],r['estimator'],r['metric'],int(value))
        if key in seen or key not in expected: raise ValueError('Missing/duplicate E2 design cell')
        seen.add(key)
        if int(r['query_count_per_repeat'])!=144: raise ValueError('E2 query budget changed')
        reliable=int(r['complete_repeats'])>=18 and float(r['query_failure_rate'])<=.01
        if (r['reliable']=='True')!=reliable: raise ValueError('Original E2 reliability rule changed')
    if seen!=expected or len(rows)!=768: raise ValueError('Complete six-manifold table required')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--table',type=Path,help='Optional complete NEW evaluated table; saved paper data untouched')
    args=p.parse_args();rows=read_rows(args.table)
    if args.verify_only: print(json.dumps(dict(rows=768,figures=6)));return
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.8,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    outputs=[]
    for method in style.METHODS:
        for metric in style.METRICS:
            fig,axes=plt.subplots(2,3,figsize=(7.2,4.6))
            fig.subplots_adjust(left=.095,right=.985,bottom=.13,top=.84,wspace=.38,hspace=.38)
            fig.suptitle(method,fontsize=11,y=.992)
            for index,(axis,(key,title)) in enumerate(zip(axes.flat,style.GEOMETRIES)):
                axis.set_title(title,pad=4)
                for name in ('Single','Multi'): style.plot_series(axis,style.selected(rows,method,key,metric,name),name)
                axis.set_xscale('log');axis.set_yscale('log');axis.set_xlim(1700,590000)
                axis.xaxis.set_major_locator(FixedLocator(style.N))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
                axis.xaxis.set_minor_locator(NullLocator());axis.yaxis.set_minor_locator(NullLocator())
                axis.spines[['top','right']].set_visible(False);axis.tick_params(direction='out',length=3,width=.7,pad=2)
                if index>=3:
                    plt.setp(axis.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
                    axis.set_xlabel(r'Sample size $n$',labelpad=3)
                else: axis.tick_params(labelbottom=False)
                if index in (0,3):
                    axis.set_ylabel(r'Average error $E_{\mathrm{avg}}$' if metric=='E_avg' else r'Maximum error $E_{\infty}$',labelpad=4)
            handles=[Line2D([],[],color=style.COLORS[name],marker=style.MARKERS[name],linestyle=style.STYLES[name],
                linewidth=1.4,markersize=4,label=method if name=='Single' else f'GBC-{method}') for name in ('Single','Multi')]
            handles.append(Line2D([],[],linestyle='none',marker='o',markerfacecolor='white',markeredgecolor='#555555',markersize=4,label='Unreliable'))
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.956),ncol=3,frameon=False,fontsize=9,handlelength=2,columnspacing=1.8)
            for extension in ('png','pdf'):
                path=args.output/f'E2_{method}_{metric}.{extension}'
                fig.savefig(path,**({'dpi':600} if extension=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}))
                outputs.append(path.name)
            plt.close(fig)
    (args.output/'EXPORTS.json').write_text(json.dumps(dict(outputs=outputs,rows=768,displayed_cells=576,
        independent_fine_Single_preserved_in_data=True,green_background=False,new_fits=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=6,exports=12)))

if __name__=='__main__': main()
