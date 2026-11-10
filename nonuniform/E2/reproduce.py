"""Supplement Figures S17/S18: STSP nonuniform finite-sample errors, no refits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--table',type=Path,help='Fresh STSP-only evaluated table (288 rows); no DR/YY run required')
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    spec=importlib.util.spec_from_file_location('original_e2_table_and_style',ROOT.parent.parent/'E2/reproduce.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    table=a.table or ROOT/'data/NONUNIFORM_SUMMARY.csv'
    rows=original.read_rows(table,methods=('STSP',) if a.table else None);style=original.style
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':13,'ytick.labelsize':13,'axes.linewidth':.9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    exported=[]
    for metric in style.METRICS:
        fig,axes=plt.subplots(2,3,figsize=(9.6,6.6));fig.subplots_adjust(left=.13,right=.985,bottom=.17,top=.77,wspace=.40,hspace=.58)
        for i,(axis,(key,title)) in enumerate(zip(axes.flat,style.GEOMETRIES)):
            if 'torus' in title:title=title.replace('torus ','torus\n')
            axis.set_title(title,pad=6)
            for name in ['Single','Multi']:
                style.plot_series(axis,style.selected(rows,'STSP',key,metric,name),name)
                axis.lines[-1].set_markersize(5);axis.lines[-1].set_linewidth(1.6)
            axis.set_xscale('log');axis.set_yscale('log');axis.set_xlim(1700,590000)
            axis.xaxis.set_major_locator(FixedLocator(style.N));axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
            axis.xaxis.set_minor_locator(NullLocator());axis.yaxis.set_minor_locator(NullLocator())
            axis.spines[['top','right']].set_visible(False);axis.tick_params(direction='out',length=3.5,width=.8,pad=3)
            if i>=3:
                plt.setp(axis.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor');axis.set_xlabel(r'Sample size $n$',labelpad=4)
            else:axis.tick_params(labelbottom=False)
            if i in [0,3]:axis.set_ylabel(r'Average error $E_{\mathrm{avg}}$' if metric=='E_avg' else r'Maximum error $E_{\infty}$',labelpad=5)
        handles=[Line2D([],[],color=style.COLORS[name],marker=style.MARKERS[name],linestyle=style.STYLES[name],linewidth=1.6,markersize=6,label='STSP' if name=='Single' else 'GBC-STSP') for name in ['Single','Multi']]
        handles.append(Line2D([],[],linestyle='none',marker='o',markerfacecolor='white',markeredgecolor='#555555',markersize=6,label='Unreliable'))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.93),ncol=3,frameon=False,fontsize=15,handlelength=2,columnspacing=1.4)
        for ext in ['png','pdf']:
            f=out/f'E2_nonuniform_STSP_{metric}.{ext}'
            fig.savefig(f,**({'dpi':400} if ext=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}));exported.append(f.name)
        plt.close(fig)
    (out/'MANIFEST.json').write_text(json.dumps(dict(rows=len(rows),selected_method='STSP',figures=2,exports=exported,summary_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),new_fits=bool(a.table)),indent=2),encoding='utf8')
    print(json.dumps(dict(figures=2,exports=4)))
if __name__=='__main__':main()
