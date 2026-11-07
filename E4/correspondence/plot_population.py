"""Population correspondence order and normalized errors, from all actual cells."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator,FuncFormatter
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Isolated figure directory required')
    out.mkdir(exist_ok=True)
    stem='P1_population_correspondence_scaling'
    if (out/(stem+'.png')).exists() or (out/(stem+'.pdf')).exists():raise ValueError('Existing population figure; no blind overwrite')
    audit=json.loads((ROOT/'population_summary/AUDIT.json').read_bytes());assert audit['status']=='PASS'
    rows=list(csv.DictReader((ROOT/'population_summary/VALUES.csv').open()))
    slopes=list(csv.DictReader((ROOT/'population_summary/SLOPES.csv').open()))
    assert len(rows)==7
    x=np.array([float(r['sigma']) for r in rows])
    plt.rcParams.update({'font.size':13,'axes.titlesize':15,'legend.fontsize':13,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.6))
    handles=[];texts=[]
    for method,color,mark in [('ENF','#3B6EA8','o'),('CP','#8E44AD','D')]:
        yy=np.array([float(r[method+'_E_match']) for r in rows])
        h,=axes[0].plot(x,yy,color=color,marker=mark,lw=1.7,label=method)
        axes[1].plot(x,yy/x**4,color=color,marker=mark,lw=1.7)
        slope=float(next(s for s in slopes if s['method']==method and s['metric']=='E_match')['slope_all7'])
        texts.append(f'{method}: {slope:.4f}');handles.append(h)
    axes[0].set_yscale('log');axes[0].set_ylabel(r'$E_{\mathrm{match}}$')
    axes[1].set_ylabel(r'$E_{\mathrm{match}}/\sigma^4$')
    axes[0].text(.04,.94,'slopes: '+' / '.join(texts),va='top',transform=axes[0].transAxes,fontsize=11)
    for ax,title in zip(axes,['(A) Population matching accuracy','(B) Fourth-order normalization']):
        ax.set_xscale('log');ax.set_xlim(.0075,.032)
        ax.xaxis.set_major_locator(FixedLocator(x));ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        ax.tick_params(axis='x',rotation=40);ax.set_title(title,pad=10)
        ax.set_xlabel(r'Noise level $\sigma$');ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.legend(handles,[h.get_label() for h in handles],ncol=2,loc='upper center',frameon=False,bbox_to_anchor=(.52,1.0))
    fig.subplots_adjust(left=.09,right=.99,bottom=.23,top=.80,wspace=.28)
    fig.savefig(out/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    (out/'population_caption.tex').write_text(r'''\caption{Population correspondence scaling on $\gamma(t)=(1.5\cos t,\sin t,0.3\sin2t)$ with $h_1=3\sigma$ and $h_2=6\sigma$. At the same 32 fixed arclength query locations, ENF and continuous CP are compared with the true-normal target intersection. The expectation is evaluated deterministically using uniform-arclength latent quadrature and the analytically Gaussian-reduced $K_6$ STSP moments; no Monte Carlo sample is used. Production and refined latent/radial resolutions are $8192/160$ and $16384/224$. Slopes use all seven noise levels; the normalized panel tests the observed boundedness of $E_{\mathrm{match}}/\sigma^4$ over this finite range. These measurements do not constitute an asymptotic proof.}'''+'\n',encoding='utf8')
    (out/'POPULATION_PLOT_MANIFEST.json').write_text(json.dumps(dict(
        source_values_sha256=sha(ROOT/'population_summary/VALUES.csv'),renderer_sha256=sha(__file__),
        exports={f'{stem}.{ext}':sha(out/f'{stem}.{ext}') for ext in ['png','pdf']}),indent=2),encoding='utf8')
    print(out/stem)

if __name__=='__main__':main()
