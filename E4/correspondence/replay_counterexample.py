"""Replay only Figure E.13 from saved roots; no numerical fitting or root solving."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from counterexample import A, r_float, g_float, plt
ROOT=Path(__file__).resolve().parent
OUT=None

def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def render(result,grid):
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'ps.fonttype':42})
    cp=np.array([float(t) for t in result['CP_angles']]);a=float(A)
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8))
    ax=axes[0];t=np.linspace(-.0006,.0006,5001)
    ax.plot(1.01*np.cos(t),1.01*np.sin(t)*1e6,c='#3B6EA8',label='Reference curve')
    ax.plot(r_float(t)*np.cos(t),r_float(t)*np.sin(t)*1e6,c='#D55E00',label='Oscillatory target')
    ax.scatter([1.01],[0],c='black',s=30,zorder=5)
    ax.plot([1.01,1.04+a],[0,0],c='#009E73',lw=1.4)
    ax.scatter([1.04+a],[0],c='#009E73',marker='s',s=45,label='ENF (local)',zorder=6)
    for tt in cp:
        x=r_float(tt)*np.cos(tt);y=r_float(tt)*np.sin(tt)*1e6
        ax.plot([1.01,x],[0,y],c='#8E44AD',ls='--',lw=1)
    ax.scatter(r_float(cp)*np.cos(cp),r_float(cp)*np.sin(cp)*1e6,c='#8E44AD',marker='D',s=45,label='Two CP points',zorder=6)
    ax.annotate(r'$y_1$',(1.01,0),xytext=(4,9),textcoords='offset points')
    ax.set_xlim(1.008,1.042);ax.set_ylim(-650,650)
    ax.set_xlabel('$x$');ax.set_ylabel(r'$y$ ($10^{-6}$)');ax.set_title('(A) Local geometry and target zoom')
    inset=ax.inset_axes([.25,.23,.60,.55])
    tt=np.linspace(-.00026,.00026,2001)
    inset.plot((r_float(tt)*np.cos(tt)-1.04)*1e6,r_float(tt)*np.sin(tt)*1e6,c='#D55E00')
    inset.scatter([a*1e6],[0],c='#009E73',marker='s',s=25)
    inset.scatter((r_float(cp)*np.cos(cp)-1.04)*1e6,r_float(cp)*np.sin(cp)*1e6,c='#8E44AD',marker='D',s=28)
    inset.set_xlabel(r'$(x-1.04)\,10^6$',fontsize=9);inset.set_ylabel(r'$y\,10^6$',fontsize=9)
    inset.tick_params(labelsize=9);inset.set_xlim(-3.4,3.4);inset.set_ylim(-280,280)
    inset.grid(alpha=.15)
    ax=axes[1]
    ax.plot(grid*1e4,(g_float(grid)-float(result['CP_g']))*1e7,c='#3B6EA8')
    ax.scatter(cp*1e4,[0,0],c='#8E44AD',marker='D',s=40)
    ax.scatter([0],[float(result['objective_gap'])*1e7],c='#009E73',marker='s',s=40)
    for x in cp:ax.axvline(x*1e4,c='#8E44AD',ls=':',alpha=.7)
    ax.set_xlabel(r'$\theta$ ($10^{-4}$ radians)');ax.set_ylabel(r'$(g(\theta)-g_{\min})\,10^7$')
    ax.set_title('(B) Continuous distance objective');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,bbox_to_anchor=(.52,1.02),fontsize=11)
    fig.subplots_adjust(top=.81,bottom=.15,left=.075,right=.99,wspace=.28)
    save(fig,'P1_oscillatory_CP_nonuniqueness')

def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    OUT=a.output.resolve()
    if OUT.exists():raise FileExistsError('Use a fresh output directory')
    source=ROOT/'counterexample'
    result=json.loads((source/'RESULT.json').read_bytes())
    with (source/'OBJECTIVE_VALUES.csv').open(encoding='utf8') as f:
        grid=np.array([float(row['theta']) for row in csv.DictReader(f)])
    OUT.mkdir(parents=True)
    render(result,grid)
    shutil.copyfile(source/'caption.tex',OUT/'caption.tex')
    print(OUT)

if __name__=='__main__':main()
