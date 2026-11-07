"""Compact E1 paper figures from sealed existing data; no fitting or integration."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator,MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
E1=ROOT
NON=WORK/'nonuniform/E1'
spec=importlib.util.spec_from_file_location('unchanged_e1_plot_validators',E1/'src/e1_simulation/plot_sigma_path.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
COLORS=['#3B6EA8','#D55E00']
FIELDS=['single','multi']
SOURCES={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_inputs():
    data=E1/'data';p=json.loads((data/'provenance.json').read_bytes())
    for name,digest in p['data_sha256'].items():
        if sha(data/name)!=digest:raise ValueError('Original E1 input changed:'+name)
        SOURCES[str(data/name)]=digest
    rows=native.load_theory_rows(data/'e1_theory_path_errors.csv')
    yy=native.load_yy_theory_rows(data/'yy_theory_path_errors.csv')
    slopes=native.load_slope_rows(data/'e1_theory_path_slopes.csv')
    yyslopes=native.load_slope_rows(data/'yy_theory_path_slopes.csv')
    native.validate_theory_rows(rows);native.validate_yy_theory_rows(yy)
    native.validate_slope_rows(slopes);native.validate_yy_slope_rows(yyslopes)
    native.validate_registered_fits(rows,slopes);native.validate_registered_fits(yy,yyslopes)
    native.validate_numerical_evidence(data/'e1_theory_path_numerical_convergence.csv',data/'e1_theory_path_audit.json',data/'e1_theory_path_gpu_gate.json')
    native.validate_yy_numerical_evidence(data/'yy_theory_path_numerical_convergence.csv',data/'yy_theory_path_audit.json',data/'yy_theory_path_gpu_gate.json')
    audit_path=NON/'imports/actual_01/SCIENCE_AUDIT.json';audit=json.loads(audit_path.read_bytes())
    final=json.loads((NON/'imports/actual_01/FINAL_RECEIPT.json').read_bytes())
    if audit['status']!='PASS' or audit['failures'] or audit['errors'] or final['science_audit_sha256']!=sha(audit_path):raise ValueError('Missing valid existing nonuniform audit')
    SOURCES[str(audit_path)]=sha(audit_path)
    for name,digest in audit['source_hashes'].items():
        if sha(NON/name)!=digest:raise ValueError('Nonuniform science source changed:'+name)
    non=[dict(row,method='STSP') for row in audit['rows']]
    if len(non)!=42:raise ValueError('Full nonuniform grid required')
    return rows+yy,non

def caption(method,metric,nonuniform=False):
    label=r'$E_{\mathrm{avg}}$' if metric=='average' else r'$E_\infty$'
    text=f'Population geometric error {label} for {method} Single and Multi on six manifolds. '
    if method=='YY':
        text+=r'The radii are $r_0=3\sigma$, $r_1/\sigma\in\{1.25,5,8\}$ and $r_2=12\sigma\sqrt{\log(1/\sigma)}$, with central-radius Single. '
    else:text+=r'The bandwidths are $h_1=3\sigma$ and $h_2=6\sigma$; Single uses $h_2$. '
    if nonuniform:text+='The nonuniform surface densities have maximum/minimum ratio3. '
    text+='Noise level decreases from left to right. Slope annotations use all seven displayed noise levels, in legend order.'
    if metric=='average':text+=' Average errors are unweighted means on the fixed query grids.'
    return '\\caption{'+text+'}\n'

def save(fig,folder,stem,target,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    exports={}
    for ext in ['png','pdf']:
        file=folder/(stem+'.'+ext)
        fig.savefig(file,bbox_inches='tight',**({'dpi':300} if ext=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}))
        exports[ext]=dict(staged=str(file),target=str(target/(stem+'.'+ext)),sha256=sha(file))
    plt.close(fig)
    (folder/(stem+'_caption.tex')).write_text(metadata.pop('caption'),encoding='utf8')
    return dict(stem=stem,exports=exports,caption=str(folder/(stem+'_caption.tex')),**metadata)

def six_panel(rows,method,metric,folder,nonuniform=False):
    fields=[f'{x}_E_{metric}' for x in FIELDS]
    selected=[r for r in rows if r['method']==method]
    vals=[float(r[f]) for r in selected for f in fields]
    if len(selected)!=42 or not np.isfinite(vals).all() or min(vals)<=0:raise ValueError('Complete finite positive actual metrics required')
    limits=[min(1e-9,min(vals)/2),max(.1,max(vals)*1.4)] if nonuniform else [10**(np.floor(np.log10(min(vals)))-.25),10**(np.ceil(np.log10(max(vals)))+.12)]
    # Reversing x brings the largest errors beside the unboxed slope text.
    # Keep the plotted values below that text without moving any data point.
    log_lower=np.log10(limits[0])
    limits[1]=max(limits[1],10**(log_lower+(np.log10(max(vals))-log_lower)/.83))
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.2),sharex=True,sharey=True)
    records=[];x=np.arange(7)
    for axis,key in zip(axes.flat,native.GEOMETRIES):
        block=sorted([r for r in selected if r['geometry']==key],key=lambda r:float(r['sigma']))
        sigmas=[float(r['sigma']) for r in block]
        if tuple(sigmas)!=native.SIGMA_GRID:raise ValueError('Original seven sigma values required')
        slopes=[];series=[]
        for i,field in enumerate(fields):
            y=np.array([float(r[field]) for r in block])
            line,=axis.plot(x,y,color=COLORS[i],marker=['o','s'][i],linestyle=['-','--'][i],lw=1.7,ms=5)
            if not np.array_equal(line.get_ydata(),y):raise ValueError('Plotted values changed')
            slopes.append(float(np.polyfit(np.log(sigmas),np.log(y),1)[0]));series.append(y.tolist())
        axis.set_yscale('log');axis.set_ylim(*limits);axis.set_title(native.TITLES[key],pad=11)
        axis.set_xticks(x,['0.008','0.01','0.0125','0.016','0.02','0.025','0.03'])
        axis.set_xlim(6.3,-.3)
        axis.tick_params(axis='both',length=4,pad=5)
        plt.setp(axis.get_xticklabels(),rotation=40,ha='right')
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18)
        text=axis.text(.02,.96,'slopes: '+' / '.join(f'{v:.2f}' for v in slopes),transform=axis.transAxes,ha='left',va='top',fontsize=14)
        if text.get_bbox_patch() is not None or len(axis.patches):raise ValueError('Frames/bars forbidden')
        records.append(dict(geometry=key,sigma=sigmas,slopes=slopes,values=series))
    label=r'Single ($h_2$)' if method!='YY' else 'Single'
    fig.legend(handles=[Line2D([],[],color=COLORS[i],marker=['o','s'][i],linestyle=['-','--'][i],lw=1.7,ms=7,label=[label,'Multi'][i]) for i in range(2)],loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False)
    metric_tex=r'E_{\mathrm{avg}}' if metric=='average' else r'E_\infty'
    fig.supxlabel(r'Noise level $\sigma$',y=.012,fontsize=18)
    fig.supylabel(r'Population error $'+metric_tex+'$',x=.01,fontsize=18)
    fig.subplots_adjust(left=.10,right=.99,bottom=.18,top=.86,wspace=.23,hspace=.40)
    # Preserve the manuscript's original includegraphics filenames.
    stem=f'E1_{method}_{metric}_sigma'
    target=E1/'figures'
    if nonuniform:
        stem='e1_stsp_nonuniform_density_'+metric
        target=NON/'outputs'/('nonuniform_average' if metric=='average' else 'nonuniform')
    return save(fig,folder,stem,target,dict(caption=caption(method,metric,nonuniform),method=method,metric=metric,nonuniform=nonuniform,x_direction='descending',axis_limits=limits,series=records))

def circle(folder):
    p=E1/'data/circle_normalized_values.csv';SOURCES[str(p)]=sha(p)
    rows=list(csv.DictReader(p.open(encoding='utf8')))
    if len(rows)!=21:raise ValueError('Full original normalized-circle grid required')
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.7));records=[]
    for axis,method in zip(axes,['STSP','DR','YY']):
        block=sorted([r for r in rows if r['method']==method],key=lambda r:float(r['sigma']));x=[float(r['sigma']) for r in block]
        if tuple(x)!=native.SIGMA_GRID:raise ValueError('Circle sigma grid changed')
        ys=[]
        for i,(name,power) in enumerate([('single',2),('multi',4)]):
            y=[float(r[f'{name}_E_infinity_over_sigma{power}']) for r in block];ys.append(y)
            axis.plot(x,y,color=COLORS[i],marker=['o','s'][i],linestyle=['-','--'][i],lw=1.7,ms=5)
        axis.set_xscale('log');axis.set_xlim(.032,.0075)
        axis.set_ylim(0,{'STSP':2.0734093883842024,'DR':174.89108148705105,'YY':7.0185239776629675}[method])
        axis.xaxis.set_major_locator(FixedLocator(x));axis.xaxis.set_minor_locator(NullLocator());axis.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5));axis.set_title(method,pad=10)
        plt.setp(axis.get_xticklabels(),rotation=40,ha='right');axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18)
        records.append(dict(method=method,sigma=x,values=ys))
    fig.legend(handles=[Line2D([],[],color=COLORS[i],marker=['o','s'][i],linestyle=['-','--'][i],label=[r'Single: $E_\infty/\sigma^2$',r'Multi: $E_\infty/\sigma^4$'][i]) for i in range(2)],loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False)
    fig.supxlabel(r'Noise level $\sigma$',y=.012,fontsize=18);fig.supylabel(r'Normalized $E_\infty$',x=.01,fontsize=18)
    fig.subplots_adjust(left=.10,right=.99,bottom=.30,top=.72,wspace=.35)
    return save(fig,folder,'E1_circle_normalized_E_infinity_STSP_DR_YY_clear',E1/'figures',dict(series=records,x_direction='descending',caption=r'\caption{Normalized population maximum errors on the circle for STSP, DR and YY. Single is divided by $\sigma^2$ and Multi by $\sigma^4$. Noise level decreases from left to right. The scales and numerical settings are those of E1.}'+'\n'))

def main():
    p=argparse.ArgumentParser(description='Current E1 figures from saved audited values; no fits.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--all-diagnostics',action='store_true',help='Also draw the unselected legacy two-curve STSP maximum figure')
    a=p.parse_args();folder=a.output.resolve()
    if folder.exists():raise FileExistsError('Use a fresh output directory')
    rows,non=read_inputs()
    plt.rcParams.update({'font.size':15,'axes.titlesize':17,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':16,'pdf.fonttype':42,'ps.fonttype':42})
    figures=[]
    for method in ['STSP','DR','YY']:
        for metric in ['average','infinity']:
            if method=='STSP' and metric=='infinity' and not a.all_diagnostics:continue
            figures.append(six_panel(rows,method,metric,folder))
    figures.append(circle(folder))
    (folder/'MANIFEST.json').write_text(json.dumps(dict(figures=figures,inputs=SOURCES,new_fits=False,new_integrations=False),indent=2),encoding='utf8')
    print(json.dumps(dict(figures=len(figures),new_fits=False)))
if __name__=='__main__':main()
