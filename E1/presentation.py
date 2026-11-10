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
COLORS=['#94989C','#C54F2D']
METHOD_NAMES={'STSP':'Successive tangent-space projection (STSP)',
              'DR':'Density ridge (DR)','YY':'Yao-Yau (YY)'}
FIELDS=['single','multi']
SOURCES={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def configure_style():
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':14,'axes.titlesize':15.5,'xtick.labelsize':12.5,
                         'ytick.labelsize':13,'legend.fontsize':13,
                         'pdf.fonttype':42,'ps.fonttype':42})

def style_axis(axis,panel,title):
    axis.set_title(title,pad=10)
    axis.tick_params(axis='both',length=3,width=.7,pad=4)
    plt.setp(axis.get_xticklabels(),rotation=0,ha='center')
    axis.spines[['top','right']].set_visible(False)
    for side in ['left','bottom']:
        axis.spines[side].set_linewidth(.7)
        axis.spines[side].set_color('#55595D')
    axis.grid(False)
    axis.text(-.115,1.06,f'({chr(97+panel)})',transform=axis.transAxes,
              va='bottom',ha='left',fontsize=13.5,fontweight='bold')

def draw_series(axis,x,y,index):
    return axis.plot(x,y,color=COLORS[index],marker=['o','s'][index],
                     linestyle=['-','--'][index],lw=[1.0,1.55][index],ms=4.2,mew=.8,
                     mfc='white' if index==0 else COLORS[index])[0]

def legend_handles(labels):
    return [Line2D([],[],color=COLORS[i],marker=['o','s'][i],
                   linestyle=['-','--'][i],lw=[1.0,1.55][i],ms=4.5,mew=.8,
                   mfc='white' if i==0 else COLORS[i],label=labels[i]) for i in range(2)]

def validate_annotation_layout(fig,annotations):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for axis,text in annotations:
        bounds=text.get_window_extent(renderer).expanded(1.02,1.08)
        if not (axis.bbox.contains(bounds.x0,bounds.y0) and axis.bbox.contains(bounds.x1,bounds.y1)):
            raise ValueError('Slope label outside panel')
        for line in axis.lines:
            if line.get_transform().transform_path(line.get_path()).intersects_bbox(bounds,filled=False):
                raise ValueError('Slope label touches an existing data curve')

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
    text=f'Population geometric error {label} for {method} and GBC-{method} on six manifolds. '
    text+=f'The legend labels {METHOD_NAMES[method]} and Geometric bias cancellation (GBC) refer to the baseline {method} estimator and GBC-{method}, respectively. '
    if method=='YY':
        text+=r'The radii are $r_0=3\sigma$, $r_1/\sigma\in\{1.25,5,8\}$ and $r_2=12\sigma\sqrt{\log(1/\sigma)}$, with the central-radius YY baseline. '
    else:text+=r'The bandwidths are $h_1=3\sigma$ and $h_2=6\sigma$; METHOD uses $h_2$. '.replace('METHOD',method)
    if nonuniform:text+='The nonuniform surface densities have maximum/minimum ratio3. '
    text+=r'Noise level decreases from left to right. The lower-left slopes $p_{\mathrm{METHOD}}$ and $p_{\mathrm{GBC\!-\!METHOD}}$ use all seven displayed noise levels, in legend order.'.replace('METHOD',method)
    if metric=='average':text+=' Average errors are unweighted means on the fixed query grids.'
    return '\\caption{'+text+'}\n'

def save(fig,folder,stem,target,metadata,png_stem=None):
    folder.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    for legend in fig.legends:
        box=legend.get_window_extent(fig.canvas.get_renderer())
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1), 'Legend exceeds canvas'
    exports={}
    for ext in ['png','pdf']:
        export_stem=png_stem if ext=='png' and png_stem else stem
        file=folder/(export_stem+'.'+ext)
        fig.savefig(file,bbox_inches='tight',**({'dpi':300} if ext=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}))
        exports[ext]=dict(staged=str(file),target=str(target/(export_stem+'.'+ext)),sha256=sha(file))
    plt.close(fig)
    (folder/(stem+'_caption.tex')).write_text(metadata.pop('caption'),encoding='utf8')
    return dict(stem=stem,exports=exports,caption=str(folder/(stem+'_caption.tex')),**metadata)

def six_panel(rows,method,metric,folder,nonuniform=False):
    configure_style()
    fields=[f'{x}_E_{metric}' for x in FIELDS]
    selected=[r for r in rows if r['method']==method]
    vals=[float(r[f]) for r in selected for f in fields]
    if len(selected)!=42 or not np.isfinite(vals).all() or min(vals)<=0:raise ValueError('Complete finite positive actual metrics required')
    limits=[min(1e-9,min(vals)/2),max(.1,max(vals)*1.4)] if nonuniform else [10**(np.floor(np.log10(min(vals)))-.25),10**(np.ceil(np.log10(max(vals)))+.12)]
    # Retain the selected figure's original numerical display limits.
    log_lower=np.log10(limits[0])
    limits[1]=max(limits[1],10**(log_lower+(np.log10(max(vals))-log_lower)/.83))
    fig,axes=plt.subplots(2,3,figsize=(14.2,7.7),sharex=True,sharey=True)
    records=[];x=np.arange(7);annotations=[]
    for panel,(axis,key) in enumerate(zip(axes.flat,native.GEOMETRIES)):
        block=sorted([r for r in selected if r['geometry']==key],key=lambda r:float(r['sigma']))
        sigmas=[float(r['sigma']) for r in block]
        if tuple(sigmas)!=native.SIGMA_GRID:raise ValueError('Original seven sigma values required')
        slopes=[];series=[]
        for i,field in enumerate(fields):
            y=np.array([float(r[field]) for r in block])
            line=draw_series(axis,x,y,i)
            if not np.array_equal(line.get_ydata(),y):raise ValueError('Plotted values changed')
            slopes.append(float(np.polyfit(np.log(sigmas),np.log(y),1)[0]));series.append(y.tolist())
        axis.set_yscale('log');axis.set_ylim(*limits)
        axis.set_xticks([6,4,1,0],['0.03','0.02','0.01','0.008'])
        axis.set_xlim(6.3,-.3)
        style_axis(axis,panel,native.TITLES[key])
        label=rf'$p_{{\mathrm{{{method}}}}}={slopes[0]:.2f},\quad p_{{\mathrm{{GBC\!-\!{method}}}}}={slopes[1]:.2f}$'
        text=axis.text(.035,.035,label,transform=axis.transAxes,ha='left',va='bottom',fontsize=12)
        annotations.append((axis,text))
        if text.get_bbox_patch() is not None or len(axis.patches):raise ValueError('Frames/bars forbidden')
        records.append(dict(geometry=key,sigma=sigmas,slopes=slopes,values=series))
    fig.legend(handles=legend_handles([METHOD_NAMES[method],'Geometric bias cancellation (GBC)']),
               loc='upper center',bbox_to_anchor=(.535,.995),ncol=2,frameon=False,
               columnspacing=2.0,handletextpad=.65)
    header=(r'$r_0=3\sigma,\quad r_1/\sigma\in\{1.25,5,8\},\quad r_2=12\sigma\sqrt{\log(1/\sigma)}$'
            if method=='YY' else r'$h_1=3\sigma,\quad h_2=6\sigma$')
    fig.text(.535,.913,header,ha='center',fontsize=14)
    metric_tex=r'E_{\mathrm{avg}}' if metric=='average' else r'E_\infty'
    fig.supxlabel(r'Noise level $\sigma$',y=.025,fontsize=16)
    fig.supylabel(r'Population error $'+metric_tex+'$',x=.014,fontsize=16)
    fig.subplots_adjust(left=.085,right=.99,bottom=.12,top=.825,wspace=.22,hspace=.42)
    validate_annotation_layout(fig,annotations)
    # Preserve the manuscript's original includegraphics filenames.
    stem=f'E1_{method}_{metric}_sigma'
    target=E1/'figures'
    if nonuniform:
        stem='e1_stsp_nonuniform_density_'+metric
        target=NON/'outputs'/('nonuniform_average' if metric=='average' else 'nonuniform')
    return save(fig,folder,stem,target,dict(caption=caption(method,metric,nonuniform),method=method,metric=metric,nonuniform=nonuniform,x_direction='descending',x_spacing='original equally spaced sigma positions',axis_limits=limits,series=records,slope_text_curve_collisions=False))

def circle(folder, rows=None):
    configure_style()
    if rows is None:
        p=E1/'data/circle_normalized_values.csv';SOURCES[str(p)]=sha(p)
        rows=list(csv.DictReader(p.open(encoding='utf8')))
    if len(rows)!=21:raise ValueError('Full original normalized-circle grid required')
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.1));records=[]
    for panel,(axis,method) in enumerate(zip(axes,['STSP','DR','YY'])):
        block=sorted([r for r in rows if r['method']==method],key=lambda r:float(r['sigma']));x=[float(r['sigma']) for r in block]
        if tuple(x)!=native.SIGMA_GRID:raise ValueError('Circle sigma grid changed')
        ys=[]
        for i,(name,power) in enumerate([('single',2),('multi',4)]):
            y=[float(r[f'{name}_E_infinity_over_sigma{power}']) for r in block];ys.append(y)
            line=draw_series(axis,x,y,i)
            if not np.array_equal(line.get_ydata(),y):raise ValueError('Normalized values changed')
        axis.set_xscale('log');axis.set_xlim(.032,.0075)
        axis.set_ylim(0,{'STSP':2.0734093883842024,'DR':174.89108148705105,'YY':7.0185239776629675}[method])
        axis.xaxis.set_major_locator(FixedLocator([.03,.02,.01,.008]));axis.xaxis.set_minor_locator(NullLocator());axis.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(axis,panel,method)
        records.append(dict(method=method,sigma=x,values=ys))
    fig.legend(handles=legend_handles([r'STSP/DR/YY: $E_\infty/\sigma^2$',r'GBC: $E_\infty/\sigma^4$']),
               loc='upper center',bbox_to_anchor=(.535,.995),ncol=2,frameon=False,
               columnspacing=2.0,handletextpad=.65)
    fig.supxlabel(r'Noise level $\sigma$',y=.025,fontsize=16);fig.supylabel(r'Normalized $E_\infty$',x=.014,fontsize=16)
    fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.70,wspace=.30)
    return save(fig,folder,'E1_circle_normalized_E_infinity_STSP_DR_YY_clear',E1/'figures',dict(series=records,x_direction='descending',caption=r'\caption{Normalized population maximum errors on the circle for STSP, DR and YY. The baseline STSP, DR and YY curves are divided by $\sigma^2$, and their GBC variants by $\sigma^4$. Noise level decreases from left to right. The scales and numerical settings are those of E1.}'+'\n'),png_stem='E1_circle_r2_normalized_E_infinity_STSP_DR_YY')

def main():
    p=argparse.ArgumentParser(description='Current E1 figures from saved audited values; no fits.')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.output.resolve()
    if folder.exists():raise FileExistsError('Use a fresh output directory')
    rows,non=read_inputs()
    configure_style()
    figures=[]
    for method in ['STSP','DR','YY']:
        for metric in ['average','infinity']:
            if method=='STSP' and metric=='infinity':continue
            figures.append(six_panel(rows,method,metric,folder))
    figures.append(circle(folder))
    (folder/'MANIFEST.json').write_text(json.dumps(dict(figures=figures,inputs=SOURCES,new_fits=False,new_integrations=False),indent=2),encoding='utf8')
    print(json.dumps(dict(figures=len(figures),new_fits=False)))
if __name__=='__main__':main()
