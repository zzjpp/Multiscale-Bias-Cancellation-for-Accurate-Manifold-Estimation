"""Render the controlled Figure 2 extension and supplementary tables from actual values."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
import numpy as np

ROOT=Path(__file__).resolve().parent
ORDER=['circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3']
TITLES=[r'$S^1\subset\mathbb{R}^2$',r'$S^2\subset\mathbb{R}^3$',
        r'Clifford torus $\subset\mathbb{R}^4$',r'$S^3\subset\mathbb{R}^4$',
        r'$S^2\subset\mathbb{R}^{10}$',r'Corrugated torus $\subset\mathbb{R}^3$']
TEXT=['Circle','S2 in R3','Clifford torus','S3 in R4','S2 in R10','Corrugated torus']
METHODS=['single','naive','proposed','ms_stsp']
LEGEND=['Successive tangent-space projection (STSP)','Squared extrapolation (SQ)',
        'Geometric bias cancellation (GBC)','Multiscale successive tangent-space projection (MS-STSP)']
COLORS=['#94989C','#294F7E','#C54F2D','#BABEC3']
MARKS=['o','^','s','D'];STYLES=['-','--','--','-.']
DISPLAY_INDICES=[0,3,1,2]
LINE_WIDTHS=[1.0,1.25,1.55,.95]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):return list(csv.DictReader(Path(path).open(encoding='utf8')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure-only',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='Replace the figure exports in an existing output directory; requires --figure-only.')
    a=p.parse_args()
    if a.overwrite and not a.figure_only:p.error('--overwrite requires --figure-only')
    dest=a.output.resolve()
    if dest.exists() and not a.overwrite:raise ValueError('Use a fresh output directory or --figure-only --overwrite')
    audit=json.loads((ROOT/'analysis/NUMERICAL_CHECK.json').read_bytes())
    precision=json.loads((ROOT/'analysis/INDEPENDENT_PRECISION.json').read_bytes())
    assert audit['status']==precision['status']=='PASS'
    rows=read_csv(ROOT/'analysis/VALUES.csv');slopes=read_csv(ROOT/'analysis/SLOPES.csv')
    constants=read_csv(ROOT/'analysis/CIRCLE_CONSTANT.csv')
    assert len(rows)==42 and len(slopes)==24 and len(constants)==7
    sigmas=[.008,.01,.0125,.016,.02,.025,.03]
    allvalues=[float(r[m+'_E_infinity']) for r in rows for m in METHODS]
    assert np.isfinite(allvalues).all() and min(allvalues)>0
    dest.mkdir(exist_ok=a.overwrite)
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':14,'axes.titlesize':15.5,'xtick.labelsize':12.5,
                         'ytick.labelsize':13,'legend.fontsize':13,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(14.2,7.7),sharex=True,sharey=True)
    slope_artists=[];series_records=[]
    for panel,(ax,key,title) in enumerate(zip(axes.flat,ORDER,TITLES)):
        block=sorted([r for r in rows if r['geometry']==key],key=lambda r:float(r['sigma']))
        assert [float(r['sigma']) for r in block]==sigmas
        annotations={};series={}
        for i in DISPLAY_INDICES:
            m=METHODS[i]
            yy=np.array([float(r[m+'_E_infinity']) for r in block])
            line,=ax.plot(sigmas,yy,c=COLORS[i],marker=MARKS[i],linestyle=STYLES[i],
                         lw=LINE_WIDTHS[i],ms=4.2,mew=.8,
                         mfc=COLORS[i] if m=='proposed' else 'white',zorder=4 if m=='proposed' else 3)
            assert np.array_equal(line.get_ydata(),yy)
            actual=float(np.polyfit(np.log(sigmas),np.log(yy),1)[0])
            saved=float(next(r for r in slopes if r['geometry']==key and r['method']==m)['slope_all_7'])
            assert abs(actual-saved)<1e-12
            annotations[m]=actual;series[m]=yy.tolist()
        assert len(ax.patches)==0
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.032,.0075);ax.set_ylim(1e-9,.1)
        ax.xaxis.set_major_locator(FixedLocator([.03,.02,.01,.008]));ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
        plt.setp(ax.get_xticklabels(),rotation=0,ha='center')
        ax.set_title(title,pad=10);ax.grid(False)
        ax.tick_params(axis='both',length=3,width=.7,pad=4)
        ax.spines[['top','right']].set_visible(False)
        for side in ['left','bottom']:
            ax.spines[side].set_linewidth(.7);ax.spines[side].set_color('#55595D')
        ax.text(-.115,1.06,f'({chr(97+panel)})',transform=ax.transAxes,
                va='bottom',ha='left',fontsize=13.5,fontweight='bold')
        first=(rf'$p_{{\mathrm{{STSP}}}}={annotations["single"]:.2f},\quad '
               rf'p_{{\mathrm{{MS\!-\!STSP}}}}={annotations["ms_stsp"]:.2f}$')
        second=(rf'$p_{{\mathrm{{sq}}}}={annotations["naive"]:.2f},\quad '
                rf'p_{{\mathrm{{GBC\!-\!STSP}}}}={annotations["proposed"]:.2f}$')
        for height,label in [(.12,first),(.035,second)]:
            slope_artists.append((ax,ax.text(.035,height,label,transform=ax.transAxes,
                                           va='bottom',ha='left',fontsize=12)))
        series_records.append(dict(geometry=key,sigma=sigmas,values=series,slopes=annotations))
    handles=[Line2D([],[],color=COLORS[i],marker=MARKS[i],linestyle=STYLES[i],
                    lw=LINE_WIDTHS[i],ms=4.5,mew=.8,
                    mfc=COLORS[i] if METHODS[i]=='proposed' else 'white',label=LEGEND[i]) for i in DISPLAY_INDICES]
    fig.legend(handles=[handles[i] for i in [0,2,1,3]],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.53,.995),columnspacing=1.6,handletextpad=.65)
    fig.text(.535,.885,r'$h_1=3\sigma,\quad h_2=6\sigma$',ha='center',fontsize=14)
    fig.supxlabel(r'Noise level $\sigma$',y=.025,fontsize=16)
    fig.supylabel(r'Population error $E_\infty$',x=.014,fontsize=16)
    fig.subplots_adjust(left=.085,right=.99,bottom=.12,top=.78,wspace=.22,hspace=.42)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for legend in fig.legends:
        box=legend.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1), 'Legend exceeds canvas'
    for ax,artist in slope_artists:
        bounds=artist.get_window_extent(renderer).expanded(1.02,1.08)
        assert ax.bbox.contains(bounds.x0,bounds.y0) and ax.bbox.contains(bounds.x1,bounds.y1)
        for line in ax.lines:
            assert not line.get_transform().transform_path(line.get_path()).intersects_bbox(bounds,filled=False), 'Slope text touches a data curve'
    stem='e1_stsp_population_infinity_with_ms_stsp'
    fig.savefig(dest/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(dest/(stem+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    caption=r'''\caption{Population maximum errors under a controlled STSP weight comparison. In the legend, Successive tangent-space projection (STSP), Multiscale successive tangent-space projection (MS-STSP), Squared extrapolation (SQ), and Geometric bias cancellation (GBC) denote Single, Original MS-STSP, Naive, and GBC-STSP (the proposed method), respectively. All three controlled estimators use exactly the same archived smooth-kernel STSP fits and estimated-normal-fiber correspondence at $h_1=3\sigma$ and $h_2=6\sigma$. Single is the matched coarse-scale point $Z_{h_2}$; Naive uses $(4/3,-1/3)$; Proposed uses the exact noise-dependent STSP bias coefficients. Original MS-STSP is the previously computed population adaptation with a hard-ball kernel, its own fixed-reference iteration and radii $\{3,4.5,6\}\sigma$; its measurements are reused without refitting. Axes are logarithmic, with noise level decreasing from left to right. The four unboxed slopes $p_{\mathrm{STSP}}$, $p_{\mathrm{MS\!-\!STSP}}$, $p_{\mathrm{sq}}$, and $p_{\mathrm{GBC\!-\!STSP}}$ are fitted over all seven displayed noise levels and correspond to the four legend entries, respectively. Errors use the original E1 queries: one canonical query on each homogeneous example and 128 fixed meridional queries on the corrugated torus. The plotted maximum is over these finite query sets, not a certified continuous supremum.}'''
    (dest/'figure_caption.tex').write_text(caption+'\n',encoding='utf8')
    if not a.figure_only:
        table_rows=[]
        for key,title in zip(ORDER,TEXT):
            table_rows.append([title]+[f"{float(next(r for r in slopes if r['geometry']==key and r['method']==m)['slope_all_7']):.5f}" for m in METHODS])
        tab=r'''% Requires booktabs. Two panels of one supplementary table.
\begin{table}[t]
\centering\small
\caption{Controlled STSP weight comparison. Panel A reports full-grid log--log slopes of $E_\infty$ against $\sigma$. Panel B checks the unit-circle Naive constant and the normalized Proposed error. All values are actual measurements from shared population fits; finite-range slopes are not asymptotic proofs.}
\label{tab:e1-weight-control}
\begin{tabular}{lrrrr}
\toprule
Geometry & Single & Naive $h^2$ & Proposed & MS-STSP\\
\midrule
'''
        pretty=[r'$S^1\subset\mathbb{R}^2$',r'$S^2\subset\mathbb{R}^3$',r'Clifford torus',r'$S^3\subset\mathbb{R}^4$',r'$S^2\subset\mathbb{R}^{10}$',r'Corrugated torus']
        for row,label in zip(table_rows,pretty):tab+=label+' & '+' & '.join(row[1:])+r'\\'+'\n'
        tab+=r'''\bottomrule
\end{tabular}
\par\medskip
\begin{tabular}{rrrrr}
\toprule
$\sigma$ & $E_{\infty,\mathrm{naive}}/\sigma^2$ & $\Gamma_{1,2}$ & Relative difference (\%) & $E_{\infty,\mathrm{prop}}/\sigma^4$\\
\midrule
'''
        for r in constants:
            tab+=' & '.join([f"{float(r['sigma']):g}",f"{float(r['naive_over_sigma2']):.9f}",f"{float(r['Gamma_1_2']):.9f}",f"{100*float(r['relative_difference_from_Gamma']):.6f}",f"{float(r['proposed_over_sigma4']):.6f}"])+r'\\'+'\n'
        tab+=r'''\bottomrule
\end{tabular}
\par\smallskip\raggedright\footnotesize
The corrugated-torus Proposed slope is 3.81725 over all seven levels and 3.91030 over the four smallest levels. The latter is a sensitivity diagnostic, not a replacement for the full-grid result. The Naive unit-circle error approaches the nonzero theoretical constant $\Gamma_{1,2}=0.476376834958\ldots$ as noise decreases.
\end{table}
'''
        (dest/'supplementary_table.tex').write_text(tab,encoding='utf8')
        paragraph=r'''\paragraph{Controlled comparison of extrapolation weights.}
To isolate the role of the bias coefficient, we reuse exactly the same smooth-kernel STSP population fits and estimated-normal-fiber correspondence at $h_1=3\sigma$ and $h_2=6\sigma$. We compare the matched coarse-scale point $Z_{h_2}$, the Naive combination $(4Z_{h_1}-Z_{h_2})/3$, and the proposed combination with weights determined by $c_{\mathrm{STSP}}(h,\sigma)$. Thus, neither the fitted points nor their correspondence is changed between the controlled estimators. Along this joint noise--bandwidth path, $c_{\mathrm{STSP}}(h_k,\sigma)=\beta_k\sigma^2$; the Naive weights do not cancel the resulting leading bias. Over all seven noise levels, the Naive slopes range from $1.99959$ to $1.99996$. The proposed slopes are $4.0041$--$4.0078$ on the five homogeneous examples and $3.8172$ on the corrugated torus; on the latter, the four-smallest-level diagnostic gives $3.9103$. These finite-range measurements are consistent with the predicted bias cancellation and do not require a nonzero fourth-order leading term on every manifold. On the unit circle, $E_{\infty,\mathrm{naive}}/\sigma^2=0.476369353$ at $\sigma=0.008$, within $0.00157\%$ of the theoretical constant $\Gamma_{1,2}=0.476376835$, while $E_{\infty,\mathrm{prop}}/\sigma^4$ varies only from $1.7471$ to $1.7571$ over the displayed range. The original MS-STSP curve is retained as a separate algorithmic reference: its hard-ball kernel and matching iteration differ, so it is not the Naive control and its comparison does not isolate the weights alone.
'''
        (dest/'section_5_1_1_discussion.tex').write_text(paragraph,encoding='utf8')
        make_table_pdf(dest,table_rows,constants)
        write_report(dest,rows,slopes,constants,audit,precision)
    manifest=dict(input_csv_sha256=sha(ROOT/'analysis/VALUES.csv'),renderer_sha256=sha(__file__),
        method_order=METHODS,display_method_order=[METHODS[i] for i in DISPLAY_INDICES],
        legend_labels=[LEGEND[i] for i in DISPLAY_INDICES],
        log_x=True,log_y=True,x_direction='descending',bars=False,slope_boxes=False,
        main_title=False,subtitle=True,query_sets_retained=True,
        four_slopes_per_panel=True,slope_text_curve_collisions=False,series=series_records,
        exports={f.name:sha(f) for f in dest.iterdir() if f.suffix in ['.png','.pdf','.tex']})
    (dest/'PLOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Actual controlled figure and requested accompanying material generated:',dest)

def make_table_pdf(dest,table_rows,constants):
    import textwrap
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.5,.955,'Controlled STSP weight comparison',ha='center',fontsize=17)
    fig.text(.08,.92,r'Shared population fits and ENF; $h_1=3\sigma$, $h_2=6\sigma$.',fontsize=11)
    def table(bounds,headers,data,widths):
        ax=fig.add_axes(bounds);ax.axis('off')
        tab=ax.table(cellText=data,colLabels=headers,cellLoc='right',colWidths=widths,bbox=[0,0,1,1])
        tab.auto_set_font_size(False);tab.set_fontsize(9)
        for (row,col),cell in tab.get_celld().items():
            cell.visible_edges='';cell.set_linewidth(.6)
            if row==0:cell.visible_edges='TB';cell.set_text_props(weight='bold')
            if row==len(data):cell.visible_edges='B'
            if col==0:cell.set_text_props(ha='left')
    fig.text(.08,.868,'A. Full-grid log-log slopes of maximum error',fontsize=12)
    table([.08,.62,.84,.23],['Geometry','Single',r'Naive $h^2$','Proposed','MS-STSP'],table_rows,[.32,.17,.17,.17,.17])
    fig.text(.08,.572,'B. Unit-circle normalized errors and theoretical constant',fontsize=12)
    cells=[]
    for r in constants:
        cells.append([f"{float(r['sigma']):g}",f"{float(r['naive_over_sigma2']):.9f}",f"{float(r['Gamma_1_2']):.9f}",f"{100*float(r['relative_difference_from_Gamma']):.6f}",f"{float(r['proposed_over_sigma4']):.6f}"])
    table([.08,.29,.84,.26],[r'$\sigma$',r'Naive$/\sigma^2$',r'$\Gamma_{1,2}$','Rel. diff. (%)',r'Proposed$/\sigma^4$'],cells,[.10,.24,.23,.20,.23])
    y=.25
    for text in [
        'All slopes use seven levels: 0.008, 0.01, 0.0125, 0.016, 0.02, 0.025 and 0.03. Finite-range slopes are measurements, not asymptotic proofs.',
        'The proposed corrugated-torus slope is 3.81725 on the full grid and 3.91030 on the four smallest levels. The latter is only a sensitivity diagnostic.',
        'Gamma(1,2) = 0.476376834958256965... is computed independently at 70-digit precision. Relative difference compares the measured normalized Naive error with this constant.',
        'MS-STSP is a separate hard-ball population adaptation with its own iteration. Only Single, Naive and Proposed share the exact same two fitted points and correspondence.',
        'Maximum errors use the frozen E1 query sets; the corrugated torus uses 128 fixed queries. No finite-sample Monte Carlo is used.']:
        wrapped=textwrap.fill(text,width=106)
        fig.text(.08,y,wrapped,fontsize=8.8,va='top')
        y-=.016*len(wrapped.splitlines())+.012
    fig.savefig(dest/'supplementary_table.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def write_report(dest,rows,slopes,constants,audit,precision):
    lines=['# Shared-fit STSP weight experiment','',
        'All 42 geometry/noise settings are complete. Main evaluation reuses 84 existing production/refined STSP packets. '
        'No main fits or MS-STSP integration were rerun. One extra unit-circle CPU expectation solve at sigma=0.008 used '
        '32768 latent nodes, radial order320 and stricter solver tolerances.','',
        '## Controlled quantities','',
        'Single, Naive and Proposed use the exact same Z_h1 and ENF-matched Z_h2 at each query. '
        'Naive uses (4/3,-1/3); Proposed uses the original archived weights after an independent one-dimensional coefficient check. '
        'Affine combinations are evaluated in the stable form Z_h1 + w2 B t2. '
        'The previous independently fitted Single baseline is retained in VALUES.csv as legacy_single; it is not silently used as the controlled Single.', '',
        '## Actual full-grid slopes','',
        '| Geometry | Single | Naive h2 | Proposed | Original MS-STSP |',
        '|---|---:|---:|---:|---:|']
    for key,title in zip(ORDER,TEXT):
        vv=[float(next(x for x in slopes if x['geometry']==key and x['method']==m)['slope_all_7']) for m in METHODS]
        lines.append('| '+title+' | '+' | '.join(f'{v:.6f}' for v in vv)+' |')
    lines+=['','The proposed full-grid corrugated slope is retained as 3.81725, not forced to4. '
        'The four-smallest-level diagnostic is3.91030. The other five proposed slopes are4.0041--4.0078. '
        'All six Naive slopes remain approximately2. These are finite-range data, not a proof of an asymptotic bound.','',
        '## Circle leading constant','',
        'Independent70-digit integral gives Gamma_1,2 = '+audit['high_precision_circle']['Gamma_1_2']+'. '
        'At sigma=.008 the observed Naive/sigma2 is0.4763693528826024 (relative difference0.00157062%). '
        'At sigma=.03 it is0.47626913326468157; the difference decreases as sigma decreases. '
        'Proposed/sigma4 stays between1.7470724303 and1.7571266003 on this grid.','',
        '## Numerical checks','',
        f"The actual production/refined maximum point difference is {audit['maximum_resolution_point_difference']:.12g}; "
        f"the largest metric difference is {audit['maximum_metric_difference']:.12g}. Both are below the frozen budget "
        f"{audit['point_metric_difference_budget']:.12g} (0.001 times the smallest Proposed error, including average). "
        f"The largest saved final root-correction diagnostic is {audit['maximum_root_correction']:.12g}, below "
        f"{audit['root_correction_budget']:.12g}. These are diagnostics, not certified error bounds.", '',
        f"Independent60-digit geometric distance checks: {precision['independent_distance_checks']} selected cases, "
        f"maximum difference {precision['maximum_independent_distance_difference']:.12g}. "
        'For the five homogeneous examples this checks all original queries; for the corrugated surface it checks '
        'the maximum-error query of every controlled method/noise/resolution, not every query. '
        'The extra strict-circle solve matches the archived shared output points to the recorded double precision.','',
        '## Interpretation and limits','',
        'Along h=a sigma, the exact STSP bias is beta(a,d,D) sigma2. Naive h2 weights do not cancel this coefficient; '
        'the independent unit-circle coefficient is nonzero. Proposed cancels it. Original MS-STSP uses a different '
        'hard-ball kernel and fixed-reference iteration, so its comparison also involves operator and correspondence differences. '
        'It is not the Naive control and is reused as a separate population-adapted algorithm.', '',
        'The first five homogeneous geometries each use one original canonical query; corrugated maximum uses128 fixed '
        'equal-area meridional queries. Thus E_infinity is a maximum over preserved queries, not a certified continuous supremum. '
        'All seven sigma levels and all saved successes are retained. No density, radius, solver or failure-rule tuning '
        'was used to force the predicted slopes. Original publication directories and organized code are not overwritten.','',
        '## Files and replay','',
        '- Figure PNG/PDF and figure_caption.tex in this directory.',
        '- supplementary_table.tex/.pdf: one table with slope and constant panels.',
        '- section_5_1_1_discussion.tex: manuscript-ready discussion.',
        '- ../analysis/VALUES.csv, SLOPES.csv, WEIGHTS.csv, CIRCLE_CONSTANT.csv and PRECISION.csv.',
        '- ../analysis/SHARED_POINTS.json.gz: all shared coordinates and outputs, both resolutions.',
        '- ../SOURCE_BINDING.json and ../PLAN.json: source hashes and frozen design.', '',
        'From the study directory, run `python -X utf8 -B render_and_report.py --output fresh_plot_directory --figure-only` '
        'with a nonexistent output directory to replay the figure. No fits or integrations occur.']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
