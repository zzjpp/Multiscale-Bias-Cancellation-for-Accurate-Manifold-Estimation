"""Deterministic oscillatory normal graph: nonunique CP, locally unique ENF."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'counterexample'
N=int(Fraction(1,5)**-6)
A=Fraction(1,25)/N

def r_float(t):return 1.04+float(A)*np.cos(N*np.asarray(t))
def g_float(t,phi=0.):
    r=r_float(t)
    return (r-1.01)**2+4*1.01*r*np.sin((np.asarray(t)-phi)/2)**2
def dg_float(t,phi=0.):
    r=r_float(t);rp=-float(A)*N*np.sin(N*np.asarray(t))
    return 2*rp*(r-1.01*np.cos(np.asarray(t)-phi))+2*1.01*r*np.sin(np.asarray(t)-phi)

def parts(t):
    r=mp.mpf('1.04')+mp.mpf(A.numerator)/A.denominator*mp.cos(N*t)
    rp=-mp.mpf('0.04')*mp.sin(N*t)
    rpp=-mp.mpf('0.04')*N*mp.cos(N*t)
    return r,rp,rpp
def g(t,phi):
    r,_,_=parts(t)
    return (r-mp.mpf('1.01'))**2+4*mp.mpf('1.01')*r*mp.sin((t-phi)/2)**2
def dg(t,phi):
    r,rp,_=parts(t);rho=mp.mpf('1.01');d=t-phi
    return 2*rp*(r-rho*mp.cos(d))+2*rho*r*mp.sin(d)
def ddg(t,phi):
    r,rp,rpp=parts(t);rho=mp.mpf('1.01');d=t-phi
    return 2*rp**2+2*rpp*(r-rho*mp.cos(d))+4*rho*rp*mp.sin(d)+2*rho*r*mp.cos(d)

def stationary_points(phi):
    phi=mp.mpf(str(phi));rho=mp.mpf('1.01');rmin=mp.mpf('1.04')-mp.mpf(A.numerator)/A.denominator
    # g(phi) is an explicit feasible upper bound. For |theta-phi| greater
    # than this angle, the radial and angular lower bound exceeds it.
    bound=mp.acos(1-(g(phi,phi)-(rmin-rho)**2)/(2*rho*rmin))
    grid=np.linspace(float(phi-bound),float(phi+bound),4097)
    vals=dg_float(grid,float(phi));brackets=[]
    for l,u,fl,fu in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
        if fl==0:brackets.append((l,l))
        elif fl*fu<0:brackets.append((l,u))
    roots=[]
    for l,u in brackets:
        if l==u:root=mp.mpf(str(l))
        else:
            root=mp.findroot(lambda t:dg(t,phi),(mp.mpf(str(l)),mp.mpf(str(u))),
                             tol=mp.mpf('1e-65'),maxsteps=100,solver='anderson')
        if abs(dg(root,phi))>mp.mpf('1e-55'):raise ArithmeticError('Unresolved stationary point')
        if not any(abs(root-x)<mp.mpf('1e-40') for x in roots):roots.append(root)
    roots.sort()
    minima=[t for t in roots if ddg(t,phi)>0]
    if not minima:raise ArithmeticError('No CP minimum found')
    best=min(g(t,phi) for t in minima)
    # At phi=0 both exactly symmetric candidates are retained.
    winners=[t for t in minima if abs(g(t,phi)-best)<mp.mpf('1e-55')]
    return phi,bound,roots,minima,winners,best

def write_csv(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=False)
    with mp.workdps(80):
        phi,bound,roots,minima,winners,best=stationary_points('0')
        assert N==15625 and len(winners)==2
        assert g(0,0)>best and g(mp.pi,0)>g(0,0) and ddg(0,0)<0
        assert abs(winners[0]+winners[1])<mp.mpf('1e-60')
        records=[dict(theta=str(t),g=str(g(t,0)),g_second_derivative=str(ddg(t,0)),
                      type='minimum' if ddg(t,0)>0 else 'maximum') for t in roots]
        perturbations=sorted(set([0.]+[s*x for x in [1e-12,1e-10,1e-8,1e-7,1e-6,5e-6,1e-5] for s in [-1,1]]))
        branch=[]
        for value in perturbations:
            pp,bb,rr,mm,ww,gg=stationary_points(value)
            for t in ww:
                r,_,_=parts(t)
                e,_,_=parts(pp)
                branch.append(dict(reference_angle=float(pp),cp_theta=float(t),enf_theta=float(pp),
                    cp_x=float(r*mp.cos(t)),cp_y=float(r*mp.sin(t)),
                    enf_x=float(e*mp.cos(pp)),enf_y=float(e*mp.sin(pp)),
                    cp_g=float(gg),enf_g=float(g(pp,pp)),minimizer_count=len(ww),
                    global_exclusion_halfwidth=float(bb)))
        for row in branch:
            if row['reference_angle']!=0:assert row['cp_theta']*row['reference_angle']>0
        r0,rp0,rpp0=parts(0)
        curvature0=abs(r0*r0+2*rp0*rp0-r0*rpp0)/(r0*r0+rp0*rp0)**mp.mpf('1.5')
        jump=2*parts(winners[1])[0]*mp.sin(winners[1])
        result=dict(status='PASS',arithmetic_digits=80,N=N,a=str(mp.mpf(A.numerator)/A.denominator),
            reference_point=['1.01','0'],ENF_point=[str(r0),'0'],
            CP_angles=[str(x) for x in winners],CP_g=str(best),ENF_g=str(g(0,0)),
            objective_gap=str(g(0,0)-best),g_second_derivative_at_zero=str(ddg(0,0)),
            target_curvature_at_zero=str(curvature0),global_exclusion_halfwidth=str(bound),
            limiting_CP_point_jump=str(jump),stationary_points=records,
            nonuniqueness_argument='g is even, and explicit g(theta*)<g(0)<g(pi); compactness implies a nonzero non-pi global minimum, hence at least two symmetric global minima. The angular lower bound excludes all distant arcs; a 4096-interval derivative census and 80-digit roots locate the pair numerically.',
            ENF_scope='Unique intersection on the local outward radial fiber; the far side of the full normal line is excluded.',
            not_STSP_generated=True,not_a_counterexample_to_fourth_order_position_error=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_csv(OUT/'STATIONARY_POINTS.csv',records);write_csv(OUT/'BRANCH_SWITCH.csv',branch)
    grid=np.linspace(-5e-4,5e-4,4001)
    write_csv(OUT/'OBJECTIVE_VALUES.csv',[dict(theta=float(t),g=float(v)) for t,v in zip(grid,g_float(grid))])
    (OUT/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    render(result,branch,grid)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def render(result,branch,grid):
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
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    for sign in [-1,1]:
        part=[b for b in branch if b['reference_angle']*sign>0]
        ax.plot([b['reference_angle']*1e6 for b in part],[b['cp_theta']*1e6 for b in part],c='#8E44AD',marker='D',ms=4,label='CP branch' if sign==1 else None)
    ax.scatter([0,0],cp*1e6,facecolors='none',edgecolors='#8E44AD',marker='D',s=60,zorder=5,label='CP at zero (two points)')
    xx=np.linspace(-10,10,101);ax.plot(xx,xx,c='#009E73',label='ENF (continuous)')
    ax.set_xlabel(r'Reference angle $\phi$ ($10^{-6}$ radians)')
    ax.set_ylabel(r'Matched target angle ($10^{-6}$ radians)')
    ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=10);fig.tight_layout()
    save(fig,'P1_CP_branch_switch')
    (OUT/'caption.tex').write_text(r'''\caption{A constructed oscillatory normal graph, not an STSP output. Here $h_1=0.1$, $h_2=0.2$, $N=15625$, and $a=2.56\times10^{-6}$. At $y_1=(1.01,0)$ the local outward radial ENF has a unique intersection, whereas the continuous squared-distance objective has two symmetric global minimizers. The inset magnifies the target oscillation; the main geometry panel uses unequal coordinate scales. Positive and negative reference-angle perturbations select different CP branches, while ENF varies continuously. This tests uniqueness and continuous selection, not failure of a fourth-order location bound.}'''+'\n',encoding='utf8')

if __name__=='__main__':main()
