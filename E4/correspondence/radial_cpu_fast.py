"""Same E1 quadrature, explicit half-integer Bessel formula for D=3,5,7,9.

Only evaluation of exp(-x) I_nu(x) is changed. For x<2 use scipy.ive;
for x>=2 use exact closed forms. Nodes, weights, precision and tail stay fixed.
"""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre,gammaln,ive

@lru_cache(maxsize=8)
def rule(order):return roots_legendre(int(order))

def scaled_bessel(nu,x):
    answer=np.empty_like(x);small=x<2
    answer[small]=ive(nu,x[small])
    y=x[~small];z=1/y;e=np.exp(-2*y);pref=1/np.sqrt(2*np.pi*y)
    if nu==.5:v=-np.expm1(-2*y)
    elif nu==1.5:v=(1-z)+(1+z)*e
    elif nu==2.5:v=(1-3*z+3*z*z)-(1+3*z+3*z*z)*e
    elif nu==3.5:v=(1-6*z+15*z*z-15*z**3)+(1+6*z+15*z*z+15*z**3)*e
    else:raise ValueError('Only the four required odd dimensions are supported')
    answer[~small]=pref*v
    return answer

def radial_mass(df,noncentral_radius,kappa,order,power=6,kernel_kind='polynomial'):
    a=np.asarray(noncentral_radius,dtype=float).reshape(-1)
    nodes,weights=rule(order);r=float(kappa)*(nodes+1)/2
    if kernel_kind!='polynomial':raise ValueError('The declared population experiment uses polynomial K6')
    kw=weights*float(kappa)/2*np.maximum(1-(r/kappa)**2,0)**int(power)
    answer=np.empty(len(a));nu=df/2-1
    for start in range(0,len(a),512):
        b=a[start:start+512];density=np.empty((len(b),len(r)));central=b==0
        if central.any():density[central]=np.exp((df-1)*np.log(r)-r*r/2-(df/2-1)*np.log(2)-gammaln(df/2))
        if (~central).any():
            bb=b[~central,None]
            with np.errstate(divide='ignore',under='ignore',invalid='ignore'):
                density[~central]=np.exp(np.log(r)+nu*(np.log(r)-np.log(bb))-.5*(r-bb)**2+np.log(scaled_bessel(nu,r*bb)))
        answer[start:start+len(b)]=density@kw
    if not np.isfinite(answer).all():raise ArithmeticError('Nonfinite half-integer radial integral')
    return answer

def radial_moments4(noncentral_radius,kappa,order):
    """The four original odd-dimension moments, sharing their common factors."""
    a=np.asarray(noncentral_radius,dtype=float).reshape(-1)
    nodes,weights=rule(order);r=float(kappa)*(nodes+1)/2
    kw=weights*float(kappa)/2*np.maximum(1-(r/kappa)**2,0)**6
    answer=np.empty((4,len(a)))
    for start in range(0,len(a),512):
        b=a[start:start+512];density=np.empty((4,len(b),len(r)));central=b==0
        if central.any():
            for j,df in enumerate([3,5,7,9]):
                density[j,central]=np.exp((df-1)*np.log(r)-r*r/2-(df/2-1)*np.log(2)-gammaln(df/2))
        if (~central).any():
            bb=b[~central,None];x=bb*r;ratio=np.broadcast_to(r/bb,x.shape)
            large=x>=2;small=~large
            z=1/x[large];e=np.exp(-2*x[large]);rr=ratio[large]
            base=np.exp(-.5*(np.broadcast_to(r,x.shape)[large]-np.broadcast_to(bb,x.shape)[large])**2)*rr/np.sqrt(2*np.pi)
            factors=[-np.expm1(-2*x[large]),(1-z)+(1+z)*e,
                (1-3*z+3*z*z)-(1+3*z+3*z*z)*e,
                (1-6*z+15*z*z-15*z**3)+(1+6*z+15*z*z+15*z**3)*e]
            logbase=np.broadcast_to(np.log(r),x.shape)[small]-.5*(np.broadcast_to(r,x.shape)[small]-np.broadcast_to(bb,x.shape)[small])**2
            for j in range(4):
                local=np.empty_like(x);local[large]=base*rr**j*factors[j]
                local[small]=np.exp(logbase+(.5+j)*np.log(ratio[small])+np.log(ive(.5+j,x[small])))
                density[j,~central]=local
        for j in range(4):answer[j,start:start+len(b)]=density[j]@kw
    if not np.isfinite(answer).all():raise ArithmeticError('Nonfinite batched radial moments')
    return answer
