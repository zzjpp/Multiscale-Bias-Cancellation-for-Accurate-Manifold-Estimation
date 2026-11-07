"""E1 Gaussian-reduced population STSP on the E4 curve, analytic derivatives.

The residual and moments use the unchanged E1 radial integral engine. The
derivative follows Gaussian moment identities m_j'=(m_j-m_{j+1})d/sigma^2.
No Monte Carlo or finite observation cloud is used.
"""
from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from radial_cpu_fast import radial_moments4

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen/E4/fitting/original/src'))
from e4.geometry import gamma,arc_table
from e4.operator import STSPState,normal_projector_derivative

PKG=ROOT/'frozen/population_engine/src/e1_theory_path'
spec=importlib.util.spec_from_file_location('_p1_E1_population',PKG/'__init__.py',submodule_search_locations=[str(PKG)])
package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
native=importlib.import_module('_p1_E1_population.population')

@lru_cache(maxsize=8)
def latent_rule(count):
    table=arc_table();theta=table.invert(np.arange(int(count))/int(count))
    points=gamma(theta);weights=np.full(int(count),1/int(count))
    points.flags.writeable=False;weights.flags.writeable=False
    return points,weights

class CurveGeometry:
    d=1;D=3;q=2
    def latent_rule(self,count):return latent_rule(count)

class PopulationSTSP:
    def __init__(self,sigma,h,latent_count=16384,radial_order=160):
        self.sigma=float(sigma);self.h=float(h);self.radial_order=int(radial_order)
        self.latent,self.weights=latent_rule(latent_count)
        self.geometry=CurveGeometry()
        self.evaluation_count=0
        self.computed_evaluation_count=0
        self._cache={}

    def evaluate(self,center,derivatives=True):
        self.evaluation_count+=1;y=np.asarray(center,dtype=float)
        key=y.tobytes()
        if key in self._cache:return self._cache[key]
        self.computed_evaluation_count+=1
        delta=self.latent-y
        radius=np.linalg.norm(delta,axis=1)/self.sigma
        active=radius<=self.h/self.sigma+12
        d=delta[active];w=self.weights[active];radius=radius[active]
        moments=radial_moments4(radius,self.h/self.sigma,self.radial_order)
        m0,m1,m2,m3=moments
        mass=float(w@m0)
        if mass<=0:raise ArithmeticError('Nonpositive population mass')
        offset=np.einsum('n,n,ni->i',w,m1,d)/mass
        raw=self.sigma**2*np.eye(3)*float(w@m1)+np.einsum('n,n,ni,nj->ij',w,m2,d,d)
        covariance=raw/mass-np.outer(offset,offset);covariance=(covariance+covariance.T)/2
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        Q=eigenvectors[:,:2]@eigenvectors[:,:2].T
        gap=float(eigenvalues[2]-eigenvalues[1])/self.h**2
        state=STSPState(y.copy(),y+offset,covariance,eigenvalues,eigenvectors,Q,-Q@offset,
            mass/self.h,gap,int(active.sum()),float(mass**2/np.sum((w*m0)**2)),
            np.empty((0,3)),np.empty(0),np.empty(0),
            dict(delta=d,weights=w,moments=moments,mass=mass,offset=offset,raw_second=raw))
        if len(self._cache)>=8:self._cache.pop(next(iter(self._cache)))
        self._cache[key]=state
        return state

    def directional(self,state,direction):
        u=np.asarray(direction,dtype=float)
        extra=state.extra;d=extra['delta'];w=extra['weights'];m0,m1,m2,m3=extra['moments']
        mass=extra['mass'];offset=extra['offset'];dot=d@u;sig2=self.sigma**2
        dmass=float(w@((m0-m1)*dot/sig2))
        dA=np.einsum('n,n,ni->i',w,(m1-m2)*dot/sig2,d)-u*float(w@m1)
        doffset=(dA-offset*dmass)/mass
        dC=np.eye(3)*float(w@((m1-m2)*dot))
        dC+=np.einsum('n,n,ni,nj->ij',w,(m2-m3)*dot/sig2,d,d)
        v=np.einsum('n,n,ni->i',w,m2,d)
        dC-=np.outer(u,v)+np.outer(v,u)
        dcov=dC/mass-extra['raw_second']*dmass/mass**2
        dcov-=np.outer(doffset,offset)+np.outer(offset,doffset)
        dcov=(dcov+dcov.T)/2
        dQ=normal_projector_derivative(state.eigenvalues,state.eigenvectors,2,dcov)
        dR=-dQ@offset-state.Q@doffset
        return dict(DR=dR,DQ=dQ,DSigma=dcov,Dmu=u+doffset)

    def fiber_jacobian(self,state,basis):
        return np.column_stack([basis.T@self.directional(state,basis[:,j])['DR'] for j in range(2)])
