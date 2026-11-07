"""Sample the exact six E1 surface densities; fitters receive unweighted data."""
import hashlib
import numpy as np

BATCH=65536


def coordinates_sha(a):
    return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()


def factor(key,points):
    x=np.asarray(points)
    if key=='clifford_torus_r4':
        phase=np.sqrt(2)*x[:,0]
    elif key=='corrugated_torus_r3':
        radial=np.hypot(x[:,0],x[:,1])-2.
        phase=radial/np.hypot(radial,x[:,2])
    else:
        phase=x[:,0]
    value=1+.5*phase
    if np.any(value<.5-1e-12) or np.any(value>1.5+1e-12):raise ArithmeticError('Invalid clean geometry in density sampler')
    return value


def generate(geometry,method,repeat,n,sigma=.005):
    if method not in ('STSP','DR','YY') or repeat not in range(20):raise ValueError('Invalid parent')
    common=[20260911,2 if method=='STSP' else 202,geometry.id,repeat]
    proposal=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(common+[0])))
    noise_rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(common+[1])))
    accept=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(common+[2])))
    blocks=[];count=0;proposed=0;accepted=0
    while count<n:
        candidates=geometry.sample_clean(proposal,BATCH)
        keep=accept.random(BATCH)<factor(geometry.key,candidates)/1.5
        values=candidates[keep];proposed+=BATCH;accepted+=len(values)
        take=values[:n-count];blocks.append(take);count+=len(take)
    clean=np.concatenate(blocks,axis=0)
    noise=noise_rng.standard_normal((n,geometry.D));observed=clean+float(sigma)*noise
    lineage=dict(algorithm='PCG64DXSM',proposal_seed=common+[0],noise_seed=common+[1],acceptance_seed=common+[2],
                 density='same six E1 surface densities; amplitude0.5, envelope1.5',
                 rejection_rule='area-uniform original geometry proposals, accept(1+0.5*phase)/1.5 BEFORE Gaussian noise',
                 proposal_batch_size=BATCH,proposed=proposed,accepted_including_unused_tail=accepted,
                 clean_coordinates_sha256=coordinates_sha(clean),noise_coordinates_sha256=coordinates_sha(noise),
                 observed_coordinates_sha256=coordinates_sha(observed),
                 prefix_coordinates_sha256={str(m):coordinates_sha(observed[:m]) for m in (2000,5000,10000,20000,50000,100000,200000,500000) if m<=n})
    return clean,noise,observed,lineage
