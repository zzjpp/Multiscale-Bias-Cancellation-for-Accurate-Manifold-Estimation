"""User-authorized E3 design; no empirical tuning or algorithm changes."""
KEYS=('circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3')
N=dict(zip(KEYS,(12000,24000,24000,60000,24000,36000)))
MAIN=dict(zip(KEYS,(256,256,256,256,256,512)))
REV=dict(zip(KEYS,(1024,8192,8192,16384,8192,1024)))
REF={k:65536 if k==KEYS[-1] else 262144 for k in KEYS}
METHODS=('STSP','DR','YY')
GROUPS={'STSP':('Single_h1','Single_h2','Multi'),'DR':('Single_h1','Single_h2','Multi'),
        'YY':('Single_0p4r1','Single_r1','Single_1p6r1','Multi')}
NEW={'STSP':KEYS[1:],'DR':(KEYS[3],KEYS[5]),'YY':(KEYS[3],KEYS[5])}
GRID=(.06,.075,.09,.11,.135,.165,.2,.245,.3,.37,.45,.55,.67,.82,1.,1.2,
      1.4,1.6,1.8,2.,2.25,2.5,2.75,3.,3.25,3.5,4.,5.)

def cases():
    return [(m,p,k) for m in METHODS for k in NEW[m] for p in ('main','rev')]

def tasks():
    reverse=[dict(method=m,phase='rev',geometry=k,repeat=0,shard=s)
             for s in range(8) for k in sorted(KEYS[1:],key=lambda x:-REV[x])
             for m in METHODS if k in NEW[m]]
    main=[dict(method=m,phase='main',geometry=k,repeat=r,shard=0)
          for r in range(20) for k in KEYS[1:] for m in METHODS if k in NEW[m]]
    return reverse+main

def n_for(key):
    return N[key]

def expected_outcomes():
    return sum((MAIN[k]*20 if p=='main' else REV[k])*28*len(GROUPS[m]) for m,p,k in cases())
