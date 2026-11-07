import json
from pathlib import Path
import unittest
import numpy as np
import native
from sampling import generate,factor


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions,_=native.load()
        _,_,cls.lookup,_=cls.functions['select']('STSP','circle_r2')

    def test_original_snapshot_unchanged(self):
        from common import ROOT,sha
        r=json.loads((ROOT/'SOURCE_BINDING.json').read_bytes())
        for name,meta in r['protected_original_members'].items():
            self.assertEqual(sha(ROOT/'frozen/E2'/name),meta['sha256'])

    def test_density_samples_and_moments(self):
        for key in native.KEYS:
            if key==native.KEYS[-1]:
                from corrugated_geometry import CorrugatedGeometry,canonical
                g=CorrugatedGeometry()
                theta=np.arange(65536)*2*np.pi/65536
                base=canonical.corrugated_area_jacobian(theta)
                expected=np.sum(np.cos(theta)*(1+.5*np.cos(theta))*base)/np.sum((1+.5*np.cos(theta))*base)
            else:
                g=self.lookup[key]
                expected=.25 if key in ('circle_r2','clifford_torus_r4') else .5/(g.d+1)
            clean,noise,observed,lineage=generate(g,'STSP',0,100000)
            phase=(factor(key,clean)-1)/.5
            self.assertLess(abs(float(phase.mean())-expected),.012)
            self.assertTrue(np.isfinite(observed).all())
            np.testing.assert_array_equal(observed,clean+.005*noise)
            self.assertLess(abs(float(noise.mean())),.012)
            self.assertLess(abs(float(noise.std())-1),.012)
            self.assertEqual(lineage['proposal_seed'][-1],0)
            self.assertEqual(lineage['noise_seed'][-1],1)
            self.assertEqual(lineage['acceptance_seed'][-1],2)

    def test_nested_and_deterministic_sampling(self):
        g=self.lookup['sphere2_r10']
        a=generate(g,'DR',3,2000);b=generate(g,'DR',3,5000);c=generate(g,'DR',3,2000)
        for i in range(3):
            np.testing.assert_array_equal(a[i],b[i][:2000])
            np.testing.assert_array_equal(a[i],c[i])

    def test_query_grid_is_original(self):
        for key,g in self.lookup.items():
            q=g.query_grid(144)
            self.assertEqual(q.shape,(144,g.D))
            self.assertTrue(np.isfinite(q).all())
            self.assertEqual(g.query_grid(144).tobytes(),q.tobytes())

    def test_native_actual_cpu_pipeline(self):
        g=self.lookup['circle_r2'];data=generate(g,'STSP',1,4096)[2]
        cfg,lib,_,_=self.functions['select']('STSP','circle_r2')
        q=g.query_grid(144)[:2]
        raw=self.functions['cell']('STSP',data,g,q,cfg,lib,False)
        groups=self.functions['normalize']('STSP',raw)
        self.assertEqual(set(groups),{'Single_h1','Single_h2','Multi'})
        metric=native.metric_function()
        for name,rows in groups.items():
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='success' for r in rows))
            self.assertTrue(np.isfinite(metric('STSP',name,raw)['E_infty']))


if __name__=='__main__':unittest.main()
