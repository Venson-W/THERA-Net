"""Input invariants for the segmented-fault experiment."""
import unittest
import numpy as np
from barrier_cases import fields

class Inputs(unittest.TestCase):
    def test_grid_independence_and_localization(self):
        coarse=fields(180,2,.014); fine=fields(360,2,.014)
        for i in range(2,6): np.testing.assert_allclose(coarse[i],fine[i][::2,::2],rtol=0,atol=0)
        x,y,ref,theta,stress,nuc=coarse
        self.assertTrue(np.all((ref[...,0]>=.008)&(ref[...,0]<=.016)))
        self.assertTrue(np.all(theta>0))
        self.assertTrue(np.all(nuc[x>0]==0))
        self.assertTrue(np.all(ref[len(x)//2,:,0]>=.014-1e-14))
        self.assertGreater(ref[len(x)//2,len(y)//2,0],ref[len(x)//2,len(y)//2,1])
        self.assertLess(ref[round((8000+36000)/400),len(y)//2,0],.012)

    def test_control_has_no_internal_band(self):
        x,y,r,*_=fields(180,0,.018)
        self.assertEqual(r[len(x)//2,len(y)//2,0],.008)

if __name__=='__main__': unittest.main()
