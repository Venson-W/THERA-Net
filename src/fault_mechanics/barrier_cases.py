"""Controlled single-event segmented-fault experiments (SI units).

The internal frictional band is a new spatial arrangement of the frozen
material-family closure. No rupture trajectory is used to train this closure.
Width means the flat band width; each edge has a fixed 500 m smooth transition.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from source_cases import smooth_box, spatial_nucleation
from source_family_thera import load_source_model, source_family_parameters, source_truth
from fault_thera import initial_state

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_barrier'
EDGE = 500.

def case_name(width_km, a_barrier):
    return f'w{width_km:g}_a{a_barrier:.3f}'.replace('.', 'p')

def fields(nx, width_km, a_barrier):
    x = np.arange(nx)*72000./nx-36000.
    y = np.arange(nx//2)*72000./nx-18000.
    xx, yy = np.meshgrid(x,y,indexing='ij')
    active = smooth_box(xx,15000.,3000.)*smooth_box(yy,7500.,3000.)
    band = smooth_box(xx,500.*width_km,EDGE) if width_km>0 else np.zeros_like(xx)
    aa = .008 + .008*(1-active) + (a_barrier-.008)*active*band
    ref = source_truth(aa)
    stress = np.full_like(xx,75e6)
    theta = initial_state(stress,1e-12,120e6,ref)
    nuc = spatial_nucleation(np.hypot(xx+9000.,yy))
    return x,y,ref,theta,stress,nuc

def prepare(nx,width_km,a_barrier):
    name=case_name(width_km,a_barrier)
    target=OUT/'inputs'/f'{name}_N{nx}'
    target.mkdir(exist_ok=True,parents=True)
    x,y,ref,theta,stress,nuc=fields(nx,width_km,a_barrier)
    learned=source_family_parameters(load_source_model(),ref[...,0])
    hashes={}
    for mode,params in [('reference',ref),('thera',learned)]:
        array=np.concatenate([params,theta[...,None],stress[...,None],nuc[...,None]],axis=-1)
        assert np.isfinite(array).all() and np.all(array[...,4]>0)
        path=target/(mode+'.bin')
        np.asarray(array,dtype='<f8').tofile(path)
        hashes[mode]=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(name=name,nx=nx,width_km=width_km,a_barrier=a_barrier,
        edge_transition_m=EDGE,a_patch=.008,a_exterior=.016,b=.012,Dc_m=.02,
        normal_pressure_Pa=120e6,prestress_Pa=75e6,nucleation_xy_m=[-9000.,0.],
        initial_state_convention='Identical reference-inverted theta0 and prestress for reference/neural branches; each law equilibrates its initial slip rate.',
        neural_model_sha256=hashlib.sha256((HERE/'results_rupture/source_family/source_model.npz').read_bytes()).hexdigest(),
        input_sha256=hashes)
    (target/'configuration.json').write_text(json.dumps(record,indent=2)+'\n')
    return target

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--nx',type=int,default=180)
    p.add_argument('--widths',type=float,nargs='+',default=[0,.5,1,2,4,6])
    p.add_argument('--a',type=float,nargs='+',default=[.018])
    args=p.parse_args()
    for a in args.a:
        for w in args.widths: print(prepare(args.nx,w,a))
