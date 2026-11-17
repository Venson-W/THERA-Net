"""Prepare matched full-dynamic reference/THERA-Net inputs; no rupture data used in training.

All resolutions sample the same 100-m master prestress field. No renormalization
is performed after subsampling, so a refinement comparison changes the numerical
grid rather than independently generating a different earthquake.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from source_cases import source_case, spatial_nucleation
from fault_thera import initial_state
from source_family_thera import load_source_model, source_family_parameters

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_rupture'
CASES=[
    dict(name='heterogeneous_D16',dimension=1.6,seed=1101,a0=.007,delta_a0=.010,hypocenter=(-6000.,-4500.)),
    dict(name='uniform',dimension=1.5,seed=1201,a0=.008,delta_a0=.008,hypocenter=(0.,0.)),
    dict(name='unseen_D13',dimension=1.3,seed=1301,a0=.0065,delta_a0=.011,hypocenter=(6000.,0.)),
]

def main():
    model=load_source_model()
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=True,parents=True)
    records=[]
    for case in CASES:
        kw={k:v for k,v in case.items() if k!='name'}
        data=source_case(shape=(720,360),b=.012,**kw)
        if case['name']=='uniform':
            data['initial_stress_Pa'].fill(75e6)
            data['initial_theta_s']=initial_state(data['initial_stress_Pa'],1e-12,120e6,data['parameters_a_b_L_f'])
        record=dict(case)
        record['master_resolution_m']=100
        record['runs']=[]
        for nx in [180,360,720]:
            stride=720//nx
            take=lambda a: a[::stride,::stride].copy()
            true=take(data['parameters_a_b_L_f'])
            learned=source_family_parameters(model,true[...,0])
            theta0=take(data['initial_theta_s'])
            stress=take(data['initial_stress_Pa'])
            nuc=take(data['nucleation_spatial_shape'])
            folder=inputs/f"{case['name']}_N{nx}";folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'fields.npz',x_m=take(data['x_m']),y_m=take(data['y_m']),
                reference_parameters=true,thera_parameters=learned,theta0_s=theta0,
                initial_stress_Pa=stress,nucleation_spatial_shape=nuc)
            for mode,params in [('reference',true),('thera',learned)]:
                fields=np.concatenate([params,theta0[...,None],stress[...,None],nuc[...,None]],axis=-1)
                path=folder/f'{mode}.bin'
                np.asarray(fields,dtype='<f8').tofile(path)
                record['runs'].append(dict(nx=nx,nz=nx//2,mode=mode,file=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        record['stress_master_mean_Pa']=float(data['initial_stress_Pa'].mean())
        record['stress_master_std_Pa']=float(data['initial_stress_Pa'].std())
        record['theta0_range_s']=[float(data['initial_theta_s'].min()),float(data['initial_theta_s'].max())]
        records.append(record)
    config=dict(cases=records,initial_state_convention='Identical reference-inverted theta0 and prestress in both models; initial velocity is equilibrated by each constitutive law.',
        source_figure_relationship='Same physical parameter families as source Figs11-13; newly generated, seeded heterogeneous stresses and explicitly prescribed hypocenters. No claim of identical source realizations or threshold-window restart.',
        start_time_s=0,model_sha256=hashlib.sha256((OUT/'source_family/source_model.npz').read_bytes()).hexdigest())
    (OUT/'input_manifest.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps([dict(name=r['name'],theta=r['theta0_range_s']) for r in records],indent=2))

if __name__=='__main__': main()
