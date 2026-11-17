"""Audit completed runs and export manuscript-sized reproducible plotting data."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from barrier_analysis import load,pair,OUT
from barrier_cases import fields

PAPER=Path(__file__).resolve().parents[2]/'manuscript'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    jobs=json.loads((OUT/'main.json').read_text())
    required=jobs+json.loads((OUT/'refinement.json').read_text())
    for batch in ['temporal','refinement_strong','temporal_strong']:
        if (OUT/f'{batch}.json').exists():required+=json.loads((OUT/f'{batch}.json').read_text())
    missing=[j['name'] for j in required if not (OUT/'runs'/j['name']/'publication_complete.json').exists()]
    if missing and not args.partial: raise RuntimeError(f'{len(missing)} required runs unfinished: {missing}')
    pairs=[]; arrays={}; metrics=[]; provenance=[]; refinements=[]
    for i in range(0,len(jobs),2):
        j,k=jobs[i:i+2]
        if any(n['name'] in missing for n in [j,k]): continue
        r=load(OUT/'runs'/j['name'],True);s=load(OUT/'runs'/k['name'],True)
        met=pair(r,s); met.update(width_km=j['width_km'],a_barrier=j['a_barrier'])
        pairs.append(met)
        for run in [r,s]:
            rec=run['record'];n=rec['name']; metrics.append(run['metrics'])
            path=OUT/'inputs'/rec['input_name']/(rec['mode']+'.bin')
            assert digest(path)==rec['input_sha256']
            cfg=json.loads(path.with_name('configuration.json').read_text())
            assert cfg['neural_model_sha256']==digest(OUT.parent/'results_rupture/source_family/source_model.npz')
            prefix=rec['input_name'].replace('_N360','')+'_'+rec['mode']
            if (rec['width_km'],rec['a_barrier']) in [(1.,.012),(2.,.014)]:
                for key in ['x','y','final','work','time','diagnostics','regional','stations']: arrays[prefix+'_'+key]=run[key]
                arrays[prefix+'_centerline_velocity']=run['velocity'][:,:,rec['nx']//4]
            provenance.append(dict(name=n,input_sha256=rec['input_sha256'],driver_sha256=rec['driver_sha256'],
                final_sha256=digest(OUT/'runs'/n/'solution_final_fields.bin')))
    # Refinement comparisons normalize using the finer reference on common nodes.
    for w,aa in [(1.,.012),(2.,.012),(2.,.014)]:
        candidates=[j for j in required if j['width_km']==w and j['a_barrier']==aa and j['mode']=='reference']
        for j in candidates:
            if j['name'] in missing: continue
            r=load(OUT/'runs'/j['name'],True)
            if r['metrics']['name'] not in [m['name'] for m in metrics]: metrics.append(r['metrics'])
            if j['nx']!=720: continue
            fine=r
            kn=next(k for k in required if k['nx']==720 and k['mode']=='thera' and k['width_km']==w and k['a_barrier']==aa)
            if kn['name'] not in missing:
                pred=load(OUT/'runs'/kn['name'],True)
                met=pair(fine,pred);met.update(width_km=w,a_barrier=aa,refinement=True);pairs.append(met)
                metrics.append(pred['metrics'])
            for k in candidates:
                if k['nx']!=360 or k['name'] in missing:continue
                coarse=load(OUT/'runs'/k['name'],True)
                assert np.array_equal(coarse['x'],fine['x'][::2]) and np.array_equal(coarse['y'],fine['y'][::2])
                assert np.allclose(coarse['time'],fine['time'],rtol=0,atol=1e-8)
                mask=coarse['active'];u=fine['velocity'][:,::2,::2][:,mask].astype(float);v=coarse['velocity'][:,mask]
                a=fine['final'][::2,::2,1];b=coarse['final'][...,1];both=mask&(a>=0)&(b>=0)
                refinements.append(dict(coarse=k['name'],fine=j['name'],width_km=w,a_barrier=aa,
                    coarse_dt_s=k['dt'],field_relative_L2=float(np.linalg.norm(v-u)/np.linalg.norm(u)),
                    arrival_rmse_s=float(np.sqrt(np.mean((a[both]-b[both])**2))),
                    receiver_outcome_agrees=coarse['metrics']['status']==fine['metrics']['status'],
                    coarse_receiver_fraction=coarse['metrics']['receiver_fraction'],fine_receiver_fraction=fine['metrics']['receiver_fraction'],
                    relative_work_difference=abs(coarse['metrics']['whole_fault_work_J']/fine['metrics']['whole_fault_work_J']-1)))
    pilot=[]
    for folder in sorted((OUT/'runs').glob('*N180*')):
        if (folder/'solution_metadata.json').exists():pilot.append(load(folder)['metrics'])
    result=dict(main_pairs=[p for p in pairs if not p.get('refinement')],fine_pairs=[p for p in pairs if p.get('refinement')],
        runs=metrics,pilot_runs=pilot,refinement_comparisons=refinements,
        all_required_complete=not missing,missing=missing,
        activation_definition='Receiver area fraction reaching |V|>=0.5 m/s by 15 s >=0.5; arrest <=0.01; otherwise partial.',
        provenance=provenance)
    (OUT/'barrier_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    if not missing or (args.partial and len(result['main_pairs'])==12):
        (PAPER/'data/barrier_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        x,y,q,*_=fields(360,2,.012)
        arrays.update(setup_x=x,setup_y=y,setup_a_minus_b=q[...,0]-q[...,1])
        np.savez_compressed(PAPER/'data/barrier_figure_data.npz',**arrays)
    print(json.dumps(dict(main_pairs=len(result['main_pairs']),fine_pairs=len(result['fine_pairs']),
        remaining=len(missing),outcome_agreement=[p['outcome_agrees'] for p in pairs],
        max_main_field_error=max([p['field_relative_L2'] for p in result['main_pairs']],default=0)),indent=2))

if __name__=='__main__':main()
