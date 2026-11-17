"""Read completed barrier runs without modifying raw solver outputs."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_barrier'

def load(folder, snapshots=False):
    folder=Path(folder)
    rec=json.loads((folder/'execution_record.json').read_text())
    assert rec['returncode']==0
    n=rec['nx']; nz=n//2; dx=72000/n
    meta=json.loads((folder/'solution_metadata.json').read_text())
    assert meta['nx']==n and meta['dt_s']==rec['dt']
    final=np.fromfile(folder/'solution_final_fields.bin',dtype='<f8').reshape(n,nz,5)
    work=np.fromfile(folder/'solution_friction_work.bin',dtype='<f8').reshape(n,nz)
    diag=np.genfromtxt(folder/'solution_diagnostics.csv',delimiter=',',names=True)
    regional=np.genfromtxt(folder/'solution_regional.csv',delimiter=',',names=True)
    stations=np.genfromtxt(folder/'solution_stations.csv',delimiter=',',names=True)
    assert np.isfinite(final).all() and np.all(final[...,3]>0)
    nt=round(rec.get('duration',15)/rec['dt'])
    assert len(diag)==len(regional)==len(stations)==nt
    assert abs(diag['time'][-1]-rec.get('duration',15))<1e-8
    x=np.arange(n)*dx-36000; y=np.arange(nz)*dx-18000
    active=(abs(x[:,None])<=15000)&(abs(y[None,:])<=7500)
    edge=rec['width_km']*500+500
    receiver=active&(x[:,None]>edge)
    barrier=active&(abs(x[:,None])<=edge)
    frac=float(np.mean(final[...,1][receiver]>=0))
    mask=receiver&(final[...,1]>=0)
    arrival=float(final[...,1][mask].min()) if mask.any() else None
    status='activated' if frac>=.5 else ('arrested' if frac<=.01 else 'partial')
    # Midline connectivity at 0.5 m/s is a diagnostic, not proof of causal triggering.
    midline=(x>=-9000)&(x<=8000)
    connected=bool(np.all(final[:,nz//2,1][midline]>=0))
    metrics=dict(name=folder.name,nx=n,dx_m=dx,dt_s=rec['dt'],mode=rec['mode'],
        width_km=rec['width_km'],a_barrier=rec['a_barrier'],status=status,
        receiver_fraction=frac,receiver_first_arrival_s=arrival,
        receiver_peak_m_s=float(final[...,2][receiver].max()),
        midline_threshold_connected=connected,
        barrier_work_J=float(work[barrier].sum()*dx**2),
        receiver_work_J=float(work[receiver].sum()*dx**2),
        whole_fault_work_J=float(work.sum()*dx**2),
        min_theta_s=float(diag['minimum_theta_s'].min()),
        min_power_W_m2=float(diag['minimum_power_W_m2'].min()),
        min_integrated_power_W=float(diag['integrated_friction_power_W'].min()),
        final_receiver_mean_slip_m=float(final[...,4][receiver].mean()),
        runtime_s=rec['process_wall_seconds'])
    assert np.isclose(work.sum()*dx**2,np.sum(diag['integrated_friction_power_W'])*rec['dt'],rtol=1e-10)
    assert np.isclose(metrics['barrier_work_J'],np.sum(regional['barrier_power_W'])*rec['dt'],rtol=1e-10)
    assert np.isclose(metrics['receiver_work_J'],np.sum(regional['right_power_W'])*rec['dt'],rtol=1e-10)
    assert np.isclose(regional['receiver_ruptured_area_m2'][-1],mask.sum()*dx**2)
    result=dict(metrics=metrics,record=rec,meta=meta,final=final,work=work,
                diagnostics=diag,regional=regional,stations=stations,x=x,y=y,
                active=active,receiver=receiver,barrier=barrier)
    if snapshots:
        time=np.loadtxt(folder/'solution.time',ndmin=2)[:,1]
        assert time[0]==0 and abs(time[-1]-rec.get('duration',15))<1e-8
        vel=2*np.fromfile(folder/'solution-DataFiles/top_velo_0.out',dtype='<f4').reshape(len(time),n,nz)
        assert np.isfinite(vel).all()
        assert np.all(final[...,2]>=abs(vel).max(axis=0)*(1-4e-6)-1e-10)
        result.update(time=time,velocity=vel)
    return result

def pair(ref,pin):
    assert ref['record']['nx']==pin['record']['nx'] and np.array_equal(ref['time'],pin['time'])
    mask=ref['active']; a=ref['final'][...,1];b=pin['final'][...,1]
    both=mask&(a>=0)&(b>=0)
    def rel(a,b): return float(np.linalg.norm(np.asarray(a,dtype=float)-b)/np.linalg.norm(np.asarray(a,dtype=float)))
    return dict(reference=ref['metrics']['name'],thera=pin['metrics']['name'],
        field_relative_L2=rel(ref['velocity'][:,mask],pin['velocity'][:,mask]),
        final_slip_relative_L2=rel(ref['final'][...,4][mask],pin['final'][...,4][mask]),
        moment_rate_relative_L2=rel(ref['diagnostics']['moment_rate_Nm_s'],pin['diagnostics']['moment_rate_Nm_s']),
        work_relative_error=abs(pin['metrics']['whole_fault_work_J']/ref['metrics']['whole_fault_work_J']-1),
        arrival_rmse_s=float(np.sqrt(np.mean((a[both]-b[both])**2))) if both.any() else None,
        rupture_mask_disagreement_nodes=int(np.sum(((a>=0)!=(b>=0))&mask)),
        outcome_agrees=ref['metrics']['status']==pin['metrics']['status'])

if __name__=='__main__':
    rows=[]
    for folder in sorted((OUT/'runs').glob('*')):
        if not (folder/'solution_metadata.json').exists() or not (folder/'solution_final_fields.bin').exists(): continue
        r=load(folder)['metrics']; rows.append(r)
        print(f"{r['name']}: {r['status']} fraction={r['receiver_fraction']:.4f} Vrec={r['receiver_peak_m_s']:.4g} arrival={r['receiver_first_arrival_s']}")
    (OUT/'run_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
