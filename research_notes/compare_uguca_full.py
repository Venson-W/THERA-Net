"""Full 15-second unmodified UGUCA/adapter equivalence check, exact parameters."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent
nf=root/'uguca-native-ext4'
af=root/'uguca-adapter-ext4'
native='TPV101_Nx180_Nz90_s2.00_tf0.17_npc1'
nt=np.loadtxt(nf/(native+'.time'))[:,1]
at=np.loadtxt(af/'adapter.time')[:,1]
report={}
for field in ('top_velo_0','top_disp_0'):
    n=2*np.fromfile(nf/(native+'-DataFiles')/(field+'.out'),dtype=np.float32).reshape(-1,180,90).astype(float)
    a=2*np.fromfile(af/'adapter-DataFiles'/(field+'.out'),dtype=np.float32).reshape(-1,180,90).astype(float)
    rows=[]
    for i,t in enumerate(nt):
        hits=np.where(np.isclose(at,t,atol=1e-8,rtol=0))[0]
        if not len(hits): continue
        d=a[hits[0]]-n[i]
        rows.append({'time_s':float(t),'max_abs':float(np.max(np.abs(d))),
            'relative_L2':float(np.linalg.norm(d)/max(np.linalg.norm(n[i]),1e-30)),
            'native_max_abs':float(np.max(np.abs(n[i]))),
            'adapter_max_abs':float(np.max(np.abs(a[hits[0]])))})
    report[field]={'common_snapshots':len(rows),'maximum_absolute_error':max(r['max_abs'] for r in rows),
        'maximum_relative_L2_nonzero_time':max(r['relative_L2'] for r in rows if r['time_s']>0),
        'final':rows[-1],'snapshots':rows}
report['note']='Relative slip and speed multiply native top-face output by 2. Float32 dump precision. Final adapter init(false) solves traction equilibrium without advancing state.'
(root/'uguca_full_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for field,value in report.items():
    if isinstance(value,dict): print(field, {k:v for k,v in value.items() if k!='snapshots'})

for mode in ('reference','pinet'):
    data=np.genfromtxt(root.parent/'PiNet-main/Fault mechanics/results_rupture/runs'/f'uniform_N360_{mode}'/'solution_diagnostics.csv',delimiter=',',names=True)
    print('uniform_N360_'+mode,{n:[float(data[n].min()),float(data[n].max())] for n in data.dtype.names})
