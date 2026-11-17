"""WSL regression: instrumentation must leave the existing dynamics unchanged."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
import numpy as np
work=Path(__file__).resolve().parents[1]
root=Path('/var/tmp/titan-barrier-20261113')
folder=root/'instrumentation_check';folder.mkdir(exist_ok=True)
exe=folder/'rupture_driver';shutil.copy2(work/'research_notes/pinet-rupture-build/rupture_driver',exe)
name='w0_a0p018_N180_reference_dt0p01_T15_pc1'
fields=root/'runs'/name/'fields.bin'
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root);env['OMP_NUM_THREADS']='1'
env['UGUCA_KERNEL_PATH']=str(root/'kernels')+'/'
cmd=['mpirun','--allow-run-as-root','--bind-to','none','-np','8',str(exe),'180','15','.01','.1',str(fields),'solution','1']
if not (folder/'solution_final_fields.bin').exists():
    with (folder/'run.log').open('w') as f:subprocess.run(cmd,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
hashes={}
for file in ['solution_final_fields.bin','solution-DataFiles/top_velo_0.out','solution-DataFiles/top_disp_0.out','solution_stations.csv']:
    # Station locations intentionally differ; compare only the unchanged fields.
    if file.endswith('stations.csv'):continue
    old=hashlib.sha256((folder/file).read_bytes()).hexdigest()
    new=hashlib.sha256((root/'runs'/name/file).read_bytes()).hexdigest()
    hashes[file]=dict(original=old,instrumented=new)
a=np.fromfile(folder/'solution_final_fields.bin',dtype='<f8').reshape(-1,5)
b=np.fromfile(root/'runs'/name/'solution_final_fields.bin',dtype='<f8').reshape(-1,5)
relative=np.linalg.norm(a-b,axis=0)/np.linalg.norm(a,axis=0)
absolute=abs(a-b).max(axis=0)
assert np.all(relative<1e-10)
assert np.all(absolute[:2]<1e-8) and absolute[2]<1e-8 and absolute[4]<1e-10
assert np.array_equal(a[:,:2]>=0,b[:,:2]>=0)
snapshot={}
for name2,tol in [('top_velo_0.out',1e-8),('top_disp_0.out',1e-10)]:
    a=np.fromfile(folder/'solution-DataFiles'/name2,dtype='<f4')
    b=np.fromfile(root/'runs'/name/'solution-DataFiles'/name2,dtype='<f4')
    diff=float(2*abs(a-b).max());assert diff<tol
    snapshot[name2]=diff
record=dict(test='Numerical equivalence of original and regionally instrumented drivers',
    nx=180,dt_s=.01,duration_s=15,reference_case=name,passed=True,output_sha256=hashes)
record.update(initial_bitwise_comparison='Not identical; numerical differences quantified below. Source comparison confirms only diagnostic additions and station changes.',
    final_field_order=['arrival_1e-3','arrival_0.5','peak_speed','theta','slip'],
    final_field_relative_L2=relative.tolist(),final_field_max_absolute=absolute.tolist(),
    snapshot_max_absolute_relative_jump=snapshot,
    acceptance='Each final field relative L2 <1e-10; arrivals <1e-8 s; peak/snapshot rates <1e-8 m/s; slip <1e-10 m; identical arrival masks.')
dest=work/'PiNet-main/Fault mechanics/results_barrier/instrumentation_verification.json'
dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
