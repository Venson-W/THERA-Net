"""Run workspace UGUCA cases on WSL ext4 and publish complete outputs serially.

Run this script with WSL Python. It stages its runtime and inputs at ROOT.
The upstream dumper flushes each coordinate line; writing those directly through
the WSL /mnt filesystem is slow. Computation and fine-grained writes stay on
ext4; copytree below performs a single sequential publication per finished run.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
FM=HERE.parent
WORK=FM.parent.parent
# /var/tmp survives WSL shutdown; /tmp may be cleaned at each WSL startup.
ROOT=Path(os.environ.get('THERA_RUPTURE_RUNTIME','/var/tmp/thera-rupture-20261112'))
DEST=FM/'results_rupture'/'runs'
CASES=('uniform','heterogeneous_D16','unseen_D13')
JOBS=[]
for nx,dt in ((360,.01),(180,.02)):
    for case in CASES:
        for mode in ('reference','thera'):
            JOBS.append((case,nx,mode,dt,''))
for mode in ('reference','thera'):
    JOBS.append(('uniform',720,mode,.005,''))
JOBS.append(('uniform',360,'reference',.005,'_dt05'))

def run(job):
    case,nx,mode,dt,suffix=job
    name=f'{case}_N{nx}_{mode}{suffix}'
    folder=ROOT/'runs'/name
    folder.mkdir(parents=True,exist_ok=True)
    fields=ROOT/'inputs'/f'{case}_N{nx}'/(mode+'.bin')
    command=['mpirun','--allow-run-as-root','--bind-to','none','-np','8',
             str(ROOT/'rupture_driver'),str(nx),'15',str(dt),'.1',str(fields),'solution','1']
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(ROOT)+':'+env.get('LD_LIBRARY_PATH','')
    env['OMP_NUM_THREADS']='1'
    env['UGUCA_KERNEL_PATH']=str(ROOT/'kernels')+'/'
    start=time.time()
    print(f'START {name}',flush=True)
    with (folder/'run.log').open('w') as stream:
        result=subprocess.run(command,cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record={'name':name,'command':command,'returncode':result.returncode,
            'process_wall_seconds':time.time()-start,'started_unix':start,
            'driver_sha256':hashlib.sha256((ROOT/'rupture_driver').read_bytes()).hexdigest(),
            'input_sha256':hashlib.sha256(fields.read_bytes()).hexdigest(),
            'runtime_directory':str(ROOT),'mpi_processes':8}
    (folder/'execution_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'COMPUTED {name} status={result.returncode} wall={record["process_wall_seconds"]:.3f}s',flush=True)
    return folder,record

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for source,target in (
        (WORK/'research_notes/thera-rupture-build/rupture_driver',ROOT/'rupture_driver'),
        (WORK/'research_notes/uguca-build/src/libuguca.so',ROOT/'libuguca.so')):
        shutil.copy2(source,target)
    shutil.copytree(WORK/'research_notes/uguca/kernels',ROOT/'kernels',dirs_exist_ok=True)
    shutil.copytree(FM/'results_rupture/inputs',ROOT/'inputs',dirs_exist_ok=True)
    print(f'STAGED runtime at {ROOT}',flush=True)
    DEST.mkdir(parents=True,exist_ok=True)
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(run,job) for job in JOBS]
        for future in as_completed(pending):
            folder,record=future.result()
            # One main thread copies complete outputs: never simultaneous /mnt writes.
            shutil.copytree(folder,DEST/folder.name,dirs_exist_ok=True)
            records.append(record)
            (ROOT/'batch_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
            shutil.copy2(ROOT/'batch_manifest.json',DEST/'batch_manifest.json')
            print(f'PUBLISHED {folder.name}',flush=True)
    failures=[r['name'] for r in records if r['returncode']]
    print(f'BATCH COMPLETE runs={len(records)} failures={failures}',flush=True)
    return bool(failures)

if __name__=='__main__':
    raise SystemExit(main())
