"""WSL runner: isolated ext4 computation, complete-run publication and hashes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,os,shutil,subprocess,time

FM=Path(__file__).resolve().parents[1]
WORK=FM.parents[1]
ROOT=Path('/var/tmp/titan-barrier-20261113')
DEST=FM/'results_barrier/runs'

def run(job):
    name=job['name']; folder=ROOT/'runs'/name
    folder.mkdir(parents=True,exist_ok=True)
    source=FM/'results_barrier/inputs'/job['input_name']/(job['mode']+'.bin')
    fields=folder/'fields.bin'; shutil.copy2(source,fields)
    command=['mpirun','--allow-run-as-root','--bind-to','none','-np','8',str(ROOT/'barrier_driver'),
        str(job['nx']),str(job.get('duration',15)),str(job['dt']),str(job.get('dump_dt',.1)),
        str(fields),'solution',str(job.get('pc',1)),str(500*job['width_km']+500)]
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(ROOT)+':'+env.get('LD_LIBRARY_PATH','')
    env['OMP_NUM_THREADS']='1';env['UGUCA_KERNEL_PATH']=str(ROOT/'kernels')+'/'
    start=time.time();print('START '+name,flush=True)
    with (folder/'run.log').open('w') as f:
        result=subprocess.run(command,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT)
    record=dict(job,command=command,returncode=result.returncode,started_unix=start,
        process_wall_seconds=time.time()-start,mpi_processes=8,
        driver_sha256=hashlib.sha256((ROOT/'barrier_driver').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(fields.read_bytes()).hexdigest())
    (folder/'execution_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'COMPUTED {name} status={result.returncode} wall={time.time()-start:.1f}s',flush=True)
    return folder,record

def main():
    p=argparse.ArgumentParser();p.add_argument('jobs');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();jobs=json.loads(Path(args.jobs).read_text())
    ROOT.mkdir(exist_ok=True,parents=True);DEST.mkdir(exist_ok=True,parents=True)
    for src,dst in [(WORK/'research_notes/pinet-rupture-build/barrier_driver',ROOT/'barrier_driver'),
                    (WORK/'research_notes/uguca-build/src/libuguca.so',ROOT/'libuguca.so')]:
        if not dst.exists() or hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dst.read_bytes()).digest():
            shutil.copy2(src,dst)
    shutil.copytree(WORK/'research_notes/uguca/kernels',ROOT/'kernels',dirs_exist_ok=True)
    pending=[]
    for j in jobs:
        path=DEST/j['name']/'execution_record.json'
        if path.exists() and json.loads(path.read_text()).get('returncode')==0:
            old=json.loads(path.read_text())
            current=FM/'results_barrier/inputs'/j['input_name']/(j['mode']+'.bin')
            if old['input_sha256']!=hashlib.sha256(current.read_bytes()).hexdigest() or old['driver_sha256']!=hashlib.sha256((ROOT/'barrier_driver').read_bytes()).hexdigest():
                raise RuntimeError('Existing completed output has different provenance: '+j['name'])
            print('EXISTS '+j['name'],flush=True)
        else: pending.append(j)
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in pending]):
            folder,record=future.result()
            shutil.copytree(folder,DEST/folder.name,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('fields.bin'))
            (DEST/folder.name/'publication_complete.json').write_text(json.dumps(record)+'\n')
            records.append(record);print('PUBLISHED '+folder.name,flush=True)
    failures=[r['name'] for r in records if r['returncode']]
    print(f'COMPLETE runs={len(records)} failures={failures}',flush=True)
    return bool(failures)

if __name__=='__main__': raise SystemExit(main())
