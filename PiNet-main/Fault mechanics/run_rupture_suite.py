"""Execute independent matched UGUCA runs in WSL, preserving logs and arguments.

Example: python run_rupture_suite.py --nx 180 360 --jobs 2 --ranks 8
The executable and inputs must already have been built/generated. This runner
does not install software, modify UGUCA, or synthesize missing solver outputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]

def linux_path(path):
    p=str(Path(path).resolve()).replace('\\','/')
    return '/mnt/'+p[0].lower()+p[2:] if len(p)>1 and p[1]==':' else p

def run(args,case,nx,mode):
    folder=ROOT/'results_rupture'/'runs'/f'{case}_N{nx}_{mode}'
    folder.mkdir(parents=True,exist_ok=True)
    dt=.02*180/nx
    source=ROOT/'results_rupture'/'inputs'/f'{case}_N{nx}'/f'{mode}.bin'
    exe=WORKSPACE/'research_notes'/'pinet-rupture-build'/'rupture_driver'
    argv=['mpirun','--allow-run-as-root','-np',str(args.ranks),linux_path(exe),str(nx),str(args.duration),str(dt),'.1',linux_path(source),'solution','1']
    cmd=['wsl.exe','-d',args.distribution,'--cd',linux_path(folder),'--']+argv if os.name=='nt' else argv
    manifest=dict(case=case,nx=nx,mode=mode,dt_s=dt,arguments=cmd,mpi_ranks=args.ranks)
    (folder/'command.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started=time.perf_counter()
    with (folder/'solver.log').open('w',encoding='utf-8') as logfile:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',cwd=folder)
        for line in p.stdout:
            logfile.write(line);logfile.flush()
            if 'step ' in line or 'Completed' in line or 'error' in line.lower():
                print(f'{case} N{nx} {mode}: {line.strip()}',flush=True)
        code=p.wait()
    manifest.update(exit_code=code,wall_seconds_including_launch=time.perf_counter()-started)
    (folder/'command.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if code: raise RuntimeError(f'{case} N{nx} {mode} failed: exit {code}; see {folder}/solver.log')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--nx',type=int,nargs='+',default=[180,360])
    ap.add_argument('--cases',nargs='+',default=['heterogeneous_D16','uniform','unseen_D13'])
    ap.add_argument('--modes',nargs='+',default=['reference','pinet'])
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--ranks',type=int,default=8)
    ap.add_argument('--duration',type=float,default=15.)
    ap.add_argument('--distribution',default='Ubuntu')
    args=ap.parse_args()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs=[pool.submit(run,args,c,n,m) for n in args.nx for c in args.cases for m in args.modes]
        for f in as_completed(jobs): print(json.dumps(f.result()),flush=True)
