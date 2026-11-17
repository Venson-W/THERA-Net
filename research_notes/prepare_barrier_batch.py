from pathlib import Path
import sys,json,argparse
fm=Path('src/fault_mechanics').resolve()
sys.path.insert(0,str(fm))
from barrier_cases import prepare,case_name
p=argparse.ArgumentParser();p.add_argument('batch')
p.add_argument('--nx',type=int,default=180);p.add_argument('--dt',type=float,default=.01)
p.add_argument('--widths',type=float,nargs='+',default=[0,.5,1,2,4,6])
p.add_argument('--a',type=float,nargs='+',default=[.018])
p.add_argument('--modes',nargs='+',default=['reference'])
p.add_argument('--duration',type=float,default=15);p.add_argument('--pc',type=int,default=1)
args=p.parse_args();jobs=[]
for a in args.a:
    for w in args.widths:
        target=prepare(args.nx,w,a)
        for mode in args.modes:
            name=f'{case_name(w,a)}_N{args.nx}_{mode}_dt{args.dt:g}_T{args.duration:g}_pc{args.pc}'.replace('.','p')
            jobs.append(dict(name=name,input_name=target.name,nx=args.nx,dt=args.dt,duration=args.duration,
                mode=mode,width_km=w,a_barrier=a,pc=args.pc,dump_dt=.1))
path=fm/'results_barrier'/f'{args.batch}.json';path.write_text(json.dumps(jobs,indent=2)+'\n')
print(path,len(jobs))
