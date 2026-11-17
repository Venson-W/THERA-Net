"""Validate, package and compare completed UGUCA reference/PiNet rupture runs.

Raw solver files are never changed. Field metrics use the geometric VW rectangle
|x|<=15 km, |y|<=7.5 km; moment rate integrates the complete periodic fault.
Arrival times come from every integration step, not from the 0.1-s snapshots.
No spatial smoothing is applied to figures, contours or metrics.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT=HERE/"results_rupture"
CASES=("heterogeneous_D16","uniform","unseen_D13")
CASE_LABELS={"heterogeneous_D16":"Heterogeneous stress, $D=1.6$",
             "uniform":"Uniform initial stress", "unseen_D13":"Heterogeneous stress, $D=1.3$"}
CASE_TABLE={"heterogeneous_D16":"$D=1.6$", "uniform":"Uniform", "unseen_D13":"$D=1.3$"}
STATIONS=tuple("ABCDEF")
os.environ.setdefault("MPLCONFIGDIR",str(HERE/".mplconfig"))
local_deps=ROOT/"research_notes"/"python_deps"
if local_deps.exists(): sys.path.insert(0,str(local_deps))


class IncompleteRun(RuntimeError):
    pass


@dataclass
class Run:
    name: str
    case: str
    mode: str
    nx: int
    meta: dict
    time: np.ndarray
    velocity: np.ndarray
    final: np.ndarray
    stations: np.ndarray
    diagnostics: np.ndarray
    diagnostic_names: list
    x: np.ndarray
    y: np.ndarray

    @property
    def vw(self):
        return (np.abs(self.x[:,None])<=15000.+1e-8)&(np.abs(self.y[None,:])<=7500.+1e-8)

    def diagnostic(self,name):
        return self.diagnostics[:,self.diagnostic_names.index(name)]


def _parse_name(name):
    match=re.fullmatch(r"(heterogeneous_D16|uniform|unseen_D13)_N(\d+)_(reference|pinet)(?:_dt\d+)?",name)
    if not match: raise ValueError(f"Unrecognized run directory {name}")
    return match[1],int(match[2]),match[3]


def read_csv(filename):
    data=np.genfromtxt(filename,delimiter=",",names=True,dtype=float)
    data=np.atleast_1d(data)
    names=list(data.dtype.names or [])
    if not names or not len(data): raise IncompleteRun(f"Empty CSV {filename}")
    result=np.column_stack([data[name] for name in names])
    if not np.all(np.isfinite(result)): raise IncompleteRun(f"Nonfinite/partial CSV {filename}")
    return result,names


def matching_indices(sample_times,full_times,tolerance=2e-8):
    """Use actual common times; do not interpolate field or diagnostic samples."""
    indices=np.searchsorted(full_times,sample_times)
    indices=np.clip(indices,0,len(full_times)-1)
    previous=np.maximum(indices-1,0)
    choose_previous=np.abs(full_times[previous]-sample_times)<np.abs(full_times[indices]-sample_times)
    indices=np.where(choose_previous,previous,indices)
    if not np.allclose(full_times[indices],sample_times,rtol=0,atol=tolerance):
        raise ValueError("Requested common time samples are not present in both runs")
    return indices


def _load_initial_audit(case,nx,mode,manifest):
    entry=next(c for c in manifest["cases"] if c["name"]==case)
    run=next((r for r in entry["runs"] if r["nx"]==nx and r["mode"]==mode),None)
    audit={"case_configuration":{k:v for k,v in entry.items() if k!="runs"},
           "initial_state_convention":manifest.get("initial_state_convention"),
           "source_figure_relationship":manifest.get("source_figure_relationship"),
           "source_model_sha256":manifest.get("model_sha256") if mode=="pinet" else None}
    if run:
        inputfile=HERE/Path(run["file"].replace("\\","/"))
        audit["input_sha256"]=run.get("sha256")
        if inputfile.is_file() and inputfile.stat().st_size==nx*(nx//2)*7*8:
            inputs=np.memmap(inputfile,dtype="<f8",mode="r",shape=(nx,nx//2,7))
            audit["initial_theta_min_s"]=float(inputs[...,4].min())
            audit["initial_theta_max_s"]=float(inputs[...,4].max())
            audit["initial_stress_mean_Pa"]=float(inputs[...,5].mean())
            audit["initial_stress_std_Pa"]=float(inputs[...,5].std())
            audit["effective_coefficient_min"]=inputs[...,:4].min(axis=(0,1)).tolist()
            audit["effective_coefficient_max"]=inputs[...,:4].max(axis=(0,1)).tolist()
    return audit


def raw_completion_fingerprint(directory):
    """Detect replaced/recomputed runs without trusting preserved copy timestamps."""
    digest=hashlib.sha256()
    for name in ["solution_final_fields.bin","solution_metadata.json","solution.time",
                 "solution_stations.csv","solution_diagnostics.csv","solution_wall_seconds.txt"]:
        path=directory/name
        if not path.is_file() or path.stat().st_size==0:return None
        digest.update(name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    execution=directory/"execution_record.json"
    if execution.is_file():digest.update(execution.read_bytes())
    return digest.hexdigest()


def load_raw(directory,manifest):
    case,nx,mode=_parse_name(directory.name)
    required=[directory/"solution_metadata.json",directory/"solution_final_fields.bin",
              directory/"solution_wall_seconds.txt",directory/"solution.time",
              directory/"solution_stations.csv",directory/"solution_diagnostics.csv",
              directory/"solution-DataFiles"/"top_velo_0.out",
              directory/"solution-DataFiles"/"top_disp_0.out"]
    missing=[p.name for p in required if not p.is_file() or p.stat().st_size==0]
    if missing: raise IncompleteRun(f"Missing/unfinished: {', '.join(missing)}")
    meta=json.loads(required[0].read_text())
    nz=nx//2
    if meta["nx"]!=nx or meta["nz"]!=nz: raise ValueError("Metadata grid disagrees with directory")
    if not np.isclose(meta["dx_m"],72000./nx): raise ValueError("Unexpected mesh spacing")
    timefile=np.loadtxt(directory/"solution.time",ndmin=2)
    if timefile.shape[1]!=2: raise ValueError("Expected two-column dump step/time file")
    time=timefile[:,1]
    if len(time)<2 or np.any(np.diff(time)<=0): raise IncompleteRun("Incomplete/nonmonotone snapshot times")
    if not np.isclose(time[0],0,atol=1e-12) or not np.isclose(time[-1],meta["duration_s"],atol=1e-8):
        raise IncompleteRun("Snapshots do not span the declared complete run")
    expected_step=np.rint(time/meta["dt_s"]).astype(int)
    if not np.array_equal(timefile[:,0].astype(int),expected_step): raise ValueError("Dump times and steps disagree")
    frames=len(time)
    for name in ["top_velo_0.out","top_disp_0.out"]:
        path=directory/"solution-DataFiles"/name
        if path.stat().st_size!=frames*nx*nz*4: raise IncompleteRun(f"Wrong/partial float32 snapshot size: {name}")
    # UGUCA stores one face's velocity/displacement; symmetric jump is twice it.
    velocity=2*np.fromfile(directory/"solution-DataFiles"/"top_velo_0.out",dtype="<f4").reshape(frames,nx,nz)
    if not np.all(np.isfinite(velocity)): raise ValueError("Nonfinite velocity snapshots")
    final_path=directory/"solution_final_fields.bin"
    if final_path.stat().st_size!=nx*nz*5*8: raise IncompleteRun("Partial final-field array")
    final=np.fromfile(final_path,dtype="<f8").reshape(nx,nz,5)
    if not np.all(np.isfinite(final)) or np.any(final[...,3]<=0): raise ValueError("Invalid final fields/state")
    last_displacement=np.memmap(directory/"solution-DataFiles"/"top_disp_0.out",dtype="<f4",mode="r",shape=(frames,nx,nz))[-1]
    if not np.allclose(final[...,4],2*last_displacement,rtol=3e-6,atol=1e-9):
        raise ValueError("Final slip disagrees with symmetric displacement jump")
    del last_displacement
    stations,station_names=read_csv(directory/"solution_stations.csv")
    diagnostics,names=read_csv(directory/"solution_diagnostics.csv")
    steps=int(round(meta["duration_s"]/meta["dt_s"]))
    if stations.shape!=(steps,7) or station_names!=["time",*STATIONS]:
        raise IncompleteRun("Wrong/partial station time history")
    if diagnostics.shape[0]!=steps: raise IncompleteRun("Wrong/partial diagnostics history")
    expected=np.arange(1,steps+1)*meta["dt_s"]
    if not np.allclose(stations[:,0],expected,rtol=0,atol=2e-8) or not np.allclose(diagnostics[:,0],expected,rtol=0,atol=2e-8):
        raise IncompleteRun("Full-step diagnostics/stations do not span declared times")
    if not np.all(final[...,2]>=np.max(np.abs(velocity),axis=0)*(1-3e-6)-1e-10):
        raise ValueError("Every-step peak field is smaller than a snapshot peak")
    for j in [0,1]:
        arrival=final[...,j]
        if np.any((arrival>=0)&(arrival>meta["duration_s"]+1e-8)): raise ValueError("Arrival outside simulated interval")
    # Check station ordering and the factor of two against matching snapshot sites.
    station_indices=matching_indices(time[1:],stations[:,0])
    station_xy=np.array(meta["stations_actual_xy_m"])
    ix=np.rint((station_xy[:,0]+36000)/meta["dx_m"]).astype(int)
    iy=np.rint((station_xy[:,1]+18000)/meta["dx_m"]).astype(int)
    selected=velocity[1:,ix,iy]
    if not np.allclose(selected,stations[station_indices,1:],rtol=4e-6,atol=1e-9):
        raise ValueError("Station coordinates/order or velocity jump factor inconsistent with snapshots")
    wall=float((directory/"solution_wall_seconds.txt").read_text().strip())
    if not np.isfinite(wall) or wall<0: raise ValueError("Invalid completed-run walltime")
    meta.update(_load_initial_audit(case,nx,mode,manifest))
    execution_path=directory/"execution_record.json"
    if execution_path.is_file():
        execution=json.loads(execution_path.read_text())
        if execution.get("name")!=directory.name or execution.get("returncode")!=0:
            raise IncompleteRun("Execution record does not confirm successful completion")
        if meta.get("input_sha256")!=execution.get("input_sha256"):
            raise ValueError("Executed input hash disagrees with input manifest")
        meta["execution_record"]=execution
    meta.update({"case":case,"mode":mode,"run_name":directory.name,"wall_seconds":wall,
                 "compact_schema_version":1,"raw_completion_fingerprint":raw_completion_fingerprint(directory),
                 "snapshot_storage":"float32 symmetric slip rate; factor 2 applied to UGUCA top velocity",
                 "final_field_columns":["arrival_0p001_s","arrival_0p5_s","peak_speed_m_s","theta_s","final_slip_m"],
                 "shape_checks_passed":True,"station_snapshot_checks_passed":True,
                 "snapshot_frames":frames,"full_steps":steps})
    x=np.arange(nx)*meta["dx_m"]-36000.
    y=np.arange(nz)*meta["dx_m"]-18000.
    return Run(directory.name,case,mode,nx,meta,time,velocity,final,stations,diagnostics,names,x,y)


def save_compact(run,path):
    np.savez_compressed(path,V=run.velocity,final_fields=run.final,stations=run.stations,
                        diagnostics=run.diagnostics,diagnostic_names=np.array(run.diagnostic_names),
                        time=run.time,x_m=run.x,y_m=run.y,metadata_json=json.dumps(run.meta))


def load_compact(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data["metadata_json"]))
        run=Run(meta["run_name"],meta["case"],meta["mode"],meta["nx"],meta,data["time"],data["V"],
                   data["final_fields"],data["stations"],data["diagnostics"],
                   data["diagnostic_names"].tolist(),data["x_m"],data["y_m"])
    case,nx,mode=_parse_name(path.stem)
    if (case,nx,mode)!=(run.case,run.nx,run.mode) or run.name!=path.stem:
        raise ValueError("Compact filename and metadata disagree")
    nz=nx//2;frames=len(run.time)
    dt=float(meta["dt_s"]);duration=float(meta["duration_s"])
    if not(dt>0 and duration>0) or not np.isclose(meta["dx_m"],72000./nx):
        raise ValueError("Invalid compact time/grid metadata")
    steps=int(round(duration/dt))
    if run.velocity.dtype!=np.dtype("float32") or run.velocity.shape!=(frames,nx,nz) or run.final.shape!=(nx,nz,5):
        raise ValueError("Compact field shape/dtype mismatch")
    if run.stations.shape!=(steps,7) or run.diagnostics.shape!=(steps,len(run.diagnostic_names)):
        raise ValueError("Compact full-step history shape mismatch")
    required=["time","moment_rate_Nm_s","integrated_friction_power_W","minimum_theta_s",
              "minimum_power_W_m2","maximum_speed_m_s"]
    if not all(name in run.diagnostic_names for name in required):
        raise ValueError("Compact diagnostics omit required physical quantities")
    if run.x.shape!=(nx,) or run.y.shape!=(nz,) or not np.allclose(run.x,np.arange(nx)*meta["dx_m"]-36000.) or not np.allclose(run.y,np.arange(nz)*meta["dx_m"]-18000.):
        raise ValueError("Compact coordinate grid mismatch")
    if frames<2 or np.any(np.diff(run.time)<=0) or not np.isclose(run.time[0],0,atol=1e-12) or not np.isclose(run.time[-1],duration,atol=1e-8):
        raise ValueError("Compact snapshots do not cover the complete declared interval")
    expected=np.arange(1,steps+1)*dt
    if not np.allclose(run.stations[:,0],expected,rtol=0,atol=2e-8) or not np.allclose(run.diagnostic("time"),expected,rtol=0,atol=2e-8):
        raise ValueError("Compact full-step histories are incomplete")
    if any(not np.all(np.isfinite(array)) for array in [run.time,run.velocity,run.final,run.stations,run.diagnostics]):
        raise ValueError("Compact file contains nonfinite numerical values")
    if np.any(run.final[...,3]<=0) or np.min(run.diagnostic("minimum_theta_s"))<=0:
        raise ValueError("Compact recorded state is not positive")
    for j in [0,1]:
        if np.any((run.final[...,j]>=0)&(run.final[...,j]>duration+1e-8)):
            raise ValueError("Compact arrival is outside the simulation interval")
    if not np.all(run.final[...,2]>=np.max(np.abs(run.velocity),axis=0)*(1-3e-6)-1e-10):
        raise ValueError("Compact peak history is inconsistent with snapshots")
    locations=np.asarray(meta["stations_actual_xy_m"])
    if locations.shape!=(6,2) or not np.all(np.isfinite(locations)):
        raise ValueError("Compact station coordinates are invalid")
    ix=np.rint((locations[:,0]+36000)/meta["dx_m"]).astype(int)
    iy=np.rint((locations[:,1]+18000)/meta["dx_m"]).astype(int)
    indices=matching_indices(run.time[1:],run.stations[:,0])
    if not np.allclose(run.velocity[1:,ix,iy],run.stations[indices,1:],rtol=4e-6,atol=1e-9):
        raise ValueError("Compact station histories disagree with snapshots")
    return run


def relative_l2(pred,true):
    true=np.asarray(true,dtype=float);pred=np.asarray(pred,dtype=float)
    denominator=float(np.sum(true*true))
    return float(np.sqrt(np.sum((pred-true)**2)/denominator)) if denominator>0 else None


def field_relative_l2(pred,true,mask):
    numerator=denominator=0.
    for begin in range(0,len(true),8):
        reference=true[begin:begin+8,mask].astype(np.float64)
        difference=pred[begin:begin+8,mask].astype(np.float64)-reference
        numerator+=float(np.sum(difference*difference));denominator+=float(np.sum(reference*reference))
    return float(np.sqrt(numerator/denominator)) if denominator>0 else None


def arrival_metrics(pred,true,mask):
    p=pred[mask];r=true[mask]
    pm=p>=0;rm=r>=0;matched=pm&rm
    error=p[matched]-r[matched]
    return {"matched_reached_points":int(matched.sum()),"reference_reached_points":int(rm.sum()),
            "pinet_or_coarse_reached_points":int(pm.sum()),"mask_disagreement_points":int(np.sum(pm!=rm)),
            "mask_disagreement_fraction":float(np.mean(pm!=rm)),
            "false_reached_points":int(np.sum(pm&~rm)),"missed_reached_points":int(np.sum(rm&~pm)),
            "RMSE_s":float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "P95_absolute_error_s":float(np.quantile(np.abs(error),.95)) if len(error) else None,
            "mean_signed_error_s":float(error.mean()) if len(error) else None,
            "maximum_absolute_error_s":float(np.max(np.abs(error))) if len(error) else None}


def run_audit(run):
    diagnostics={name:run.diagnostic(name) for name in run.diagnostic_names}
    maximum=float(np.max(diagnostics["maximum_speed_m_s"]))
    minimum_theta=float(np.min(diagnostics["minimum_theta_s"]))
    minimum_power=float(np.min(diagnostics["minimum_power_W_m2"]))
    initial_max=run.meta.get("initial_theta_max_s")
    upper_bound=initial_max+run.meta["duration_s"] if initial_max is not None else None
    peaks=np.max(np.abs(run.stations[:,1:]),axis=0)
    speed_minima=np.min(np.abs(run.velocity),axis=0)
    audit={"maximum_speed_m_s":maximum,"minimum_theta_s":minimum_theta,
           "minimum_friction_power_W_m2":minimum_power,
           "negative_minimum_power_step_count":int(np.sum(diagnostics["minimum_power_W_m2"]<0)),
           "integrated_frictional_energy_J_right_rectangle":float(run.meta["dt_s"]*np.sum(diagnostics["integrated_friction_power_W"])),
           "minimum_integrated_friction_power_W":float(np.min(diagnostics["integrated_friction_power_W"])),
           "maximum_integrated_friction_power_W":float(np.max(diagnostics["integrated_friction_power_W"])),
           "negative_integrated_friction_power_step_count":int(np.sum(diagnostics["integrated_friction_power_W"]<0)),
           "station_peak_speed_m_s":dict(zip(STATIONS,map(float,peaks))),
           "all_stations_exceed_0p001_m_s":bool(np.all(peaks>=.001)),
           "all_stations_exceed_0p5_m_s":bool(np.all(peaks>=.5)),
           "stations_below_0p5_m_s":[name for name,peak in zip(STATIONS,peaks) if peak<.5],
           "source_family_training_domain":{"abs_velocity_m_s":[1e-12,10.],"theta_s":[1e-3,1e24]},
           "speed_exceeds_training_upper_bound":bool(maximum>10.),
           "state_below_training_lower_bound":bool(minimum_theta<1e-3),
           "state_upper_bound_s_from_initial_plus_duration":upper_bound,
           "state_upper_bound_exceeds_training_upper_bound":bool(upper_bound>1e24) if upper_bound is not None else None,
           "snapshot_minimum_abs_speed_m_s":float(speed_minima.min()),
           "snapshot_points_ever_below_training_speed_lower_bound":int(np.sum(speed_minima<1e-12)),
           "positive_state_all_recorded_steps":bool(minimum_theta>0),
           "final_reached_0p5_points_whole_fault":int(np.sum(run.final[...,1]>=0)),
           "wall_seconds":run.meta["wall_seconds"],"shape_checks_passed":run.meta.get("shape_checks_passed",False)}
    execution=run.meta.get("execution_record",{})
    audit["executed_driver_sha256"]=execution.get("driver_sha256")
    audit["executed_input_sha256"]=execution.get("input_sha256")
    audit["execution_returncode"]=execution.get("returncode")
    if "maximum_transverse_speed_m_s" in diagnostics:
        audit["maximum_transverse_speed_m_s"]=float(np.max(diagnostics["maximum_transverse_speed_m_s"]))
    return audit


def compare(pred,true,refinement=False):
    if pred.case!=true.case: raise ValueError("Comparisons must share case")
    if refinement:
        if true.nx%pred.nx: raise ValueError("Refinement grids are not nested")
        ratio=true.nx//pred.nx
    else:
        if true.nx!=pred.nx: raise ValueError("Learned/reference grids differ")
        ratio=1
    if not np.allclose(pred.x,true.x[::ratio],atol=1e-8) or not np.allclose(pred.y,true.y[::ratio],atol=1e-8):
        raise ValueError("Comparison spatial nodes are not common")
    indices=matching_indices(pred.time,true.time)
    ref_velocity=true.velocity[indices,::ratio,::ratio]
    ref_final=true.final[::ratio,::ratio,:]
    mask=pred.vw
    diag_indices=matching_indices(pred.diagnostic("time"),true.diagnostic("time"))
    moments_ref=true.diagnostic("moment_rate_Nm_s")[diag_indices]
    moments_pred=pred.diagnostic("moment_rate_Nm_s")
    result={"case":pred.case,"comparison":"coarse_reference_vs_fine_reference" if refinement else "pinet_vs_reference",
            "nx":pred.nx,"reference_nx":true.nx,"dx_m":pred.meta["dx_m"],
            "reference_dx_m":true.meta["dx_m"],"dt_s":pred.meta["dt_s"],"reference_dt_s":true.meta["dt_s"],
            "fault_region":"|x|<=15000 m and |y|<=7500 m (geometric VW rectangle)",
            "region_points":int(mask.sum()),"snapshot_count":len(pred.time),
            "common_time_alignment":"exact matching saved times; no interpolation",
            "velocity_relative_L2_VW":field_relative_l2(pred.velocity,ref_velocity,mask),
            "final_slip_relative_L2_VW":relative_l2(pred.final[...,4][mask],ref_final[...,4][mask]),
            "peak_speed_relative_L2_VW":relative_l2(pred.final[...,2][mask],ref_final[...,2][mask]),
            "moment_rate_relative_L2_whole_fault":relative_l2(moments_pred,moments_ref),
            "arrival_0p5":arrival_metrics(pred.final[...,1],ref_final[...,1],mask),
            "arrival_0p001":arrival_metrics(pred.final[...,0],ref_final[...,0],mask)}
    if not refinement:
        st_indices=matching_indices(pred.stations[:,0],true.stations[:,0])
        if not np.allclose(pred.meta["stations_actual_xy_m"],true.meta["stations_actual_xy_m"]):
            raise ValueError("Station positions differ between learned/reference pair")
        result["station_velocity_relative_L2"]={s:relative_l2(pred.stations[:,j+1],true.stations[st_indices,j+1])
                                                for j,s in enumerate(STATIONS)}
    else:
        result["interpretation"]="Coupled spatial/time-step refinement; compares fine field sampled at common coarse nodes, not a convergence claim"
    return result


def _setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    return plt


def plot_case(reference,pinet,figure_dir,preview_dir):
    plt=_setup_plot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig=plt.figure(figsize=(7.4,6.8),layout="constrained")
    grid=fig.add_gridspec(3,3,height_ratios=[1.75,1,1])
    ax=fig.add_subplot(grid[0,:])
    time_fields=[reference.final[...,1],pinet.final[...,1]]
    for data,color,style,lw in zip(time_fields,["black","#ca352c"],["solid","dashed"],[.85,.85]):
        masked=np.ma.masked_where(data<0,data)
        valid=data[data>=0]
        if len(valid) and np.max(valid)>np.min(valid):
            levels=np.arange(.5,reference.meta["duration_s"]+.01,.5)
            levels=levels[(levels>np.min(valid))&(levels<np.max(valid))]
            if len(levels):
                contours=ax.contour(reference.x/1000,reference.y/1000,masked.T,levels=levels,
                                    colors=color,linestyles=style,linewidths=lw)
                if color=="black":
                    label_levels=[level for level in levels if np.isclose(level%1.,0.)]
                    if label_levels:
                        labels=ax.clabel(contours,levels=label_levels,inline=True,fontsize=6,fmt="%g s")
                        for label in labels:
                            px,py=label.get_position()
                            if abs(px)>=14.5 or abs(py)>=7.:label.remove()
    ax.add_patch(Rectangle((-15,-7.5),30,15,fill=False,edgecolor=".55",ls=":",lw=.7))
    xy=np.array(reference.meta["stations_actual_xy_m"])/1000
    ax.scatter(xy[:,0],xy[:,1],s=12,color="#236a91",zorder=5)
    for label,(x,y) in zip(STATIONS,xy):
        ax.annotate(label,(x,y),xytext=(4,4),textcoords="offset points",fontsize=8,weight="bold",
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":.5})
    hypocenter=reference.meta.get("case_configuration",{}).get("hypocenter")
    if hypocenter is not None:
        ax.scatter(hypocenter[0]/1000,hypocenter[1]/1000,marker="*",s=60,color="#cb9c20",edgecolor="black",lw=.5,zorder=6)
    ax.set(xlim=(-17,17),ylim=(-9,9),xlabel="$x$ (km)",ylabel="$y$ (km)",
           title=CASE_LABELS[reference.case]+f"; grid {reference.meta['dx_m']:g} m")
    ax.set_aspect("equal")
    ax.legend(handles=[Line2D([],[],color="black",lw=1,label="Reference"),
                       Line2D([],[],color="#ca352c",ls="--",lw=1,label="Fault PiNet")],
              loc="upper center",ncol=2,fontsize=8,framealpha=.9)
    for j,label in enumerate(STATIONS):
        axis=fig.add_subplot(grid[1+j//3,j%3])
        axis.plot(reference.stations[:,0],reference.stations[:,j+1],color="black",lw=1)
        axis.plot(pinet.stations[:,0],pinet.stations[:,j+1],color="#ca352c",ls="--",lw=.9)
        axis.set(xlim=(0,reference.meta["duration_s"]),xlabel="Time (s)",ylabel="Slip rate (m/s)",
                 title=f"{label}  ({xy[j,0]:g}, {xy[j,1]:g}) km")
        axis.tick_params(labelsize=8);axis.grid(alpha=.15)
    name=f"rupture_{reference.case}"
    fig.savefig(figure_dir/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(preview_dir/f"{name}.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def plot_summary(pairs,figure_dir,preview_dir):
    plt=_setup_plot()
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(len(pairs),2,figsize=(7.4,2.65*len(pairs)),squeeze=False,layout="constrained")
    for row,(reference,pinet) in enumerate(pairs):
        left,right=axes[row]
        for run,color,style,label in [(reference,"black","-","Reference"),(pinet,"#ca352c","--","Fault PiNet")]:
            left.plot(run.diagnostic("time"),run.diagnostic("moment_rate_Nm_s")/1e18,color=color,ls=style,lw=1.1,label=label)
        left.set(xlabel="Time (s)",ylabel="Moment rate ($10^{18}$ Nm/s)",title=CASE_LABELS[reference.case])
        left.grid(alpha=.15)
        if row==0:left.legend(fontsize=8)
        reference_arrival=reference.final[...,1];pinet_arrival=pinet.final[...,1]
        matched=(reference_arrival>=0)&(pinet_arrival>=0)
        delta=np.where(matched,1000*(pinet_arrival-reference_arrival),np.nan)
        in_view=matched&reference.vw
        limit=float(np.max(np.abs(delta[in_view]))) if np.any(in_view) else 1.
        if limit==0:limit=np.finfo(float).eps
        mesh=right.pcolormesh(reference.x/1000,reference.y/1000,delta.T,shading="nearest",cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),rasterized=True)
        disagreement=((reference_arrival>=0)!=(pinet_arrival>=0))&reference.vw
        if np.any(disagreement):
            ix,iy=np.where(disagreement)
            right.scatter(reference.x[ix]/1000,reference.y[iy]/1000,s=6,marker="x",color="#c000c0")
        right.add_patch(Rectangle((-15,-7.5),30,15,fill=False,edgecolor=".35",ls=":",lw=.7))
        right.set(xlim=(-16,16),ylim=(-8.5,8.5),xlabel="$x$ (km)",ylabel="$y$ (km)",
                  title="Arrival difference at 0.5 m/s")
        right.set_aspect("equal")
        fig.colorbar(mesh,ax=right,orientation="horizontal",pad=.12,label="PiNet minus reference (ms)",shrink=.92)
    fig.savefig(figure_dir/"rupture_summary.pdf",bbox_inches="tight")
    fig.savefig(preview_dir/"rupture_summary.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def plot_uniform_refinement(references,figure_dir,preview_dir):
    """Raw reference curves at three meshes; station B's mesh offset is explicit."""
    plt=_setup_plot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    references=sorted(references,key=lambda run:run.nx)
    colors=["#8c6d31","#2878a5","black"]
    styles=[":","--","-"]
    fig=plt.figure(figsize=(7.4,5.0),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[1.6,1])
    ax=fig.add_subplot(grid[0,:])
    for run,color,style in zip(references,colors,styles):
        arrival=np.ma.masked_where(run.final[...,1]<0,run.final[...,1])
        contours=ax.contour(run.x/1000,run.y/1000,arrival.T,levels=np.arange(1,15.1,1),
                            colors=color,linestyles=style,linewidths=.8)
        if run is references[-1]:
            labels=ax.clabel(contours,inline=True,fontsize=6,fmt="%g s")
            for label in labels:
                px,py=label.get_position()
                if abs(px)>=14.5 or abs(py)>=7.:label.remove()
    ax.add_patch(Rectangle((-15,-7.5),30,15,fill=False,edgecolor=".55",ls=":",lw=.7))
    ax.set(xlim=(-17,17),ylim=(-9,9),xlabel="$x$ (km)",ylabel="$y$ (km)",
           title="Uniform reference: coupled space/time refinement")
    ax.set_aspect("equal")
    ax.legend(handles=[Line2D([],[],color=color,ls=style,lw=1,
              label=f"{run.meta['dx_m']:g} m / {run.meta['dt_s']:g} s")
              for run,color,style in zip(references,colors,styles)],loc="upper center",ncol=3,fontsize=8)
    for j,label in enumerate(STATIONS[:3]):
        axis=fig.add_subplot(grid[1,j])
        for run,color,style in zip(references,colors,styles):
            axis.plot(run.stations[:,0],run.stations[:,j+1],color=color,ls=style,lw=.85)
        positions=np.array([run.meta["stations_actual_xy_m"][j] for run in references])/1000
        if np.allclose(positions,positions[0]):
            title=f"{label}  ({positions[0,0]:g}, {positions[0,1]:g}) km"
        else:
            title=f"{label}  $x={positions[-1,0]:g}$ km\n$y={positions[0,1]:g}$ / ${positions[-1,1]:g}$ km"
        axis.set(xlim=(0,references[-1].meta["duration_s"]),xlabel="Time (s)",
                 ylabel="Slip rate (m/s)",title=title)
        axis.tick_params(labelsize=8);axis.grid(alpha=.15)
    fig.savefig(figure_dir/"rupture_uniform_refinement.pdf",bbox_inches="tight")
    fig.savefig(preview_dir/"rupture_uniform_refinement.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def texnum(value,scale=1.):
    if value is None:return "--"
    value*=scale
    if value==0:return "$0$"
    power=int(np.floor(np.log10(abs(value))))
    return f"${value/(10**power):.2f}\\times10^{{{power}}}$"


def tables(learning,refinement,manuscript_dir):
    lines=[r"\begin{table}[htbp]",r"\centering\small",
        r"\caption{Measured learned/reference errors on the same grid. Field velocity, final slip and arrival metrics use $|x|\le15$ km and $|y|\le7.5$ km; moment rate covers the whole fault. Arrival RMSE uses jointly reached points at the 0.5 m/s threshold. All quantities are computed without smoothing. Peak-rate errors, arrival percentiles and crossing masks are retained in the machine-readable results.}",
        r"\label{tab:rupture}",r"\begin{tabular}{llrrrr}\toprule",
        r"Case & $\Delta x$ (m) & $e_V$ & $e_\delta$ & $e_{\dot M}$ & \shortstack{Arrival\\RMSE (ms)} \\\midrule"]
    for row in learning:
        arrival=row["arrival_0p5"]
        values=[CASE_TABLE[row["case"]],f"{row['dx_m']:g}",texnum(row["velocity_relative_L2_VW"]),
                texnum(row["final_slip_relative_L2_VW"]),
                texnum(row["moment_rate_relative_L2_whole_fault"]),texnum(arrival["RMSE_s"],1000)]
        lines.append(" & ".join(values)+r" \\")
    lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
    (manuscript_dir/"rupture_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lookup={(row["case"],row["nx"]):row for row in learning}
    lines=[r"\begin{table}[htbp]",r"\centering\small",
        r"\caption{Measured reference-solver changes under coupled spatial/time-step refinement, sampled at common nodes and saved times. The last two columns repeat the same-grid constitutive-learning error to distinguish it from the discretization change. These differences do not, by themselves, certify convergence.}",
        r"\label{tab:rupture_refinement}",r"\begin{tabular}{llrrrrr}\toprule",
        r"Case & \shortstack{Grid pair\\(m)} & Ref. $e_V$ & Ref. $e_{\dot M}$ & \shortstack{Arrival\\RMSE (ms)} & \shortstack{Learning $e_V$\\coarse} & \shortstack{Learning $e_V$\\fine} \\\midrule"]
    for row in refinement:
        coarse=lookup.get((row["case"],row["nx"]),{}).get("velocity_relative_L2_VW")
        fine=lookup.get((row["case"],row["reference_nx"]),{}).get("velocity_relative_L2_VW")
        values=[CASE_TABLE[row["case"]],f"{row['dx_m']:g}--{row['reference_dx_m']:g}",
                texnum(row["velocity_relative_L2_VW"]),texnum(row["moment_rate_relative_L2_whole_fault"]),
                texnum(row["arrival_0p5"]["RMSE_s"],1000),texnum(coarse),texnum(fine)]
        lines.append(" & ".join(values)+r" \\")
    lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
    (manuscript_dir/"rupture_refinement_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,default=OUTPUT/"runs")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--manuscript",type=Path,default=ROOT/"manuscript")
    parser.add_argument("--figure-nx",type=int,default=360)
    parser.add_argument("--from-compact","--compact-only",action="store_true",help="Reproduce metrics/figures without raw solver binaries")
    parser.add_argument("--require-main",action="store_true",help="Fail unless all three main-resolution pairs are complete")
    parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args()
    compact=args.output/"compact";compact.mkdir(parents=True,exist_ok=True)
    previews=args.output/"figures";previews.mkdir(parents=True,exist_ok=True)
    (args.manuscript/"figures").mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/"input_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"cases":[]}
    runs={};controls={};pending=[]
    def register(run):
        if re.search(r"_dt\d+$",run.name):controls[run.name]=run
        else:runs[(run.case,run.nx,run.mode)]=run
    if args.from_compact:
        for path in sorted(compact.glob("*.npz")):
            run=load_compact(path);register(run)
    else:
        for directory in sorted(args.runs.glob("*_N*_*")):
            if not directory.is_dir():continue
            try:
                _parse_name(directory.name)
            except ValueError:continue
            try:
                cache=compact/f"{directory.name}.npz"
                endfile=directory/"solution_final_fields.bin"
                fingerprint=raw_completion_fingerprint(directory)
                cached=load_compact(cache) if cache.is_file() else None
                if cached is not None and fingerprint is not None and cached.meta.get("raw_completion_fingerprint")==fingerprint:
                    run=cached
                else:
                    run=load_raw(directory,manifest)
                    save_compact(run,cache)
                register(run)
                print(f"Validated/loaded {run.name}",flush=True)
            except IncompleteRun as exc:
                pending.append({"run":directory.name,"reason":str(exc)})
                print(f"Pending {directory.name}: {exc}",flush=True)
    if not runs:
        print("No complete runs accepted; no previous metric/table/figure files changed.")
        return 2 if args.require_main else 0
    learning=[];refinement=[];main_pairs=[]
    for case in CASES:
        grids=sorted({key[1] for key in runs if key[0]==case})
        for nx in grids:
            ref=runs.get((case,nx,"reference"));pinet=runs.get((case,nx,"pinet"))
            if ref is not None and pinet is not None:
                learning.append(compare(pinet,ref))
                if nx==args.figure_nx:main_pairs.append((ref,pinet))
        reference_grids=sorted({key[1] for key in runs if key[0]==case and key[2]=="reference"})
        for coarse_n,fine_n in zip(reference_grids[:-1],reference_grids[1:]):
            refinement.append(compare(runs[(case,coarse_n,"reference")],runs[(case,fine_n,"reference")],refinement=True))
    lookup={(item["case"],item["nx"]):item for item in learning}
    for item in refinement:
        item["coarse_learning_velocity_relative_L2"]=lookup.get((item["case"],item["nx"]),{}).get("velocity_relative_L2_VW")
        item["fine_learning_velocity_relative_L2"]=lookup.get((item["case"],item["reference_nx"]),{}).get("velocity_relative_L2_VW")
    control_comparisons=[]
    for control in controls.values():
        base=runs.get((control.case,control.nx,"reference"))
        if base is not None:
            if control.meta["dt_s"]>=base.meta["dt_s"]:
                raise ValueError("Time-step control must reduce dt relative to baseline")
            item=compare(base,control,refinement=True)
            item["comparison"]="same_grid_reference_time_step_refinement"
            item["reference_run"]=control.name;item["compared_run"]=base.name
            item["interpretation"]="Time-step sensitivity at fixed spatial grid and exactly common saved times"
            indices=matching_indices(base.stations[:,0],control.stations[:,0])
            item["station_velocity_relative_L2"]={s:relative_l2(base.stations[:,j+1],control.stations[indices,j+1])
                                                   for j,s in enumerate(STATIONS)}
            control_comparisons.append(item)
        fine=runs.get((control.case,control.nx*2,"reference"))
        if fine is not None and np.isclose(fine.meta["dt_s"],control.meta["dt_s"],rtol=0,atol=1e-12):
            item=compare(control,fine,refinement=True)
            item["comparison"]="same_time_step_reference_spatial_refinement"
            item["reference_run"]=fine.name;item["compared_run"]=control.name
            item["interpretation"]="Spatial-grid sensitivity at fixed dt and exactly common spatial nodes/saved times"
            control_comparisons.append(item)
    all_runs=[*runs.values(),*controls.values()]
    metrics={"status":"main_pairs_complete" if len(main_pairs)==3 else "partial",
        "scope":"Actual completed spatial SBI solver comparisons; source-parameter families with newly generated specified realizations",
        "figure_nx":args.figure_nx,"main_pairs_complete":len(main_pairs),"completed_runs":len(all_runs),
        "pending_runs":pending,"learning_comparisons":learning,"reference_grid_refinement":refinement,
        "reference_controls":control_comparisons,
        "run_audits":{run.name:run_audit(run) for run in all_runs},
        "storage":{"compact":"NPZ with float32 physical slip-rate snapshots, float64 final fields/full-step stations/diagnostics, times and metadata",
                   "arrival_provenance":"Threshold crossing tracked at every integration step with linear crossing-time interpolation",
                   "snapshot_interval_requested_s":.1,"contour_smoothing":"none"},
        "error_definitions":{"relative_L2":"sqrt(sum((prediction-reference)^2)/sum(reference^2))",
            "arrival_RMSE":"sqrt(mean((t_prediction-t_reference)^2)) on jointly reached VW points",
            "arrival_P95":"95th percentile of absolute arrival errors on jointly reached VW points",
            "mask_disagreement":"Count and fraction of VW nodes where only one run reaches the threshold",
            "reference_refinement":"Fine reference sampled on exact coarse nodes/times; time step also changes"}}
    (args.output/"rupture_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    tables(learning,refinement,args.manuscript)
    if not args.no_figures:
        for reference,pinet in main_pairs:plot_case(reference,pinet,args.manuscript/"figures",previews)
        if main_pairs:plot_summary(main_pairs,args.manuscript/"figures",previews)
        uniform_grids=[runs[("uniform",nx,"reference")] for nx in (180,360,720)
                       if ("uniform",nx,"reference") in runs]
        if len(uniform_grids)==3:plot_uniform_refinement(uniform_grids,args.manuscript/"figures",previews)
    print(json.dumps({"status":metrics["status"],"completed_runs":len(all_runs),"learning_comparisons":learning,
                      "reference_grid_refinement":refinement,"reference_controls":control_comparisons},indent=2),flush=True)
    if args.require_main and len(main_pairs)!=3:return 2
    return 0


if __name__=="__main__":raise SystemExit(main())
