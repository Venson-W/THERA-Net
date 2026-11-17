"""Reproduce synthetic constitutive learning and reduced slider verification.

Requires NumPy, SciPy, Matplotlib. This script does not reproduce the source
paper's FNO or SBI datasets and does not measure a rupture-solver speedup.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent/".mplconfig"))

# Optional project-local scientific dependencies, installed for this execution.
LOCAL_DEPS = Path(__file__).resolve().parents[2]/"research_notes"/"python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fault_pinet import (FaultPiNet, MLP, LOWER, UPPER, PARAMETER_NAMES, VREF,
    traction, tangent, potential, aging_update, true_parameters,
    generate_dataset, spring_slider, SLIDER_COLUMNS)
from source_cases import SOURCE_PARAMETERS, source_case


def baseline_inputs(data):
    return np.column_stack([data["descriptor"],
        (np.log10(np.abs(data["velocity"]))+5)/3,
        (np.log10(data["theta"])-3.5)/4.5, np.sign(data["velocity"])])


def fit_pinet(train, validation, seed, maxiter):
    model = FaultPiNet(seed)
    start = time.perf_counter()
    history = []
    best = [np.inf, model.network.pack().copy(), 0]
    steps = [0]
    def callback(vector):
        steps[0] += 1
        if steps[0] % 10 == 0 or steps[0] == 1:
            loss, _ = model.loss_and_gradient(vector, train)
            val, _ = model.loss_and_gradient(vector, validation)
            history.append([steps[0], loss, val])
            if val < best[0]:
                best[:] = [val, vector.copy(), steps[0]]
    result = minimize(lambda w: model.loss_and_gradient(w, train), model.network.pack(),
                      jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": maxiter, "maxls": 40, "ftol": 1e-13, "gtol": 1e-8,
                               "maxcor": 30})
    callback(result.x)
    final_val, _ = model.loss_and_gradient(result.x, validation)
    if final_val < best[0]:
        best[:] = [final_val, result.x.copy(), int(result.nit)]
    model.network.unpack(best[1])
    metadata = {"seed": seed, "seconds": time.perf_counter()-start,
                "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                "iterations": int(result.nit), "selected_iteration": best[2],
                "validation_loss": best[0], "parameter_count": len(result.x)}
    return model, np.asarray(history), metadata


def fit_baseline(train, validation, seed, maxiter):
    network = MLP([4, 16, 16, 1], seed)
    xtrain, xval = baseline_inputs(train), baseline_inputs(validation)
    def loss_grad(vector):
        network.unpack(vector)
        result, cache = network.forward(xtrain, cache=True)
        error = (result[:, 0]-train["mu"])/0.1
        return float(np.mean(error**2)), network.backward(cache, (2*error/len(error)/0.1)[:, None])
    best = [np.inf, network.pack().copy()]
    def callback(vector):
        network.unpack(vector)
        loss = np.mean(((network.forward(xval)[:, 0]-validation["mu"])/0.1)**2)
        if loss < best[0]:
            best[:] = [float(loss), vector.copy()]
    result = minimize(loss_grad, network.pack(), jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": maxiter, "maxls": 40, "ftol": 1e-13, "gtol": 1e-8,
                               "maxcor": 30})
    callback(result.x)
    network.unpack(best[1])
    return network, {"seed": seed, "optimizer_success": bool(result.success),
                      "optimizer_message": str(result.message), "iterations": int(result.nit),
                      "validation_mu_scaled_mse": best[0], "parameter_count": len(result.x)}


def relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))


def evaluate(model, data):
    params = model.parameters(data["descriptor"])
    mu = traction(data["velocity"], data["theta"], 1.0, params)
    theta = aging_update(data["theta"], data["velocity"], data["dt"], params[:, 2])
    actual = true_parameters(data["descriptor"])
    errors = {"traction_relative_L2": relative_l2(mu, data["mu"]),
        "traction_RMSE_MPa_at_120MPa": float(np.sqrt(np.mean((120*(mu-data["mu"]))**2))),
        "log_state_step_RMSE": float(np.sqrt(np.mean(np.log(theta/data["theta_next"])**2))),
        "parameter_relative_L2": {name:relative_l2(params[:, i], actual[:, i])
                                   for i, name in enumerate(PARAMETER_NAMES)}}
    return errors


def history_test(model, descriptor=0.37, dt=0.02):
    time_grid = np.arange(0, 30+dt/2, dt)
    velocity = np.select([time_grid < 5, time_grid < 10, time_grid < 15,
                          time_grid < 20, time_grid < 25],
                         [1e-6, 1e-3, 0., -1e-3, -1e-5], default=1e-2)
    real_params = true_parameters(descriptor)
    learned = model.parameters([descriptor])[0]
    state_real, state_pred = 2e4, 2e4
    rows = []
    for t, v in zip(time_grid, velocity):
        tau_real = float(traction(v, state_real, 120e6, real_params))
        tau_pred = float(traction(v, state_pred, 120e6, learned))
        rows.append([t, v, state_real, state_pred, tau_real, tau_pred])
        state_real = float(aging_update(state_real, v, dt, real_params[2]))
        state_pred = float(aging_update(state_pred, v, dt, learned[2]))
    return np.asarray(rows)


def figure_save(fig, output, name):
    fig.savefig(output/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output/f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plots(output, models, datasets, baseline, histories, slider_true, slider_pred, history, case):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    blue, orange = "#186b9e", "#c75328"
    x = np.linspace(-1, 1, 250)
    truth = true_parameters(x)
    pred = np.stack([m.parameters(x) for m in models])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    labels = ["Direct effect a", "Evolution effect b", "Slip distance L (m)", "Reference friction f*"]
    for i, ax in enumerate(axes.flat):
        ax.plot(x, truth[:, i], color="black", lw=1.6, label="Synthetic truth")
        ax.plot(x, pred.mean(axis=0)[:, i], color=blue, ls="--", label="Fault PiNet mean")
        ax.fill_between(x, pred.min(axis=0)[:, i], pred.max(axis=0)[:, i], color=blue, alpha=.18,
                        label="Range across three seeds")
        ax.set(xlabel="Fixed material descriptor", ylabel=labels[i])
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    figure_save(fig, output, "learned_parameters")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for name, data, color in [("Interpolation",datasets["test"],blue),
                             ("Low-speed holdout",datasets["low_speed"],orange),
                             ("High-speed holdout",datasets["high_speed"],"#6f4790")]:
        p = models[0].parameters(data["descriptor"])
        actual, predicted = 120*data["mu"], 120*traction(data["velocity"],data["theta"],1.,p)
        axes[0].scatter(actual, predicted, s=4, alpha=.2, color=color, label=name)
        axes[1].scatter(np.log10(np.abs(data["velocity"])), np.abs(actual-predicted),
                        s=4, alpha=.25, color=color)
    axes[0].plot([-110,110],[-110,110],color="black",lw=.7)
    axes[0].set(xlabel="Synthetic traction (MPa)",ylabel="Fault PiNet traction (MPa)")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="log10 |V| (m/s)",ylabel="Absolute traction error (MPa)",yscale="log")
    axes[1].axvspan(-8,-2,color="gray",alpha=.1)
    for ax in axes: ax.grid(alpha=.15)
    figure_save(fig, output, "constitutive_predictions")

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.7), constrained_layout=True)
    for column, label, ax in zip([1,2,3],["Slip rate (m/s)","Slip (m)","State theta (s)"],axes):
        ax.plot(slider_true[:,0],slider_true[:,column],color="black",lw=1.4,label="Synthetic truth")
        ax.plot(slider_pred[:,0],slider_pred[:,column],color=blue,ls="--",lw=1.3,label="Fault PiNet")
        ax.set(xlabel="Time (s)",ylabel=label)
        if column==3: ax.set_yscale("log")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7)
    figure_save(fig, output, "slider_transient")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    arr=slider_pred
    axes[0].plot(arr[:,0],np.cumsum(arr[:,6])/1e6,label="External work",color="black")
    axes[0].plot(arr[:,0],(arr[:,5]-arr[0,5])/1e6,label="Stored + kinetic energy",color=blue)
    axes[0].plot(arr[:,0],np.cumsum(arr[:,7])/1e6,label="Physical dissipation",color=orange)
    axes[0].plot(arr[:,0],np.cumsum(arr[:,8])/1e6,label="Numerical dissipation",color="#6f4790")
    axes[0].set(xlabel="Time (s)",ylabel="Energy per unit area (MJ/m²)")
    axes[0].legend(fontsize=7)
    axes[1].plot(arr[1:,0],np.abs(arr[1:,9])/max(np.max(np.abs(arr[:,6])),1.),color=blue)
    axes[1].set(xlabel="Time (s)",ylabel="Absolute step balance / peak step work",yscale="log")
    for ax in axes: ax.grid(alpha=.15)
    figure_save(fig, output, "discrete_energy_balance")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True, constrained_layout=True)
    axes[0].plot(history[:,0],history[:,4]/1e6,color="black",label="Synthetic truth")
    axes[0].plot(history[:,0],history[:,5]/1e6,color=blue,ls="--",label="Fault PiNet")
    axes[0].set(ylabel="Signed traction (MPa)")
    axes[0].legend(fontsize=8)
    axes[1].plot(history[:,0],history[:,2],color="black")
    axes[1].plot(history[:,0],history[:,3],color=blue,ls="--")
    axes[1].set(xlabel="Time (s)",ylabel="State theta (s)",yscale="log")
    for ax in axes:
        ax.axvspan(10,15,color="gray",alpha=.15)
        ax.grid(alpha=.15)
    figure_save(fig, output, "hold_and_reversal")

    fig,axes=plt.subplots(1,3,figsize=(9.4,2.9),constrained_layout=True)
    for arr,label,ax in zip([case["initial_stress_Pa"]/1e6,case["parameters_a_b_L_f"][...,0],
                              np.log10(case["initial_theta_s"])],
                            ["Initial shear stress (MPa)","Direct effect a","log10 initial theta (s)"],axes):
        im=ax.imshow(arr.T,origin="lower",extent=[-36,36,-18,18],aspect="equal",cmap="viridis")
        ax.set(xlabel="x (km)",ylabel="y (km)",title=label)
        fig.colorbar(im,ax=ax,orientation="horizontal",pad=.23,shrink=.85)
    figure_save(fig, output, "source_case_inputs")

    fig,ax=plt.subplots(figsize=(5.4,3.0),constrained_layout=True)
    for seed,h in enumerate(histories):
        ax.semilogy(h[:,0],h[:,1],label=f"Train seed {seed}")
        ax.semilogy(h[:,0],h[:,2],ls="--",label=f"Validation seed {seed}")
    ax.set(xlabel="L-BFGS iteration",ylabel="Dimensionless joint loss")
    ax.legend(ncol=2,fontsize=7)
    ax.grid(alpha=.15)
    figure_save(fig,output,"training_history")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--maxiter",type=int,default=1200)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    # Whole material descriptors are held out; points from a descriptor never
    # appear in different train/validation/test groups. Boundary groups train.
    descriptors=np.linspace(-1,1,61)
    ids=np.arange(1,60)
    np.random.default_rng(20261112).shuffle(ids)
    test_ids=ids[:12]; validation_ids=ids[12:22]
    train_ids=np.concatenate([[0,60],ids[22:]])
    datasets={
        "train":generate_dataset(descriptors[train_ids],32,101),
        "validation":generate_dataset(descriptors[validation_ids],32,102),
        "test":generate_dataset(descriptors[test_ids],64,103),
        "low_speed":generate_dataset(descriptors[test_ids],64,104,(-12,-8)),
        "high_speed":generate_dataset(descriptors[test_ids],64,105,(-2,0))}
    for name,data in datasets.items(): np.savez(output/f"data_{name}.npz",**data)
    config={"scope":"Synthetic constitutive identification and single-DOF forced slider; no SBI/FNO replication",
        "pressure_for_reported_traction_Pa":120e6,"velocity_training_log10_range":[-8,-2],
        "theta_training_log10_range":[-1,8],"signed_training_velocities":True,
        "samples":{name:len(data["mu"]) for name,data in datasets.items()},
        "split_seed":20261112,"training_descriptor_indices":train_ids.tolist(),
        "validation_descriptor_indices":validation_ids.tolist(),"test_descriptor_indices":test_ids.tolist(),
        "all_descriptors":descriptors.tolist(),"network_layers":[1,12,12,4],
        "baseline_layers":[4,16,16,1],"parameter_lower":LOWER.tolist(),"parameter_upper":UPPER.tolist(),
        "optimizer":"L-BFGS-B","maxiter":args.maxiter,"seeds":args.seeds,
        "noise":"none; exact synthetic teacher", "label_scaling":{"mu":0.1,"log_theta_ratio":1.0},
        "training_supervision":"traction and exact frozen-speed state increments; no parameter labels",
        "baseline_scope":"direct traction MLP given current theta; architecture reference, not equal-information ablation",
        "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,
        "platform":platform.platform(),"source_parameters":SOURCE_PARAMETERS}
    (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    models=[]; histories=[]; metrics={"runs":[]}
    for seed in args.seeds:
        print(f"Training seed {seed}",flush=True)
        model,history,metadata=fit_pinet(datasets["train"],datasets["validation"],seed,args.maxiter)
        models.append(model); histories.append(history)
        model.save(output/f"fault_pinet_seed_{seed}.npz")
        np.savetxt(output/f"loss_seed_{seed}.csv",history,delimiter=",",header="iteration,train_loss,validation_loss",comments="")
        metadata["metrics"]={name:evaluate(model,data) for name,data in datasets.items()}
        metrics["runs"].append(metadata)
        print(json.dumps(metadata),flush=True)
    print("Training direct traction reference",flush=True)
    baseline,bmeta=fit_baseline(datasets["train"],datasets["validation"],0,args.maxiter)
    np.savez(output/"baseline_seed_0.npz",vector=baseline.pack(),layers=baseline.layers)
    bmeta["metrics"]={name:{"traction_relative_L2":relative_l2(baseline.forward(baseline_inputs(data))[:,0],data["mu"]),
        "traction_RMSE_MPa_at_120MPa":float(np.sqrt(np.mean((120*(baseline.forward(baseline_inputs(data))[:,0]-data["mu"]))**2)))}
        for name,data in datasets.items()}
    metrics["baseline"]=bmeta
    rng=np.random.default_rng(667)
    n=20000
    x=rng.uniform(-1.25,1.25,n)
    v=10**rng.uniform(-12,1,n)*rng.choice([-1.,1.],n)
    theta=10**rng.uniform(-3,12,n)
    params=models[0].parameters(x)
    dissipation=traction(v,theta,120e6,params)*v
    phi=potential(v,theta,120e6,params)
    tang=tangent(v,theta,120e6,params)
    metrics["physics_probes"]={"count":n,"descriptor_range":[-1.25,1.25],
        "abs_velocity_m_s_range":[1e-12,10.],"theta_s_range":[1e-3,1e12],
        "minimum_dissipation_W_m2":float(dissipation.min()),"minimum_potential_W_m2":float(phi.min()),
        "minimum_tangent_Pa_s_m":float(tang.min()),"negative_dissipation_count":int(np.sum(dissipation<0)),
        "negative_potential_count":int(np.sum(phi<0)),"nonpositive_tangent_count":int(np.sum(tang<=0))}
    history=history_test(models[0])
    np.savetxt(output/"hold_reversal.csv",history,delimiter=",",comments="",
               header="time_s,velocity_m_s,theta_true_s,theta_predicted_s,traction_true_Pa,traction_predicted_Pa")
    metrics["hold_reversal"]={"descriptor":0.37,"traction_relative_L2":relative_l2(history[:,5],history[:,4]),
                              "state_relative_L2":relative_l2(history[:,3],history[:,2])}
    print("Running implicit slider and time-step refinement",flush=True)
    descriptor=-0.73
    slider_true=spring_slider(true_parameters(descriptor),dt=.005)
    slider_pred=spring_slider(models[0].parameters([descriptor])[0],dt=.005)
    for name,arr in [("reference",slider_true),("pinet",slider_pred)]:
        np.savetxt(output/f"slider_{name}.csv",arr,delimiter=",",header=SLIDER_COLUMNS,comments="")
    metrics["slider"]={"descriptor":descriptor,"time_step_s":.005,"end_time_s":5.,
        "mass_kg_m2":267000.,"stiffness_Pa_m":1e8,"radiation_Pa_s_m":2670*3464/2,
        "velocity_relative_L2":relative_l2(slider_pred[:,1],slider_true[:,1]),
        "slip_relative_L2":relative_l2(slider_pred[:,2],slider_true[:,2]),
        "maximum_step_energy_residual_J_m2":float(np.max(np.abs(slider_pred[:,9]))),
        "maximum_step_residual_over_peak_step_work":float(np.max(np.abs(slider_pred[:,9]))/np.max(np.abs(slider_pred[:,6]))),
        "total_physical_dissipation_J_m2":float(slider_pred[:,7].sum()),
        "total_frictional_dissipation_J_m2":float(.005*np.sum(slider_pred[1:,4]*slider_pred[1:,1])),
        "total_radiation_dissipation_J_m2":float(.005*(2670*3464/2)*np.sum(slider_pred[1:,1]**2)),
        "total_numerical_dissipation_J_m2":float(slider_pred[:,8].sum()),
        "final_external_work_J_m2":float(slider_pred[:,6].sum()),
        "prestress_work_J_m2":float(.005*75e6*np.sum(slider_pred[1:,1])),
        "nucleation_work_J_m2":float(slider_pred[:,6].sum()-.005*75e6*np.sum(slider_pred[1:,1])),
        "final_energy_J_m2":float(slider_pred[-1,5])}
    reference_fine=spring_slider(true_parameters(descriptor),dt=.00125)
    refinement=[]
    for dt in [.02,.01,.005,.0025]:
        coarse=slider_true if dt==.005 else spring_slider(true_parameters(descriptor),dt=dt)
        fine_on_grid=np.interp(coarse[:,0],reference_fine[:,0],reference_fine[:,1])
        refinement.append({"dt_s":dt,"velocity_relative_L2_vs_dt_0.00125":relative_l2(coarse[:,1],fine_on_grid)})
    metrics["time_refinement"]=refinement
    case=source_case((128,64),dimension=1.3,seed=867)
    np.savez(output/"source_case_2d_plane_inputs.npz",**case)
    np.savez(output/"source_case_1d_line_inputs.npz",**source_case((256,),dimension=1.5,seed=868))
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    plots(output,models,datasets,baseline,histories,slider_true,slider_pred,history,case)
    print(json.dumps(metrics,indent=2),flush=True)
    print(f"Saved reproducible results to {output}",flush=True)


if __name__=="__main__": main()
