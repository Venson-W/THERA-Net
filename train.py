"""Train and validate THERA-Net on a synthetic material family."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

import scipy
from scipy.optimize import minimize
from model import (THERANet, traction, aging_update, initial_state,
                         LOWER, UPPER, PARAMETER_NAMES)
from material import (DEFAULT_OUTPUT, material_descriptor, source_truth,
                                 source_family_parameters, export_source_field)


def generate(a_groups, samples_per_group, seed):
    rng = np.random.default_rng(seed)
    a = np.repeat(np.asarray(a_groups), samples_per_group)
    v = 10**rng.uniform(-12., 1., len(a))*rng.choice([-1.,1.], len(a))
    theta = 10**rng.uniform(-3., 24., len(a))
    dt = rng.uniform(0.05, 2.0, len(a))*0.020/np.abs(v)
    params = source_truth(a)
    return {"descriptor": material_descriptor(a), "a_source": a, "velocity": v,
            "theta": theta, "dt": dt, "mu": traction(v,theta,1.,params),
            "theta_next": aging_update(theta,v,dt,params[:,2])}


def relative(pred, true):
    return float(np.linalg.norm(pred-true)/np.linalg.norm(true))


def evaluate(model, data):
    params = source_family_parameters(model, data["a_source"])
    actual = source_truth(data["a_source"])
    mu = traction(data["velocity"], data["theta"], 1., params)
    state = aging_update(data["theta"],data["velocity"],data["dt"],params[:,2])
    return {"traction_relative_L2": relative(mu,data["mu"]),
            "traction_RMSE_MPa_at_p120MPa": float(120*np.sqrt(np.mean((mu-data["mu"])**2))),
            "traction_max_abs_error_MPa_at_p120MPa": float(120*np.max(np.abs(mu-data["mu"]))),
            "state_log_ratio_RMSE":float(np.sqrt(np.mean(np.log(state/data["theta_next"])**2))),
            "parameter_relative_L2": {name:relative(params[:,j],actual[:,j]) for j,name in enumerate(PARAMETER_NAMES)},
            "parameter_max_abs_error": {name:float(np.max(np.abs(params[:,j]-actual[:,j])))
                                         for j,name in enumerate(PARAMETER_NAMES)}}


def fit(train, validation, seed, maxiter):
    model = THERANet(seed)
    best = [np.inf, None, 0]
    history = []
    count = [0]
    start = time.perf_counter()
    def monitor(weights):
        count[0] += 1
        if count[0] % 10 == 0 or count[0] == 1:
            train_loss,_ = model.loss_and_gradient(weights,train)
            val_loss,_ = model.loss_and_gradient(weights,validation)
            history.append([count[0],train_loss,val_loss])
            if val_loss < best[0]: best[:] = [val_loss,weights.copy(),count[0]]
    result = minimize(lambda w:model.loss_and_gradient(w,train),model.network.pack(),
                      jac=True,method="L-BFGS-B",callback=monitor,
                      options={"maxiter":maxiter,"maxls":40,"maxcor":40,
                               "ftol":1e-15,"gtol":1e-9})
    val_loss,_ = model.loss_and_gradient(result.x,validation)
    if val_loss < best[0]: best[:] = [val_loss,result.x.copy(),int(result.nit)]
    model.network.unpack(best[1])
    return model,np.array(history),{"seed":seed,"max_iterations":maxiter,
        "actual_iterations":int(result.nit),"selected_iteration":best[2],
        "selected_validation_loss":float(best[0]),"optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),"elapsed_training_seconds":time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter",type=int,default=5000)
    parser.add_argument("--seeds",type=int,nargs="+",default=[10,11,12])
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    a_groups = np.linspace(.006,.018,97)
    source_endpoint_ids = np.array([4,8,16,80,88,92])
    remaining = np.setdiff1d(np.arange(1,96),source_endpoint_ids)
    np.random.default_rng(20261113).shuffle(remaining)
    test_ids = np.concatenate([source_endpoint_ids,remaining[:14]])
    validation_ids = remaining[14:30]
    train_ids = np.concatenate([[0,96],remaining[30:]])
    data = {"train":generate(a_groups[train_ids],48,301),
            "validation":generate(a_groups[validation_ids],48,302),
            "test":generate(a_groups[test_ids],128,303)}
    for name,values in data.items(): np.savez(output/f"data_{name}.npz",**values)
    config = {"purpose":"Neural source-family material closure for spatial solver coupling; not a learned rupture operator",
        "a_training_range":[.006,.018],"descriptor":"z=(a_source-0.012)/0.006",
        "teacher_constants":{"b":.012,"L_m":.020,"f_ref":.600},
        "all_material_groups_a":a_groups.tolist(),"train_group_indices":train_ids.tolist(),
        "validation_group_indices":validation_ids.tolist(),"test_group_indices":test_ids.tolist(),
        "source_case_endpoint_groups_held_out_a":a_groups[source_endpoint_ids].tolist(),
        "split_seed":20261113,"sample_seeds":{"train":301,"validation":302,"test":303},
        "samples":{name:len(d["mu"]) for name,d in data.items()},
        "signed_velocity":True,"log10_abs_velocity_m_s_range":[-12.,1.],
        "log10_theta_s_range":[-3.,24.],"dt_s":"u*0.020/abs(V), u uniform[0.05,2]",
        "labels":"Noiseless regularized RSF traction and exact frozen-speed aging state steps; no parameter labels",
        "layers":[1,12,12,4],"trainable_parameters":232,"parameter_lower":LOWER.tolist(),
        "parameter_upper":UPPER.tolist(),"loss":"mean[((mu_pred-mu)/0.1)^2+log(theta_next_pred/theta_next)^2]",
        "selection":"Lowest whole-material-group validation loss across checkpoints and seeds; test excluded",
        "seeds":args.seeds,"max_iterations":args.maxiter,"numpy_version":np.__version__,
        "scipy_version":scipy.__version__,"python":sys.version,"platform":platform.platform()}
    (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    models=[]; runs=[]
    for seed in args.seeds:
        print(f"Training source-family seed {seed}",flush=True)
        model,history,metadata = fit(data["train"],data["validation"],seed,args.maxiter)
        metadata["metrics"] = {name:evaluate(model,values) for name,values in data.items()}
        models.append(model); runs.append(metadata)
        model.save(output/f"model_seed_{seed}.npz")
        np.savetxt(output/f"loss_seed_{seed}.csv",history,delimiter=",",comments="",
                   header="iteration,train_joint_loss,validation_joint_loss")
        print(json.dumps(metadata),flush=True)
    selected_index = int(np.argmin([run["selected_validation_loss"] for run in runs]))
    selected = models[selected_index]
    selected_seed = runs[selected_index]["seed"]
    selected.save(output/"model.npz")
    dense_a = np.linspace(.006,.018,4001)
    pred = source_family_parameters(selected,dense_a)
    truth = source_truth(dense_a)
    np.savetxt(output/"material_family_dense.csv",np.column_stack([dense_a,pred]),delimiter=",",comments="",
               header="source_a,predicted_a,predicted_b,predicted_L_m,predicted_f_ref")
    dense_metrics = {"grid_size":len(dense_a),"parameter_relative_L2":{},"parameter_max_abs_error":{}}
    for j,name in enumerate(PARAMETER_NAMES):
        dense_metrics["parameter_relative_L2"][name] = relative(pred[:,j],truth[:,j])
        dense_metrics["parameter_max_abs_error"][name] = float(np.max(np.abs(pred[:,j]-truth[:,j])))
    # Audit the three actual central/outer source coefficients, all held out from
    # both training and validation; the dense transition is a separate interpolation.
    endpoints = np.array([.007,.017,.008,.016,.0065,.0175])
    endpoints_pred = source_family_parameters(selected,endpoints)
    np.savetxt(output/"source_case_coefficients.csv",np.column_stack([endpoints,endpoints_pred]),delimiter=",",comments="",
               header="source_a,predicted_a,predicted_b,predicted_L_m,predicted_f_ref")
    # Stress-state initialization sensitivity, because source theta is inferred
    # from tau0 and Vini and exponentiates small coefficient errors.
    stress,a_grid = np.meshgrid(np.linspace(45e6,105e6,121),dense_a[::40],indexing="ij")
    true_theta = initial_state(stress,1e-12,120e6,source_truth(a_grid))
    pred_theta = initial_state(stress,1e-12,120e6,source_family_parameters(selected,a_grid))
    initial_metrics = {"shear_stress_Pa_range":[45e6,105e6],"V_ini_m_s":1e-12,"p_Pa":120e6,
                      "grid_size":int(stress.size),"true_theta_s_min":float(true_theta.min()),
                      "true_theta_s_max":float(true_theta.max()),
                      "theta_relative_L2":relative(pred_theta,true_theta),
                      "maximum_abs_log_theta_ratio":float(np.max(np.abs(np.log(pred_theta/true_theta))))}
    # A 2x3 deterministic exporter smoke check is saved for the solver interface.
    example_field = endpoints.reshape(2,3)
    exported = export_source_field(output/"example_coefficients.npy",example_field,model=selected)
    reloaded = np.load(output/"example_coefficients.npy")
    if not np.array_equal(exported,reloaded): raise AssertionError("field export/load mismatch")
    # Reloaded checkpoint must reproduce all held-out traction metrics exactly.
    loaded = THERANet.load(output/"model.npz")
    checkpoint_error = float(np.max(np.abs(source_family_parameters(loaded,dense_a)-pred)))
    if checkpoint_error != 0: raise AssertionError("checkpoint round trip changed neural output")
    digest = hashlib.sha256((output/"model.npz").read_bytes()).hexdigest()
    metrics = {"runs":runs,"selected_seed":selected_seed,
               "selected_model_sha256":digest,"selected_test_metrics":evaluate(selected,data["test"]),
               "dense_material_audit":dense_metrics,"source_initial_state_audit":initial_metrics,
               "saved_checkpoint_max_parameter_difference":checkpoint_error,
               "exported_field_shape":list(exported.shape),"exported_field_dtype":str(exported.dtype),
               "source_case_endpoint_coefficients":endpoints_pred.tolist()}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2),flush=True)
    print(f"Selected seed {selected_seed}, saved {output/'model.npz'}",flush=True)


if __name__ == "__main__": main()
