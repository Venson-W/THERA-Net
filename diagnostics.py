"""Rupture arrivals, field errors, moment rate, and interface-work diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np

SHEAR_MODULUS = 2670.*3464.**2


def relative_l2(predicted, reference, mask=None):
    predicted, reference = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape or not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("matching finite arrays required")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim < 1 or mask.shape != reference.shape[-mask.ndim:] or not mask.any():
            raise ValueError("nonempty spatial mask must match the trailing axes")
        predicted, reference = predicted[..., mask], reference[..., mask]
    error, norm = np.linalg.norm(predicted-reference), np.linalg.norm(reference)
    if norm == 0:
        if error == 0:
            return 0.
        raise ValueError("relative error is undefined for a zero reference")
    return float(error/norm)


def matching_indices(values, grid):
    """Match common samples without interpolation or dynamic alignment."""
    values, grid = np.asarray(values), np.asarray(grid)
    if (values.ndim != 1 or grid.ndim != 1 or not len(values) or not len(grid)
            or not np.isfinite(values).all() or not np.isfinite(grid).all()
            or np.any(np.diff(values) <= 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError("finite increasing sample vectors required")
    indices = np.searchsorted(grid, values)
    right = np.clip(indices, 0, len(grid)-1)
    left = np.clip(indices-1, 0, len(grid)-1)
    indices = np.where(abs(grid[left]-values) < abs(grid[right]-values), left, right)
    if not np.allclose(grid[indices], values, rtol=0, atol=1e-10):
        raise ValueError("samples are not shared")
    return indices


def arrival_times(velocity, time, threshold=.5):
    """First |V| crossing at the supplied sampling rate; -1 denotes unreached."""
    velocity, time = np.asarray(velocity), np.asarray(time)
    if (velocity.ndim < 2 or len(time) != len(velocity) or not len(time)
            or np.any(np.diff(time) <= 0) or not np.isfinite(time).all()
            or not np.isfinite(velocity).all() or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("finite histories, increasing times and positive threshold required")
    speed = abs(velocity)
    result = np.full(velocity.shape[1:], -1.)
    result[speed[0] >= threshold] = time[0]
    for n in range(1, len(time)):
        crossed = (result < 0) & (speed[n] >= threshold)
        fraction = (threshold-speed[n-1][crossed])/(speed[n][crossed]-speed[n-1][crossed])
        result[crossed] = time[n-1]+np.clip(fraction, 0., 1.)*(time[n]-time[n-1])
    return result


def arrival_metrics(predicted, reference, mask):
    predicted, reference, mask = np.asarray(predicted), np.asarray(reference), np.asarray(mask, dtype=bool)
    if predicted.shape != reference.shape or mask.shape != reference.shape or not mask.any():
        raise ValueError("matching arrivals and nonempty mask required")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("finite arrivals required; use -1 for unreached nodes")
    pred_reached, ref_reached = (predicted >= 0) & mask, (reference >= 0) & mask
    both = pred_reached & ref_reached
    return dict(matched_reached_points=int(both.sum()),
        RMSE_s=float(np.sqrt(np.mean((predicted[both]-reference[both])**2))) if both.any() else None,
        mask_disagreement_points=int(np.sum(pred_reached != ref_reached)),
        false_reached_points=int(np.sum(pred_reached & ~ref_reached)),
        missed_reached_points=int(np.sum(ref_reached & ~pred_reached)))


def moment_rate(velocity, cell_area, shear_modulus=SHEAR_MODULUS):
    """Signed whole-fault moment rate; spatial axes are the final two axes."""
    velocity = np.asarray(velocity)
    if velocity.ndim < 2 or not np.isfinite(velocity).all() or cell_area <= 0 or shear_modulus <= 0:
        raise ValueError("finite grid and positive area/modulus required")
    return shear_modulus*cell_area*np.sum(velocity, axis=(-2, -1))


def integrate_power(power, time, initial_time=0.):
    """Right-endpoint integral; signed values are retained without clipping."""
    power, time = np.asarray(power), np.asarray(time)
    increments = np.diff(np.r_[initial_time, time])
    if (time.ndim != 1 or not len(time) or power.shape[0] != len(time)
            or not np.isfinite(power).all() or not np.isfinite(time).all()
            or not np.isfinite(initial_time) or np.any(increments <= 0)):
        raise ValueError("finite power and increasing times after initial_time required")
    return np.tensordot(increments, power, axes=(0, 0))


def region_masks(x, y, barrier_halfwidth):
    if not np.isfinite(barrier_halfwidth) or not 0 <= barrier_halfwidth < 15000:
        raise ValueError("barrier halfwidth must be in [0,15000) m")
    x, y = np.asarray(x), np.asarray(y)
    active = (abs(x[:, None]) <= 15000) & (abs(y[None, :]) <= 7500)
    return dict(active=active, left=active & (x[:, None] < -barrier_halfwidth),
                barrier=active & (abs(x[:, None]) <= barrier_halfwidth),
                receiver=active & (x[:, None] > barrier_halfwidth))


def receiver_fraction(peak_speed, mask, threshold=.5):
    peak_speed, mask = np.asarray(peak_speed), np.asarray(mask, dtype=bool)
    if peak_speed.shape != mask.shape or not mask.any() or not np.isfinite(peak_speed).all():
        raise ValueError("finite peaks and nonempty receiver mask required")
    return float(np.mean(peak_speed[mask] >= threshold))


def receiver_status(fraction):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0,1]")
    return "activated" if fraction >= .5 else ("arrested" if fraction <= .01 else "partial")


def load_run(folder, prefix="solution", snapshots=False):
    """Read adapter output and reject incomplete time series or invalid states."""
    folder = Path(folder)
    meta = json.loads((folder/f"{prefix}_metadata.json").read_text())
    nx, ny, dt = meta["nx"], meta["nz"], meta["dt_s"]
    final = np.fromfile(folder/f"{prefix}_final_fields.bin", dtype="<f8").reshape(nx, ny, 5)
    work = np.fromfile(folder/f"{prefix}_friction_work.bin", dtype="<f8").reshape(nx, ny)
    diag = np.genfromtxt(folder/f"{prefix}_diagnostics.csv", delimiter=",", names=True, ndmin=1)
    regional = np.genfromtxt(folder/f"{prefix}_regional.csv", delimiter=",", names=True, ndmin=1)
    stations = np.genfromtxt(folder/f"{prefix}_stations.csv", delimiter=",", names=True, ndmin=1)
    expected_time = np.arange(1, round(meta["duration_s"]/dt)+1)*dt
    for table in (diag, regional, stations):
        if len(table) != len(expected_time) or not np.allclose(table["time"], expected_time, rtol=0, atol=1e-10):
            raise ValueError("incomplete or inconsistent output times")
        if any(not np.isfinite(table[name]).all() for name in table.dtype.names):
            raise ValueError("non-finite diagnostics")
    if (not np.isfinite(final).all() or not np.isfinite(work).all()
            or np.any(final[..., 3] <= 0) or np.any(diag["minimum_theta_s"] <= 0)):
        raise ValueError("invalid final fields or state")
    area = meta["dx_m"]**2
    if not np.isclose(work.sum()*area, integrate_power(diag["integrated_friction_power_W"], diag["time"]), rtol=1e-9, atol=1e-6):
        raise ValueError("nodal work and integrated power disagree")
    x = np.arange(nx)*meta["dx_m"]-36000.
    y = np.arange(ny)*meta["dx_m"]-18000.
    masks = region_masks(x, y, meta["barrier_halfwidth_m"])
    for name, column in [("left", "left_power_W"), ("barrier", "barrier_power_W"), ("receiver", "right_power_W")]:
        if not np.isclose(work[masks[name]].sum()*area,
                          integrate_power(regional[column], diag["time"]), rtol=1e-9, atol=1e-6):
            raise ValueError("regional and nodal work disagree")
    fraction = receiver_fraction(final[..., 2], masks["receiver"])
    if not np.isclose(regional["receiver_ruptured_area_m2"][-1],
                      fraction*masks["receiver"].sum()*area):
        raise ValueError("receiver area and peak-speed classification disagree")
    result = dict(meta=meta, final=final, work=work, diagnostics=diag, regional=regional,
                  stations=stations, x=x, y=y, masks=masks)
    result["summary"] = dict(receiver_fraction=fraction, receiver_status=receiver_status(fraction),
        whole_fault_work_J=float(work.sum()*area),
        minimum_state_s=float(diag["minimum_theta_s"].min()),
        minimum_local_power_W_m2=float(diag["minimum_power_W_m2"].min()))
    if snapshots:
        time = np.loadtxt(folder/f"{prefix}.time", ndmin=2)[:, 1]
        if time[0] != 0 or not np.isclose(time[-1], meta["duration_s"]) or np.any(np.diff(time) <= 0):
            raise ValueError("incomplete snapshot history")
        velocity = 2*np.fromfile(folder/f"{prefix}-DataFiles/top_velo_0.out", dtype="<f4").reshape(len(time), nx, ny)
        if not np.isfinite(velocity).all():
            raise ValueError("invalid velocity snapshots")
        result.update(time=time, velocity=velocity)
    return result


def compare_runs(predicted, reference):
    """Use reference coordinates/times shared exactly with the predicted run."""
    def shared(a, b):
        common = a if len(a) <= len(b) else b
        return matching_indices(common, a), matching_indices(common, b)
    px, rx = shared(predicted["x"], reference["x"])
    py, ry = shared(predicted["y"], reference["y"])
    pt, rt = shared(predicted["time"], reference["time"])
    mask = reference["masks"]["active"][rx][:, ry]
    pred_velocity = predicted["velocity"][pt][:, px][:, :, py]
    ref_velocity = reference["velocity"][rt][:, rx][:, :, ry]
    pred_final = predicted["final"][px][:, py]
    ref_final = reference["final"][rx][:, ry]
    ph, rh = shared(predicted["diagnostics"]["time"], reference["diagnostics"]["time"])
    return dict(velocity_relative_L2=relative_l2(pred_velocity, ref_velocity, mask),
        final_slip_relative_L2=relative_l2(pred_final[..., 4], ref_final[..., 4], mask),
        peak_speed_relative_L2=relative_l2(pred_final[..., 2], ref_final[..., 2], mask),
        moment_rate_relative_L2=relative_l2(predicted["diagnostics"]["moment_rate_Nm_s"][ph],
                                          reference["diagnostics"]["moment_rate_Nm_s"][rh]),
        interface_work_relative_error=relative_l2(np.array(predicted["summary"]["whole_fault_work_J"]),
                                                np.array(reference["summary"]["whole_fault_work_J"])),
        arrival=arrival_metrics(pred_final[..., 1], ref_final[..., 1], mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run = load_run(args.folder, snapshots=args.reference is not None)
    report = run["summary"]
    if args.reference:
        report["comparison"] = compare_runs(run, load_run(args.reference, snapshots=True))
    print(json.dumps(report, indent=2))
