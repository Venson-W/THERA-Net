"""Audit completed uniform runs against the attributed March SBI reference.

The archive omits physical inputs. These external consistency differences are
not paired neural errors. The half-amplitude check is never a correction.
Completed 100-m outputs are sampled every second node to match the archive.
Run again after fine outputs arrive to refresh JSON and the manuscript table.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "research_notes/rupture_data/upstream_SCEC_v1.npz"


def relative_l2(values, reference):
    difference = np.asarray(values, dtype=np.float64) - reference
    return float(np.sqrt(np.sum(difference**2) / np.sum(reference**2)))


def completed_run(folder, nx, source_times, from_compact=False):
    """Check successful publication and grid/time completeness before loading."""
    record_path = folder / "execution_record.json"
    compact_path = HERE / "results_rupture/compact" / f"{folder.name}.npz"
    if from_compact or not (folder / "solution.time").exists():
        if not compact_path.exists():
            return None, "No complete raw output or compact archive"
        from postprocess_rupture import load_compact
        run = load_compact(compact_path)
        record = run.meta.get("execution_record", {})
        if record.get("returncode") != 0:
            return None, "Compact metadata does not confirm successful execution"
        ids = np.array([np.argmin(np.abs(run.time - t)) for t in source_times])
        mismatch = float(np.max(np.abs(run.time[ids] - source_times)))
        if mismatch >= 1e-7:
            return None, "Compact archive lacks required comparison times"
        stride = nx // 360
        # Compact V already stores the physical jump; do not apply another factor 2.
        values = np.asarray(run.velocity[ids, ::stride, ::stride], dtype=np.float64)
        if values.shape != (76, 360, 180):
            raise ValueError(f"Invalid sampled compact field: {compact_path}")
        return (values, record, mismatch, stride, "compact"), "complete"
    if not record_path.exists():
        return None, "No published execution record"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("returncode") != 0:
        return None, "Nonzero or missing process return code"
    time_path = folder / "solution.time"
    field_path = folder / "solution-DataFiles/top_velo_0.out"
    if not time_path.exists() or not field_path.exists():
        return None, "Published output is incomplete"
    times = np.atleast_2d(np.loadtxt(time_path))[:, 1]
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"Nonmonotone times: {folder}")
    ids = np.array([np.argmin(np.abs(times - t)) for t in source_times])
    mismatch = float(np.max(np.abs(times[ids] - source_times)))
    if mismatch >= 1e-7:
        return None, f"Required 0-15 s samples missing; mismatch {mismatch:g} s"
    expected_bytes = len(times) * nx * (nx // 2) * 4
    if field_path.stat().st_size != expected_bytes:
        return None, "Published field size does not match time/grid dimensions"
    raw = np.memmap(field_path, dtype="<f4", mode="r", shape=(len(times), nx, nx // 2))
    stride = nx // 360
    # Uniform inputs are reflection-symmetric; retain native y-array ordering.
    values = 2.0 * np.asarray(raw[ids, ::stride, ::stride], dtype=np.float64)
    if values.shape != (76, 360, 180) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid sampled field: {folder}")
    return (values, record, mismatch, stride, "raw"), "complete"


def write_latex(grids):
    """Only completed outputs produce rows; no uncomputed placeholders."""
    reference = grids["N360"]["reference"]
    fine_text = (
        " The $100$ m solutions are sampled at every second node before comparison, "
        "so all rows use the same archived grid."
        if grids.get("N720") else ""
    )
    fine_reference = grids.get("N720", {}).get("reference")
    refinement_text = (
        f" At $100$ m, the reference-law field discrepancy decreases to "
        f"${fine_reference['field_relative_l2']:.5f}$ and its central-history discrepancy "
        f"to ${fine_reference['center_hist_relative_l2']:.5f}$, compared with "
        f"${reference['center_hist_relative_l2']:.5f}$ at $200$ m. "
        "The strong improvement at the center does not remove the residual discrepancy "
        "over the complete fault."
        if fine_reference else ""
    )
    rows = []
    for results in grids.values():
        for mode in ("reference", "pinet"):
            if mode not in results:
                continue
            m = results[mode]
            label = "Reference law" if mode == "reference" else "PiNet law"
            rows.append(f"{m['solver_dx_m']:.0f} & {label} & "
                        f"{m['field_relative_l2']:.5f} & {m['moment_relative_l2']:.5f} & "
                        f"{m['center_hist_relative_l2']:.5f} " + r"\\")
    text = (
        "An independent uniform-case check retains the discrepancy with the earlier "
        "SBI archive rather than treating it as a PiNet error. Table~\\ref{tab:externaluniform} "
        "compares complete-fault slip-rate fields, spatially integrated moment rates and "
        "the central station at the archive's 76 times from $0$ to $15$ s."
        + fine_text
        + f" At $200$ m, the reference-law field discrepancy is ${reference['field_relative_l2']:.5f}$, "
        + f"whereas the moment-rate discrepancy is ${reference['moment_relative_l2']:.5f}$; "
        + "close agreement of the spatial integral therefore does not establish waveform "
        "agreement."
        + refinement_text
        + " Halving the computed $200$ m reference slip rate increases its field discrepancy "
        + f"to ${reference['half_amplitude_diagnostic_l2']:.5f}$, "
        + "providing no support for a missing factor of two as the explanation. No amplitude "
        "correction is applied. Because the March archive omits the original physical input "
        "fields and predates the supplied preprint, these differences leave the consistency "
        "of the reconstructed reference problem unresolved; they cannot be attributed "
        "uniquely to discretization, source initialization or a particular implementation. "
        "The paired neural-versus-reference errors and this external discrepancy answer "
        "separate questions.\n\n"
        + "\\begin{table}[htbp]\n\\centering\n\\small\n"
        + "\\caption{External uniform-case relative $L^2$ discrepancies against the March "
        "SBI reference arrays \\citep{tainpakdipat2025data_v1,tainpakdipat2025postprocess}. "
        "All comparisons use the complete $360\\times180$ archive grid at $0.2$ s intervals. "
        "Solver spacing precedes any spatial subsampling. These are not paired PiNet/reference errors.}\n"
        + "\\label{tab:externaluniform}\n\\begin{tabular}{@{}rlrrr@{}}\n\\toprule\n"
        + r"$h$ (m) & Constitutive branch & Slip-rate field & Moment rate & Central history \\" + "\n"
        + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        + "\\end{tabular}\n\\end{table}\n"
    )
    (ROOT / "manuscript/external_rupture_results.tex").write_text(text, encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--from-compact", action="store_true",
                        help="Reproduce the external audit from packaged compact outputs")
    args = parser.parse_args()
    with np.load(ARCHIVE, allow_pickle=False) as source:
        truth = source["upstream_ground_truth"].transpose(2, 0, 1).astype(np.float64)
        source_times = source["time_s"].astype(np.float64)
        metadata = json.loads(str(source["metadata_json"]))
        np.testing.assert_allclose(source["x_m"], -36000.0 + 200.0 * np.arange(360), atol=1e-9)
        np.testing.assert_allclose(source["y_array_m"], -18000.0 + 200.0 * np.arange(180), atol=1e-9)
    np.testing.assert_allclose(source_times, np.linspace(0, 15, 76), atol=1e-10)
    grids, status = {}, {}
    for nx in (360, 720):
        results = {}
        for mode in ("reference", "pinet"):
            folder = HERE / "results_rupture/runs" / f"uniform_N{nx}_{mode}"
            loaded, state = completed_run(folder, nx, source_times, args.from_compact)
            status[folder.name] = state
            if loaded is None:
                if nx == 360:
                    raise RuntimeError(f"Primary run unavailable: {folder}: {state}")
                print(f"SKIP {folder.name}: {state}")
                continue
            ours, record, mismatch, stride, input_storage = loaded
            m = dict(
                field_relative_l2=relative_l2(ours, truth),
                half_amplitude_diagnostic_l2=relative_l2(0.5 * ours, truth),
                moment_relative_l2=relative_l2(ours.sum((1, 2)), truth.sum((1, 2))),
                ours_peak_m_s=float(ours.max()), archive_peak_m_s=float(truth.max()),
                center_hist_relative_l2=relative_l2(ours[:, 180, 90], truth[:, 180, 90]),
                solver_nx=nx, solver_nz=nx // 2, solver_dx_m=72000.0 / nx,
                archive_sampling_stride=stride, matched_times=76,
                input_storage=input_storage,
                maximum_time_mismatch_s=mismatch,
                execution_record=str((folder / "execution_record.json").relative_to(ROOT)),
                driver_sha256=record.get("driver_sha256"), input_sha256=record.get("input_sha256"),
            )
            results[mode] = m
            print(f"N{nx} {mode}: field={m['field_relative_l2']:.9g}, "
                  f"moment={m['moment_relative_l2']:.9g}, center={m['center_hist_relative_l2']:.9g}, "
                  f"half-amplitude={m['half_amplitude_diagnostic_l2']:.9g}")
        if results:
            grids[f"N{nx}"] = results
    out = dict(
        reference=grids["N360"]["reference"], pinet=grids["N360"]["pinet"],
        grid_comparisons=grids, completion_status=status,
        archive=dict(
            path=str(ARCHIVE.relative_to(ROOT)), sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            source_doi=metadata["source_doi"], coordinate_script_doi=metadata["coordinate_script_doi"],
            license=metadata["license"], creators=metadata["creators"], component="V_zx", units="m/s",
            shape_time_x_y=list(truth.shape), spatial_sampling_m=200.0, time_sampling_s=0.2,
            reused_array="upstream_ground_truth only; FNO predictions are not used",
        ),
        scope="Three new source-family propagation cases use independently generated inputs. "
              "This external audit concerns only their uniform case; it does not establish "
              "identity of the original source inputs or reproduce the full FNO experiment.",
        interpretation="External reference-problem consistency remains uncertain because exact "
                       "archived physical inputs are absent and the data are from an earlier release. "
                       "These discrepancies are not paired neural errors. No amplitude correction "
                       "is applied and no single cause is established by this audit.",
    )
    (HERE / "results_rupture/external_uniform_audit.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    write_latex(grids)
    print(out["scope"])
    print(out["interpretation"])


if __name__ == "__main__":
    main()
