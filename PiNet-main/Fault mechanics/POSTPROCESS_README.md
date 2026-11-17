# Spatial rupture output analysis

`postprocess_rupture.py` consumes completed `results_rupture/runs/` calculations.
It leaves all raw solver files intact. A run is accepted only after final fields,
wall time, metadata, complete full-step station/diagnostic histories, and exact
snapshot binary sizes are present. It checks the grid, recorded times, finite
values, positive states, peak-rate consistency, station coordinates, and the
factor of two between the symmetric slip jump and UGUCA's one-face outputs.
Completion fingerprints invalidate cached analyses when a run is recomputed.
Successful execution records, including the actual binary and input hashes, are
retained in compact metadata and exposed in the numerical audit.

```text
python postprocess_rupture.py --require-main
python test_postprocess_rupture.py
```

`--require-main` requires completed reference/PiNet pairs for all three cases at
Nx=360 (200 m). `--figure-nx` selects another available figure resolution. Partial
runs are explicitly listed and are not included in numerical comparisons.

The raw field files use float32 one-face velocity/displacement. The processor
multiplies them by two. Threshold arrivals and peak speed are read from the final
float64 fields tracked at every integration step. Snapshot times (nominally every
0.1 s) are used only for the field-velocity error, not to estimate arrival times.
No Gaussian or other spatial smoothing is applied.

Field velocity, final slip, peak speed and arrival errors use the geometric
velocity-weakening rectangle `|x| <= 15000 m, |y| <= 7500 m`. Arrival RMSE and
95th-percentile absolute error use jointly reached points; false/missed crossing
counts and fractions are reported separately. Moment-rate error uses full-step
whole-fault integrals. Intergrid field comparisons sample the finer reference
on exactly common coarse nodes and times. They change both spatial and temporal
resolution and do not alone establish continuum convergence. Intergrid station
waveforms are excluded from quantitative metrics because nearest-node station
coordinates may differ.

Each run receives a `compact/<run_name>.npz` containing physical slip rate `V`,
final fields, stations, diagnostics, names of diagnostic columns, times,
coordinates and complete metadata. Velocity remains float32 as written by the
solver; final fields and histories are float64. The same metrics, tables and
figures can be regenerated from a distributed compact package without the large
raw binaries or a solver rerun:

```text
python postprocess_rupture.py --from-compact --require-main
```

`--compact-only` is an alias. Compact loading independently repeats the shape,
metadata, coordinate, time, state and station consistency checks; it does not
simply trust a saved validation flag. Optional transverse-speed diagnostics are
read by CSV header name, so original and extended diagnostic schemas are supported.

Outputs are `results_rupture/rupture_metrics.json`, three
`manuscript/figures/rupture_<case>.pdf` figures, `rupture_summary.pdf`,
`manuscript/rupture_table.tex`, and `rupture_refinement_table.tex`. PNG previews are
in `results_rupture/figures/`. The individual figures show raw 0.5 m/s arrival
contours at 0.5 s spacing and six full-step station traces. The summary compares
whole-fault moment rate and arrival differences, with missing threshold crossings
kept distinct from numerical timing errors.

When the uniform 100 m reference is available, the additional
`rupture_uniform_refinement.pdf` shows 400/200/100 m reference arrival contours
(1 s spacing) and raw station A/B/C traces. Station B lies at (-6,-3.2) km on the
400 m mesh and (-6,-3) km on the two finer meshes, and that location difference
is labelled explicitly. Its coarse/fine waveform difference includes this
spatial sampling offset. A and C use the same coordinates on all three meshes.

An additional `uniform_N360_reference_dt05` run is kept separate from the
baseline 200 m reference. The JSON `reference_controls` array reports its
fixed-grid comparison against dt=0.01 s and its fixed-dt=0.005 s comparison
against the 100 m reference. These controls distinguish time-step sensitivity
from spatial-grid sensitivity; neither overwrites the main coupled-refinement
table or the baseline main figures.

The audit records maximum speed, minimum state, state-range bounds, station peak
speeds, local minimum frictional power, the count of negative minima, and event
integrated frictional energy. Negative numerical power values are retained, not
clipped. The physical constitutive law's nonnegative power does not imply an
exact discrete global energy certificate for the wave solver. Speeds above
10 m/s or states below 1e-3 s are flagged as outside the source-family training
range; the prescribed constitutive prior is then being extrapolated.
