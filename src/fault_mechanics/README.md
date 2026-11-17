# THERA-Net: thermodynamic constitutive learning and rupture coupling

This directory implements thermodynamically admissible neural constitutive
modeling of rate-and-state friction in NumPy. A learned material-parameter map
is coupled to analytical friction and state-evolution operators.

The workflow contains **synthetic constitutive identification, a forced unit-area
slider, and matched analytical-coefficient/THERA-Net runs in a full dynamic UGUCA
spectral boundary-integral solver**. The spatial fault is two-dimensional inside
a three-dimensional elastic medium; strike slip is active, while opening and the
other tangential slip component are constrained. Causal elastic-wave history is
retained. This is a scalar friction closure coupled to spatial elastodynamics.

All 15 spatial production runs completed successfully through 15 s: the paired
three-case runs at 400 m and 200 m, the uniform pair at 100 m, and a uniform
200 m reference run at half timestep. Saved execution records, solver logs and
arrays support the figures and metrics. The native solver's 32 tests passed,
and a separate full 15 s native-versus-adapter comparison was completed.

The new heterogeneous cases use deterministic seeds and follow the physical
families of source Figs. 11–13. They do not reproduce the exact source realizations,
its FNO training or its reported speedup. Selected older public benchmark arrays
and a provenance audit are available separately in `../../research_notes/rupture_data`.

## Run

With Python 3.10 or newer:

```text
python -m pip install -r requirements.txt
python test_fault_thera.py
python run_experiments.py --maxiter 1200 --seeds 0 1 2
```

For the current workspace execution the bundled Python runtime was used, and
SciPy/Matplotlib were installed locally in `../../research_notes/python_deps`.
`run_experiments.py` detects that optional local path. The core model and its tests
require only NumPy. The results contain the actual runtime versions. Running again
overwrites generated files inside `results`; use `--output another_directory` to
preserve a previous run.

## Model and conventions

All calculations use SI units. Pressure is compression positive. The scalar
traction `tau` is the signed resisting traction conjugate to slip rate, so that
`tau * V` is the nonnegative frictional dissipation. A solver must apply opposite
traction signs to the two fault faces consistently with its jump convention.
The constitutive implementation is scalar, with a prescribed slip direction.
The spatial adapter constrains the second tangential velocity. A general vector
friction extension remains a theoretical construction in the manuscript.

A tanh network with layers `[1,12,12,4]` maps a **fixed material descriptor** to
bounded positive `a, b, L, f_ref`. Current velocity, current state and time are not
network inputs. The characteristic speed and dissipation potential are

```text
c(theta) = 2 V_ref exp[-(f_ref + b ln(V_ref theta / L))/a]
phi(V;theta) = p a [V asinh(V/c) - sqrt(V^2+c^2) + c]
tau = d phi/dV = p a asinh(V/c)
d tau/dV = p a / sqrt(V^2+c^2) > 0      (p > 0)
```

`phi` has units W/m², is convex at fixed state, and is not the frictional power
`tau*V`. The interface free energy is set to zero. The aging state is nonenergetic
kinetic memory. This construction does not derive the aging evolution from an
associative dissipation principle and is not a complete generalized standard
material for joint slip-state evolution. Thermodynamic admissibility and physical
accuracy of the memory law are separate claims.
Traction uses a log-domain asinh evaluation; the equivalent stable potential is
`p a V [y - tanh(y/2)]`, with `y=asinh(V/c)`. No exponent clipping changes the
constitutive law. Finite-input validation and an extreme-state test guard the
implemented operators. The randomized numerical audit covers material descriptors
`[-1.25,1.25]`, speeds `1e-12 ... 10 m/s`, and states `1e-3 ... 1e12 s`;
arbitrary IEEE floating-point extremes are not a supported physical domain.

The state update is exact for a frozen slip-speed magnitude over a step:

```text
z = abs(V) h/L
theta_next = theta exp(-z) + h[-expm1(-z)/z]
```

The zero-speed limit is `theta_next = theta + h`. This update is positive.
Material coefficients must remain independent of instantaneous velocity for the
above convexity and derivative statements to hold.

## Synthetic experiment design

The teacher is an explicitly declared smooth material family in
`true_parameters`. It is not an experimentally inferred fault property field.
There are 61 descriptors over `[-1,1]`: 39 train, 10 validation, 12 test. Samples
from one descriptor never cross those splits. The boundary descriptors train;
test descriptors are interpolation points. Train and validation use 32 signed
velocity/state samples per descriptor; test uses 64. Training spans
`abs(V)=1e-8 ... 1e-2 m/s` and `theta=1e-1 ... 1e8 s`.

The data are noiseless. The loss uses traction labels and exact state-transition
labels, with no parameter labels. State-transition supervision is needed because
friction-only observations at a known state confound `f_ref` and `L` through
`f_ref - b ln(L)`. Validation loss selects a checkpoint. Test data do not affect
scaling, optimization, or checkpoint choice. Three initializations are saved.

Low-speed (`1e-12 ... 1e-8 m/s`) and high-speed (`1e-2 ... 1 m/s`) tests keep the
held-out material descriptors. These evaluate extrapolation of a prescribed
constitutive prior, not discovery of unknown friction mechanisms. The numerical
advantage of this exact-prior synthetic problem must not be generalized to real
fault data without additional experiments.

A direct-traction tanh MLP `[4,16,16,1]` provides an architectural reference. It
receives descriptor, signed velocity information, and current state, and learns
traction labels only. It has a different parameter count and does not use future
state labels, so the comparison is **not an equal-information or equal-capacity
ablation** and does not establish general superiority over neural operators.

## Reduced mechanical verification

The slider uses mass `267000 kg/m²`, stiffness `1e8 Pa/m`, and scalar radiation
impedance `rho cs/2`. This impedance is a local surrogate and omits the SBI history
convolution. A smooth external traction increase forces a transient. This slider
does not represent spatial rupture fronts or an elastic-wave solution; the
separate UGUCA workflow below supplies the spatial dynamics.

The scheme is

```text
M (V_next - V)/h + k delta_next + eta V_next
    + tau(V_next, theta_old) = tau0 + q_next
delta_next = delta + h V_next
```

The frozen-state scalar solve is strictly monotone. After it, the state receives
the exact frozen-new-speed update. With `E = M V²/2 + k delta²/2`, the discrete
identity includes both physical and backward-Euler numerical dissipation:

```text
E_next-E + h[eta V_next² + tau_next V_next]
  + M(V_next-V)²/2 + k(delta_next-delta)²/2
  = h(tau0+q_next)V_next.
```

Numerical dissipation is logged separately; it is not attributed to the friction
law. A four-level time-step refinement study compares to `h=0.00125 s`.

## Source-case inputs and restart data

`source_cases.py` records the supplied preprint's SI-unit parameters, training and
held-out fractal dimensions, paired direct-effect coefficients, smooth VW/VS
transition and smooth nucleation. `source_case` outputs a 1D line or 2D plane of
coordinates, stress, friction parameters, source-consistent initial state and
spatial nucleation shape. This is an input builder, not a rupture solver.

The random-field generator specifies the amplitude spectrum as
`|k|^(-(2.5-D))`, uses signed FFT frequencies and Hermitian random phases, and
normalizes population standard deviation. This is an explicit reproducible
convention; it cannot exactly regenerate unpublished source random realizations.

For complete SBI restarts and later dataset studies, retain the following fields
grouped by realization. This is a restart-data specification, not a claim that the
current reduced output set includes every internal history:

| Quantity | 1D fault line | 2D fault plane | Unit |
| --- | --- | --- | --- |
| time | `[nt]` | `[nt]` | s |
| coordinates | `[nx,1]` | `[nx,ny,2]` | m |
| slip / slip-rate vectors | `[nt,nx,ncomp]` | `[nt,nx,ny,ncomp]` | m / m/s |
| theta | `[nt,nx]` | `[nt,nx,ny]` | s |
| effective pressure | `[nt,nx]` | `[nt,nx,ny]` | Pa |
| traction / external perturbation | `[nt,nx,ncomp]` | `[nt,nx,ny,ncomp]` | Pa |
| fixed material parameters | `[nx,4]` | `[nx,ny,4]` | mixed, documented |
| SBI history | solver-specific full restart state | solver-specific full restart state | solver-specific |

Include realization ID, seed, mesh/solver convergence, face/jump convention,
boundary conditions, time origin, density and elastic constants. A velocity-only
snapshot is not a closed Markov state for rate-and-state friction plus SBI memory.
Do not split shifted time windows from the same realization between train/test.

## Source-family calibration and spatial runs

`SOURCE_FAMILY_README.md` documents the separately trained source-family model.
It has 232 parameters in the same `[1,12,12,4]` architecture, but a new teacher:
`a=a_source`, `b=0.012`, `L=0.020 m`, `f_ref=0.600`. Its fixed input is
`z=(a_source-0.012)/0.006`. The descriptor therefore already supplies the true
direct-effect coefficient up to an affine transformation; the other teacher
coefficients are constants. This is an identity-plus-constants synthetic
calibration, even though all four effective coefficients are actual neural
outputs and training uses traction/state labels rather than coefficient labels.
It tests propagation of constitutive approximation error, not identification of
unknown fault properties. No rupture snapshots are used for fitting, validation
or checkpoint selection. A new stress dimension or hypocenter is not a new input
to this fixed-material network. In particular, `D=1.3` is not constitutive
extrapolation for this THERA-Net.

From this directory:

```text
python train_source_family.py --maxiter 5000 --seeds 10 11 12
python test_source_family.py
python prepare_rupture_inputs.py
```

Reusing the saved `results_rupture/source_family/source_model.npz` avoids retraining.
The source-family record contains its checkpoint hash, material-group splits,
training histories and actual optimizer outcomes. Its four tests passed. The
older local experiment and `results/` are preserved separately.

The fixed spatial inputs are:

| Case | D | a0 | Delta a0 | Seed | Hypocenter, km |
| --- | --- | --- | --- | --- | --- |
| Heterogeneous | 1.6 | 0.007 | 0.010 | 1101 | (-6, -4.5) |
| Uniform 75 MPa | not applicable | 0.008 | 0.008 | not applicable | (0, 0) |
| Heterogeneous | 1.3 | 0.0065 | 0.011 | 1301 | (6, 0) |

All begin at `t=0`. The heterogeneous fields are generated on a `720x360`
master grid at 100 m spacing; coarser grids subsample the same field without
renormalizing it. `results_rupture/input_manifest.json` records seeds, grid,
state ranges and binary-file hashes. The uniform branch overwrites its generated
stress with exactly 75 MPa, so its generator seed has no physical effect.
Both branches receive the same reference-inverted initial state and prestress;
each law then equilibrates its own initial slip velocity. Tiny differences from
the nominal `1e-12 m/s` initial speed are consequently possible in the neural run.

### Native build

The adapter in `rupture_solver/` uses UGUCA revision
`ea1c4c252f8cb119f48602e0b1dba885851c1764`, built separately. Requirements are
CMake, a C++17-capable compiler, FFTW, MPI, and BLAS for the configuration used
here. UGUCA's kernel files must remain available at the source path recorded by
its build. From the workspace root, in Linux or WSL:

```sh
cmake -S research_notes/uguca -B research_notes/uguca-build -DCMAKE_BUILD_TYPE=Release -DUCA_USE_BLAS=ON -DUCA_USE_MPI=ON -DUCA_BENCHMARKS=ON
cmake --build research_notes/uguca-build --parallel
cd research_notes/uguca-build
ctest --output-on-failure
cd ../..
cmake -S "src/fault_mechanics/rupture_solver" -B research_notes/thera-rupture-build -Duguca_DIR="$PWD/research_notes/uguca-build" -DCMAKE_BUILD_TYPE=Release
cmake --build research_notes/thera-rupture-build --parallel
```

The 32 native UGUCA tests passed in the execution environment. At 400 m, a full
15 s analytical-coefficient run was checked against native TPV101 at 26 common
output times. The largest absolute relative-slip-speed discrepancy was
`2.3841858e-7 m/s`, at the scale of the single-precision field output. Evidence is
in `research_notes/uguca_full_comparison.json`; the earlier smoke check is also
retained. These compare implementations, not a finite-grid solution with an exact
continuum solution.

### Runtime and output

The canonical runner is `rupture_solver/run_rupture_ext4.py`, executed with
Linux Python from this directory:

```sh
python3 rupture_solver/run_rupture_ext4.py
```

For this Windows workspace the equivalent PowerShell command is:

```powershell
wsl.exe -d Ubuntu -- python3 "/mnt/e/Work/Research/PingsCode/src/fault_mechanics/rupture_solver/run_rupture_ext4.py"
```

It stages the executable, shared library, kernels, inputs and output on the
Linux-native filesystem at `/var/tmp/thera-rupture-20261112` by default. Set
`THERA_RUPTURE_RUNTIME` to use another native directory. The staging location
survives WSL restarts; preserve it until the results have been copied and checked.
The runner sets `UGUCA_KERNEL_PATH` to its staged kernels directory with a trailing
slash and `LD_LIBRARY_PATH` to its staged library. The adapter honors the kernel
override; without it, it uses UGUCA's configured source-kernel path.
UGUCA's many small coordinate-output writes were
slow on the mounted Windows filesystem. Native staging and bulk copying back
avoid that bottleneck. Preserve run arguments, logs and exit statuses with the
copied results; do not interpret an interrupted output directory as complete.
Each run has `execution_record.json`, and `results_rupture/runs/batch_manifest.json`
collects completed process records, including nonzero returns if a run fails.
The script schedules paired reference/THERA-Net runs for all three cases at Nx180
and Nx360, paired uniform runs at Nx720, and a uniform Nx360 reference run at
half timestep: 15 requested runs, with two concurrent runs and eight MPI ranks
per run. All 15 runs in the reported batch completed with return code zero.
The original `run_rupture_suite.py` remains an optional runner
using workspace-mounted directories and is not the preferred long-run WSL path:

```text
python run_rupture_suite.py --nx 180 360 --duration 15 --jobs 2 --ranks 8
```

The native executable takes these positional arguments:

```text
rupture_driver Nx duration_s dt_s dump_interval_s fields.bin output_prefix predictor_corrector_iterations
```

The configured comparison has a 72x36 km fault, one predictor-corrector iteration,
15 s duration, full-field output every 0.1 s and `dt=0.02*(180/Nx) s`. Thus
`Nx=180,360,720` means 400,200,100 m spacing and `dt=0.02,0.01,0.005 s`.
The extra uniform 200 m reference control uses `dt=0.005 s`. The completed
resolution/case combinations are those listed above; no full 100 m heterogeneous
ensemble is claimed.

The driver tracks rupture arrivals at every step with linear crossing
interpolation, using thresholds 0.5 and 0.001 m/s. It records six station histories,
moment rate, minimum state, interface frictional power and final fields. Matching
THERA-Net/reference errors must use the same grid, output times and spatial mask.
Interface-power positivity does not establish a global balance of bulk energy,
radiation and external work. The exact discrete energy identity above belongs
to the slider, not this full SBI integrator.

### Measured outcomes and limits

For the three main 200 m comparisons, THERA-Net/reference slip-rate errors within
`|x|<=15 km, |y|<=7.5 km` are 0.0744% (D=1.6), 0.1874% (uniform) and 0.2194%
(D=1.3). Whole-fault moment-rate relative errors are `3.8696e-5`, `1.0019e-4` and
`9.6894e-5`, respectively. The 100 m uniform paired field error is 0.1805%.
See `results_rupture/rupture_metrics.json` for full precision and masks.

Reference-grid sensitivity is much larger: the uniform 200-to-100 m joint
grid/time refinement changes its sampled weakening-region slip-rate field by
about 43.18%. Small same-grid THERA-Net errors do not establish converged continuum
waveforms. Sub-millisecond differences in interpolated arrivals are also not
sub-millisecond physical timing accuracy. Main-case peak speeds exceed the
calibration limit of 10 m/s, and the D=1.3 minimum state falls slightly below
0.001 s; these evaluations rely on the constitutive prior. Whole-fault frictional
power stays positive, but near-rest pointwise values reach about `-1.58e-4 W/m²`
because of finite root tolerance and cancellation. The theoretical sign guarantee
is distinct from a strict floating-point power certificate for the SBI integrator.

Regenerate spatial figures/tables and the external audit from saved raw runs:

```text
python postprocess_rupture.py
python audit_external_uniform.py
```

The packaged compact outputs reproduce these analyses without raw native binaries:

```text
python postprocess_rupture.py --from-compact
python audit_external_uniform.py --from-compact
```

The external audit additionally uses the attributed `upstream_SCEC_v1.npz` below.
Its 16 numerical metrics reproduce exactly from raw and compact arrays.

### External benchmark provenance

`research_notes/rupture_data/UPSTREAM_BENCHMARK_AUDIT.txt` records selected source
SBI/FNO arrays from Zenodo version 1, DOI `10.5281/zenodo.15085623`, and the
postprocessor from version 2, DOI `10.5281/zenodo.15091084`. Both are CC-BY-4.0.
Only selected ZIP members were downloaded by byte range and verified by member
CRC32; the postprocessor's published MD5 was verified. The supplied safe NPZ
conversions load with `allow_pickle=False` and explicitly identify upstream FNO.

These March arrays contain `360x180x76` slip-rate samples, 200 m spacing and
0–15 s time at 0.2 s intervals. They lack stress, friction, state and nucleation
inputs. They are not verified exact data for the later October paper figures.
The postprocessor supplies the axis and station conventions, including the sign
difference between its array y coordinate and plotted y coordinate. See the
audit before using them for an external check. The new paired runs use their
own generated inputs and reference-law trajectories. FNO predictions are neither
THERA-Net outputs nor training targets; different maps and training information
preclude interpreting these paired errors as superiority over the published FNO.

The independent external uniform comparison retains a substantial discrepancy:
after sampling the 100 m reference run onto the archive grid, complete-fault
field, moment-rate and central-history relative errors are `0.27204`, `0.01693`
and `0.01287`. At 200 m they are `0.33131`, `0.01790` and `0.28736`. The center
improves strongly under refinement, while field disagreement remains. These are
reference-problem consistency differences, not paired neural errors; no unique
cause is established without the original physical inputs. Halving the computed
slip rate makes agreement worse, and no amplitude correction is applied.

The data license is [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit Napat Tainpakdipat, Mohamed Abdelmeguid, Chunhui Zhao and Ahmed Elbanna,
link the exact March release, and identify the safe NPZ conversion and added
coordinate/provenance metadata when redistributing these files.

## Outputs

`results/config.json` records splits, units, bounds, seeds and software versions.
`results/metrics.json` contains actual error, optimizer-status, admissibility,
time-refinement and discrete-energy results. NPZ files store datasets and weights;
CSV files store loss/slider/history arrays. Figures are written as PNG and
vector PDF. The code uses no publication-result constants or placeholder metrics.

## Attribution

UGUCA is Copyright (C) 2021 ETH Zurich (David S. Kammer), with authors D. S.
Kammer, G. Albertini and C.-Y. Ke, and is distributed under LGPL-3.0-or-later.
The adapter derives from its TPV101 driver and rate-and-state class and carries
the same LGPL-3.0-or-later notices. Retain these notices and the upstream license
when redistributing those sources or binaries. No new blanket license is
assigned by this release. Attribute reused March benchmark data to Napat
Tainpakdipat, Mohamed Abdelmeguid, Chunhui Zhao and Ahmed Elbanna and identify
the conversion/coordinate supplementation when redistributing the safe NPZ files.
