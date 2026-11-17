# THERA-Net

Thermodynamically admissible neural modeling of rate-and-state friction.
Calculations use SI units and signed relative slip velocity.

| File | Contents |
| --- | --- |
| `model.py` | Bounded parameter network, convex potential, traction, tangents, steady sliding, exact frozen-speed aging, initial-state inversion, and analytical gradients. |
| `material.py`, `train.py` | Synthetic material family, training, validation, and parameter export. |
| `dynamics.py` | Impedance solve, convex stage functional, predictor–corrector coupling, and a reduced slider with a discrete energy check. |
| `cases.py` | Uniform/heterogeneous prestress, spatial tapers, nucleation, barrier fields, and solver inputs. |
| `diagnostics.py` | Arrivals, field errors, moment rate, receiver activation, and regional frictional work. |
| `adapter/` | C++ friction adapter and one dynamic-rupture executable linked to external UGUCA. |
| `test_math.py` | Numerical derivative, conservation, convergence, spatial-input, diagnostic, and optional adapter tests. |

## Run and test

Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_math
python train.py
```

Training saves the model, datasets, and metrics in `results/`. For a short run,
use `python train.py --maxiter 100 --seeds 10`. The Python tests need no saved
data. Two integration tests are skipped until the external adapter is configured.

The local coupling solves `eta*V + tau(V, theta) = S`. Its potential is
`J(V) = eta*V**2/2 + Phi(V) - S*V`. State updates use the exact aging solution
at the supplied frozen speed; the predictor and corrector both start from the
old state. `predictor_corrector_step` accepts an elastic-history callback;
the full spatial wave history is provided by UGUCA.

## External UGUCA

UGUCA is not bundled. Obtain it from [the upstream repository](https://gitlab.com/uguca/uguca).
The supported revision is `ea1c4c252f8cb119f48602e0b1dba885851c1764`.
Use Linux or WSL with a C++17 compiler, CMake, MPI, FFTW, and BLAS.
Set `UGUCA_SRC` and `UGUCA_BUILD` to absolute paths outside this repository:

```sh
cmake -S "$UGUCA_SRC" -B "$UGUCA_BUILD" -DCMAKE_BUILD_TYPE=Release -DUCA_USE_MPI=ON -DUCA_USE_BLAS=ON
cmake --build "$UGUCA_BUILD" --parallel 4
cmake -S adapter -B ../thera-build -Duguca_DIR="$UGUCA_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build ../thera-build --parallel 4
(cd ../thera-build && ctest --output-on-failure)
export UGUCA_KERNEL_PATH="$UGUCA_SRC/kernels/"
export LD_LIBRARY_PATH="$UGUCA_BUILD/src:${LD_LIBRARY_PATH:-}"
export THERA_RUPTURE_EXECUTABLE="$(realpath ../thera-build/rupture)"
python -m unittest -v test_math
```

With a trained model, run a paired example from the repository root:

```sh
python cases.py --case uniform --nx 180 --model results/model.npz --output inputs/uniform
input_dir="$PWD/inputs/uniform"
for mode in reference thera; do
  mkdir -p "runs/$mode"
  (cd "runs/$mode" && mpirun -np 2 "$THERA_RUPTURE_EXECUTABLE" 180 15 .02 .1 "$input_dir/$mode.bin" solution 1)
done
python diagnostics.py runs/thera --reference runs/reference
```

The executable arguments are `Nx duration dt dump_interval fields.bin output_prefix
pc_iterations [barrier_region_halfwidth_m]`. Use fresh output folders for each run.
Inputs are little-endian float64 in `(Nx, Nx/2, 7)` C order:
`a, b, Dc, f_ref, theta0, prestress, nucleation_shape`.
The two branches share initial state and loading; all four learned coefficients
come from the trained model.

For a barrier, generate inputs with `--case barrier --width-km 2 --a-barrier .014`
and append `1500` to the executable arguments (1000 m flat halfwidth plus a
500 m transition). Other connected configurations are `--case heterogeneous`
and `--case unseen`. Grid sizes must be even divisors of 720, at least 8;
heterogeneous runs sample a fixed master field so refinement preserves loading.

Interface work is the signed integral of `tau*V`, with no clipping. It is not a
complete discrete bulk-energy balance. The slider separately checks physical
and numerical dissipation in its reduced mechanical system. Arrival errors
exclude unreached nodes and report mask disagreements; receiver activation
applies only over the simulated time interval.

The UGUCA-derived adapter retains its LGPL-3.0-or-later notices and the license
texts in `adapter/COPYING` and `adapter/COPYING.LESSER`.
