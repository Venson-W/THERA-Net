# THERA-Net manuscript code

Source code accompanying **THERA-Net: Thermodynamically Admissible Neural Constitutive Modeling of Rate-and-State Friction**, by Yu-Han Wang.

The implementation learns bounded material parameters and evaluates a convex frozen-state friction potential, its traction, and an exact frozen-speed aging update. Experiments include synthetic constitutive learning, a forced slider, dynamic rupture comparisons, and segmented-fault barrier studies. Historical `PiNet`, `pinet`, and `TITAN` filenames and data keys are retained to match the manuscript's computation records.

## Release contents

- `PiNet-main/Fault mechanics/`: constitutive model, training, input generation, analysis, tests, and detailed workflow notes.
- `PiNet-main/Fault mechanics/rupture_solver/`: manuscript-specific C++ coupling adapters and batch runners, with their original license notices.
- `IJSS_PiNet_Rupture/scripts/`: manuscript figure/table generators.
- `IJSS_PiNet_Rupture/figures/pinet_framework.tex`: the original TikZ framework schematic.
- `research_notes/`: batch preparation and solver-verification scripts plus the original build record.
- `MANIFEST.sha256`: checksums of the published files.

**This is a source-code release.** The upstream UGUCA solver, kernels, libraries, and executables are excluded. Trained checkpoints, calibration datasets, figure data, batch/result JSON archives, and simulation histories are also excluded. Generate the needed outputs locally or obtain the corresponding manuscript data separately before running data-dependent analyses. Original workflow notes describe the larger research workspace and may refer to those omitted artifacts. Manuscript PDFs, cover letters, third-party Python packages, and unrelated original PiNet examples are not included.

The original relative directory layout is preserved. Numerical source files are unchanged. Run commands below from this directory unless another working directory is specified.

## Python setup and independent tests

Use Python 3.10 or newer (the original runs used Python 3.12):

```sh
python -m pip install -r "PiNet-main/Fault mechanics/requirements.txt"
cd "PiNet-main/Fault mechanics"
python -m unittest -v test_fault_pinet test_barrier_cases
python -m unittest -v test_postprocess_rupture.TestRupturePostprocess.test_relative_field_norm_and_region test_postprocess_rupture.TestRupturePostprocess.test_arrival_censoring_and_mismatches test_postprocess_rupture.TestRupturePostprocess.test_common_time_alignment_rejects_noncommon_samples
```

These 17 tests do not need archived checkpoints or simulation data. The four `test_source_family` tests require training outputs. The other two postprocessing tests require completed compact rupture records, so unrestricted test discovery is not suitable for a fresh source-only checkout.

## Training and input generation

From `PiNet-main/Fault mechanics`:

```sh
python run_experiments.py --maxiter 1200 --seeds 0 1 2
python train_source_family.py --maxiter 5000 --seeds 10 11 12
python -m unittest -v test_source_family
python prepare_rupture_inputs.py
```

Training writes calibration data, model weights, and metrics to its default output directories; rerunning overwrites them. Retraining may differ with numerical-library versions, so regenerating a model does not guarantee a byte-identical manuscript checkpoint. No rupture trajectories are used for fitting or checkpoint selection. See `SOURCE_FAMILY_README.md` for the calibration protocol.

## Figures and tables

Install `IJSS_PiNet_Rupture/requirements.txt`. Figure scripts also require `pdflatex` on PATH with PGF/TikZ, standalone, Latin Modern, and amsmath. The schematic can be built directly:

```sh
python IJSS_PiNet_Rupture/scripts/build_framework.py
```

The following scripts require their corresponding manuscript NPZ/JSON files under `IJSS_PiNet_Rupture/data/`, which are not included in this source-only release:

```sh
python IJSS_PiNet_Rupture/scripts/build_figures.py
python IJSS_PiNet_Rupture/scripts/build_barrier_figures.py
python IJSS_PiNet_Rupture/scripts/build_table.py
python IJSS_PiNet_Rupture/scripts/build_loss_table.py
python IJSS_PiNet_Rupture/scripts/write_barrier_section.py
```

See `IJSS_PiNet_Rupture/scripts/README_figures.txt` for required files and plotting conventions. `build_figures.py --prepare-data` extracts plotting data from completed simulation outputs and calibration results; it does not run the solver.

## Dynamic rupture: install UGUCA separately

Use [UGUCA](https://gitlab.com/uguca/uguca), revision `ea1c4c252f8cb119f48602e0b1dba885851c1764`. On Linux or Ubuntu under WSL, install CMake, a C++17 compiler, MPI, FFTW, and BLAS. From this directory:

```sh
git clone https://gitlab.com/uguca/uguca.git research_notes/uguca
git -C research_notes/uguca checkout ea1c4c252f8cb119f48602e0b1dba885851c1764
cmake -S research_notes/uguca -B research_notes/uguca-build -DCMAKE_BUILD_TYPE=Release -DUCA_BENCHMARKS=ON -DUCA_TEST=ON -DUCA_EXAMPLES=ON -DUCA_USE_MPI=ON -DUCA_USE_BLAS=ON
cmake --build research_notes/uguca-build --parallel 4
cmake -S "PiNet-main/Fault mechanics/rupture_solver" -B research_notes/pinet-rupture-build -Duguca_DIR="$PWD/research_notes/uguca-build" -DCMAKE_BUILD_TYPE=Release
cmake --build research_notes/pinet-rupture-build --parallel 4
python "PiNet-main/Fault mechanics/rupture_solver/run_rupture_ext4.py"
```

Generate the model and rupture inputs first. The retained runners stage computation under `/var/tmp` and copy completed results back. Their MPI process counts and concurrency reflect the original workstation; review them for your machine. Full simulations require substantially more time and storage than this source package.

For the main barrier batch, from this directory after training:

```sh
python research_notes/prepare_barrier_batch.py main --nx 360 --dt .01 --widths 1 2 4 --a .010 .011 .012 .014 --modes reference pinet
python "PiNet-main/Fault mechanics/rupture_solver/run_barrier_suite.py" "PiNet-main/Fault mechanics/results_barrier/main.json"
```

`BARRIER_README.txt` describes additional pilot/refinement/temporal batches required for complete analysis. Each batch and its input fields must be generated before running it. The original build and verification record is in `research_notes/uguca_build_notes.txt`.

## Scientific scope and attribution

The manuscript compares analytical and learned coefficients within the same numerical solver; paired errors do not establish continuum accuracy or solver acceleration. The material family is prescribed synthetic calibration, not identification of unknown fault properties. Detailed code READMEs document the numerical limitations.

Conceptual background: [PiNet](https://github.com/PinZhang3/PiNet), DOI `10.1016/j.compgeo.2023.105511`, and t-PiNet, DOI `10.1016/j.jmps.2025.106049`. UGUCA reference: DOI `10.1016/j.softx.2021.100785`.

The UGUCA-derived adapter retains its LGPL-3.0-or-later notices and accompanying `COPYING` and `COPYING.LESSER` texts. Those notices apply to the adapter. No new blanket license is assigned by this packaging step.
