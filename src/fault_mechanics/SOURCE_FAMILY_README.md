# Source-family THERA-Net for spatial solver coupling

The original synthetic material family in `fault_thera.py` varies all four
coefficients and does not represent the attached paper's source cases. This
separate experiment trains the same bounded `[1,12,12,4]` network on the material
family `a_source in [0.006,0.018]`, `b=0.012`, `L=0.020 m`, and `f_ref=0.600`.
It preserves the original `results/` directory.

This is a learned constitutive material-family representation. The prescribed
source coefficient enters as a fixed descriptor, `z=(a_source-0.012)/0.006`.
The network predicts all four effective coefficients; the implementation does
not replace any neural output by the teacher value. The current slip rate and
state are excluded from the network inputs. The existing rate-state potential,
traction, aging update and admissibility arguments are therefore unchanged.

This experiment does not learn unknown friction physics or a space-time rupture
operator. A spatial solver evaluates the network once for each fixed material
field, then uses the resulting coefficients in its constitutive updates. A
speedup of the rupture dynamics does not follow from this implementation.

## Reproduce

```text
python train_source_family.py --maxiter 5000 --seeds 10 11 12
python test_source_family.py
```

Outputs are in `results_rupture/source_family/`. `source_model.npz` is selected
solely by the lowest whole-material-group validation loss across checkpoints and
the three seeds. `config.json` records all groups, bounds, sample seeds, units,
software versions and optimizer settings; `metrics.json` reports actual results.
Training receives traction and state-step labels, not parameter labels. The
teacher data are noiseless and exactly follow the prescribed constitutive prior.

There are 97 equally spaced material groups: 61 training, 16 validation and 20
test groups. Each training/validation group has 48 samples; each test group has
128 samples. The six actual central/outer source coefficients `0.0065, 0.007,
0.008, 0.016, 0.017, 0.0175` are held out from both training and validation.
The signed slip-speed magnitude covers `1e-12 ... 10 m/s`; the state covers
`1e-3 ... 1e24 s`, sampled uniformly in base-10 logarithm. Broad state coverage
is necessary because inversion of the source's initial traction at `1e-12 m/s`
produces very large kinetic-state values. These are internal variables, not
literal geological ages.

The loss is the mean of `((mu_pred-mu)/0.1)^2` plus
`log(theta_next_pred/theta_next)^2`. The frozen-speed step uses
`h=u*0.020/abs(V)` with uniform `u in [0.05,2]`. Future-state supervision resolves
the friction-only confounding between reference friction and slip distance.

## Solver interface

```python
from source_family_thera import load_source_model, source_family_parameters
from fault_thera import traction, aging_update

model = load_source_model()
parameters = source_family_parameters(model, a_field)  # shape a_field.shape+(4,)
tau = traction(velocity, theta, effective_pressure, parameters)
theta_next = aging_update(theta, velocity, dt, parameters[..., 2])
```

The last parameter axis is `[a, b, L_m, f_ref]`. The material descriptor must lie
inside the trained range; the interface rejects values outside it rather than
silently clipping them. `source_truth` is provided only for the analytic reference
and must not be used in a learned-model solver branch.

For a native solver:

```python
from source_family_thera import export_source_field
export_source_field("neural_coefficients.npy", a_field, model=model)
```

The NPY file stores a little-endian float64 array in C row-major order, with the
four coefficients contiguous at each site. `example_coefficients.npy` and
`source_case_coefficients.csv` demonstrate the field shape and all six actual
central/outer values. The exporter always evaluates the trained weights.

The source model is scalar. A vector dynamic-rupture solver must apply these
scalar friction coefficients to its established isotropic friction law using
slip-speed magnitude and correctly oriented resisting traction. Its bulk solver,
spectral history, wave propagation, boundary conditions, state integration,
convergence and sign conventions require independent validation.

## Locked training results

Seed 12 was selected using validation, at iteration 4600. All three training
runs ended with optimizer convergence status. The held-out traction relative L2
error is `2.1821918e-6`; the RMSE at 120 MPa pressure is `0.0002151625 MPa`.
A 4001-point material grid has maximum absolute errors
`[3.6502322e-7, 2.2789395e-7, 3.8436675e-7 m, 3.4591396e-6]` in `[a,b,L,f_ref]`.

The initialization audit spans shear stresses 45–105 MPa and the entire material
range. Its analytic initial states span approximately `0.144 ... 1.793e23 s`,
inside the training state range. The maximum absolute logarithmic initial-state
ratio is `6.955978e-4`. This checks a source initialization sensitivity; it does
not establish trajectory or rupture-front accuracy.

Selected checkpoint SHA-256:
`7307121720ef68469f97056bddc3abead11878cf73e21bf964af8286ee57405d`.

The checkpoint round trip reproduces neural coefficients exactly. See
`metrics.json` for complete seed-specific results and numerical precision.
