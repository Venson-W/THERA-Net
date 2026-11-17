"""Neural material-family closure for the supplied paper's three rupture cases.

The input is the prescribed direct-effect coefficient a_source as a fixed material
descriptor. All four effective coefficients are outputs of actual trained weights.
No ground-truth coefficients replace the network outputs in the coupling API.
The current velocity and state are deliberately excluded from the network inputs.
"""
from pathlib import Path
import numpy as np
from fault_thera import THERANet

A_MIN, A_MAX = 0.006, 0.018
A_CENTER, A_HALF_RANGE = 0.012, 0.006
DEFAULT_OUTPUT = Path(__file__).parent / "results_rupture" / "source_family"


def material_descriptor(a_source):
    a_source = np.asarray(a_source, dtype=float)
    if np.any(~np.isfinite(a_source)) or np.any(a_source < A_MIN-1e-14) or np.any(a_source > A_MAX+1e-14):
        raise ValueError("source a must be finite and within the trained [0.006,0.018] material range")
    return (a_source-A_CENTER)/A_HALF_RANGE


def source_truth(a_source):
    """Analytic teacher only; do not call this for learned model solver runs."""
    a_source = np.asarray(a_source, dtype=float)
    return np.stack([a_source, np.full_like(a_source, 0.012),
                     np.full_like(a_source, 0.020), np.full_like(a_source, 0.600)], axis=-1)


def load_source_model(path=None):
    """Load the checkpoint selected using whole-material-group validation only."""
    return THERANet.load(DEFAULT_OUTPUT / "source_model.npz" if path is None else path)


def source_family_parameters(model, a_field):
    """Map scalar/array a_source to learned [...,4] coefficients [a,b,L_m,f_ref].

    This is executed once per fixed-material field. The solver then uses these
    coefficients in its rate-state traction and aging operators at every step.
    It is a constitutive parameter surrogate, not a learned spatial/time operator.
    """
    a_field = np.asarray(a_field, dtype=float)
    descriptor = material_descriptor(a_field)
    return model.parameters(descriptor.ravel()).reshape(a_field.shape+(4,))


def export_source_field(filename, a_field, model=None):
    """Save little-endian float64 NPY coefficients in NumPy/C row-major order.

    Shape is a_field.shape+(4,); the last contiguous axis is a,b,L,f. The function
    always obtains coefficients by evaluating trained neural weights.
    """
    parameters = source_family_parameters(load_source_model() if model is None else model, a_field)
    np.save(filename, np.asarray(parameters, dtype="<f8", order="C"))
    return parameters
