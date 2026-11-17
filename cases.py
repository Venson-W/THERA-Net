"""Periodic spatial material fields, prestress, and nucleation inputs (SI units)."""
import argparse
from pathlib import Path
import numpy as np
from model import initial_state
from material import load_source_model, source_family_parameters, source_truth

CASES = {
    "uniform": (.008, .008, None, 0, (0., 0.)),
    "heterogeneous": (.007, .010, 1.6, 1101, (-6000., -4500.)),
    "unseen": (.0065, .011, 1.3, 1301, (6000., 0.)),
}


def smooth_box(coordinate, halfwidth, transition):
    if halfwidth < 0 or transition <= 0:
        raise ValueError("halfwidth >= 0 and transition > 0 required")
    x = np.abs(np.asarray(coordinate, dtype=float))
    result = np.zeros_like(x)
    result[x <= halfwidth] = 1.
    mask = (x > halfwidth) & (x < halfwidth+transition)
    z = x[mask]-halfwidth
    result[mask] = 0.5*(1+np.tanh(transition/(z-transition)+transition/z))
    return result


def fractal_field(shape, dimension, seed, mean=75e6, std=5e6):
    """Hermitian phases with |k|^{-(2.5-D)} amplitude and a zero fluctuation mean."""
    if len(shape) not in (1, 2) or min(shape) < 3 or not np.isfinite(dimension) or std < 0:
        raise ValueError("valid 1D/2D grid, dimension, and nonnegative std required")
    spectrum = np.fft.fftn(np.random.default_rng(seed).normal(size=shape))
    phases = spectrum/np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    radius = np.sqrt(sum(g*g for g in grids))
    amplitude = np.zeros_like(radius)
    nonzero = radius > 0
    amplitude[nonzero] = radius[nonzero]**(-(2.5-dimension))
    raw = np.fft.ifftn(phases*amplitude).real
    return (raw-raw.mean())/raw.std()*std+mean


def spatial_nucleation(radius, maximum_radius=3000.):
    if maximum_radius <= 0:
        raise ValueError("positive radius required")
    radius = np.asarray(radius, dtype=float)
    result = np.zeros_like(radius)
    mask = np.abs(radius) < maximum_radius
    result[mask] = np.exp(radius[mask]**2/(radius[mask]**2-maximum_radius**2))
    return result


def temporal_nucleation(time, rise_time=1.):
    if rise_time <= 0:
        raise ValueError("positive rise time required")
    time = np.asarray(time, dtype=float)
    result = np.zeros_like(time)
    result[time >= rise_time] = 1.
    mask = (time > 0) & (time < rise_time)
    result[mask] = np.exp((time[mask]-rise_time)**2/(time[mask]*(time[mask]-2*rise_time)))
    return result


def fields(nx=180, case="uniform", width_km=2., a_barrier=.014):
    """72 x 36 km periodic grid; heterogeneous stress samples a fixed 100 m grid.

    Arrays use (x,y) order. Barrier width is the flat width, with a 500 m
    transition on each side. Reference and learned runs share reference theta0.
    """
    if not isinstance(nx, (int, np.integer)) or nx < 8 or nx % 2 or 720 % nx:
        raise ValueError("nx must be an even divisor of 720, at least 8")
    x = np.arange(nx)*72000./nx-36000.
    y = np.arange(nx//2)*72000./nx-18000.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    active = smooth_box(xx, 15000., 3000.)*smooth_box(yy, 7500., 3000.)
    if case == "barrier":
        if not np.isfinite(width_km) or not 0 <= width_km < 29 or not .006 <= a_barrier <= .018:
            raise ValueError("barrier width must be in [0,29) km and a in [.006,.018]")
        band = smooth_box(xx, 500.*width_km, 500.) if width_km else np.zeros_like(xx)
        aa = .008+.008*(1-active)+(a_barrier-.008)*active*band
        stress = np.full_like(xx, 75e6)
        center = (-9000., 0.)
    else:
        a0, delta, dimension, seed, center = CASES[case]
        aa = a0+delta*(1-active)
        stride = 720//nx
        stress = (np.full_like(xx, 75e6) if dimension is None else
                  fractal_field((720, 360), dimension, seed)[::stride, ::stride])
    parameters = source_truth(aa)
    theta = initial_state(stress, 1e-12, 120e6, parameters)
    nucleation = spatial_nucleation(np.hypot(xx-center[0], yy-center[1]))
    return dict(x=x, y=y, parameters=parameters, theta=theta,
                prestress=stress, nucleation=nucleation)


def write_inputs(directory, data, model=None):
    """Export C-order little-endian (a,b,Dc,f,theta0,prestress,nucleation) fields.

    Always writes reference.bin. Passing a model also writes thera.bin using all
    four learned coefficients while keeping identical initial state and loading.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    branches = {"reference": data["parameters"]}
    if model is not None:
        branches["thera"] = source_family_parameters(model, data["parameters"][..., 0])
    paths = {}
    for name, parameters in branches.items():
        array = np.concatenate([parameters, data["theta"][..., None],
                                data["prestress"][..., None], data["nucleation"][..., None]], axis=-1)
        if not np.isfinite(array).all() or np.any(array[..., :5] <= 0):
            raise ValueError("invalid material field or state")
        paths[name] = directory/(name+".bin")
        np.asarray(array, dtype="<f8").tofile(paths[name])
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "barrier"], default="uniform")
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--width-km", type=float, default=2.)
    parser.add_argument("--a-barrier", type=float, default=.014)
    parser.add_argument("--model", type=Path, help="optional trained checkpoint")
    parser.add_argument("--output", type=Path, default=Path("inputs"))
    args = parser.parse_args()
    data = fields(args.nx, args.case, args.width_km, args.a_barrier)
    model = load_source_model(args.model) if args.model else None
    for name, path in write_inputs(args.output, data, model).items():
        print(f"{name}: {path}")
