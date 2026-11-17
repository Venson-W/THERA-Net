"""SI-unit source-case input builders; these do not solve dynamic rupture.

Source: attached Tainpakdipat et al. preprint, doi:10.22541/essoar.174526096.60853381/v2.
Fractal spectra use Hermitian phases and signed FFT wavenumbers, an explicit
reproducible convention where the source's discrete FFT description is incomplete.
"""
import numpy as np

SOURCE_PARAMETERS = {
    "cs_m_s": 3464., "cp_m_s": 6000., "density_kg_m3": 2670.,
    "f_ref": 0.6, "L_m": 0.02, "V_ref_m_s": 1e-6, "V_initial_m_s": 1e-12,
    "normal_pressure_Pa": 120e6, "mean_shear_stress_Pa": 75e6,
    "std_shear_stress_Pa": 5e6, "W_m": 15000., "transition_width_m": 3000.,
    "fault_lengths_m": [72000., 36000.], "nucleation_radius_m": 3000.,
    "nucleation_amplitude_Pa": 25e6, "nucleation_rise_s": 1.,
    "fractal_dimensions_training": [1.2, 1.5, 1.6],
    "fractal_dimension_holdout": 1.3,
    "b_training": [0.012, 0.014],
    "a0_delta_a0_pairs_training": [[0.009, 0.006], [0.008, 0.008], [0.007, 0.010],
                                  [0.006, 0.012], [0.0085, 0.007], [0.0075, 0.009],
                                  [0.0065, 0.011]],
    "a0_delta_a0_pairs_unseen": [[0.0087, 0.007], [0.0077, 0.009], [0.0067, 0.011]],
    "input_velocity_thresholds_m_s": [0., 1e-4, 1e-3, 1e-2]
}


def smooth_box(coordinate, halfwidth, transition):
    x = np.abs(np.asarray(coordinate, dtype=float))
    result = np.zeros_like(x)
    result[x <= halfwidth] = 1.
    mask = (x > halfwidth) & (x < halfwidth+transition)
    z = x[mask]-halfwidth
    result[mask] = 0.5*(1+np.tanh(transition/(z-transition)+transition/z))
    return result


def direct_effect_field(x, y=0., a0=0.008, delta_a0=0.008, W=15000., width=3000.):
    return a0+delta_a0*(1-smooth_box(x, W, width)*smooth_box(y, W/2, width))


def fractal_field(shape, dimension, seed, mean=75e6, std=5e6):
    """Real periodic field with spectral amplitude |k|^{-(2.5-D)}.

    Gaussian white-noise FFT supplies Hermitian unit phases. Radius uses normalized
    signed FFT frequencies. A 1D line or 2D plane can be generated. The DC value
    is zero before exact centering and population-standard-deviation scaling.
    """
    if len(shape) not in (1, 2) or min(shape) < 3:
        raise ValueError("shape must be a 1D/2D grid, each dimension >= 3")
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fftn(rng.normal(size=shape))
    phases = spectrum/np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    radius = np.sqrt(sum(g*g for g in grids))
    amplitude = np.where(radius > 0, np.maximum(radius, 1e-100)**(-(2.5-dimension)), 0.)
    raw_complex = np.fft.ifftn(phases*amplitude)
    raw = raw_complex.real
    field = (raw-raw.mean())/raw.std()*std+mean
    return field


def spatial_nucleation(radius, maximum_radius=3000.):
    radius = np.asarray(radius, dtype=float)
    result = np.zeros_like(radius)
    mask = np.abs(radius) < maximum_radius
    result[mask] = np.exp(radius[mask]**2/(radius[mask]**2-maximum_radius**2))
    return result


def temporal_nucleation(time, rise_time=1.):
    time = np.asarray(time, dtype=float)
    result = np.zeros_like(time)
    result[time >= rise_time] = 1.
    mask = (time > 0) & (time < rise_time)
    result[mask] = np.exp((time[mask]-rise_time)**2/(time[mask]*(time[mask]-2*rise_time)))
    return result


def source_case(shape=(128,), dimension=1.5, seed=1, a0=0.008, delta_a0=0.008,
                b=0.012, hypocenter=(0., 0.)):
    from fault_pinet import initial_state
    x = np.linspace(-36000., 36000., shape[0], endpoint=False)
    if len(shape) == 1:
        xx, yy = x, np.zeros_like(x)
    else:
        y = np.linspace(-18000., 18000., shape[1], endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="ij")
    aa = direct_effect_field(xx, yy, a0, delta_a0)
    parameters = np.stack([aa, np.full(shape, b), np.full(shape, 0.02), np.full(shape, 0.6)], axis=-1)
    stress = fractal_field(shape, dimension, seed)
    theta = initial_state(stress, 1e-12, 120e6, parameters)
    radius = np.hypot(xx-hypocenter[0], yy-hypocenter[1])
    return {"x_m": xx, "y_m": yy, "parameters_a_b_L_f": parameters,
            "initial_stress_Pa": stress, "initial_theta_s": theta,
            "nucleation_spatial_shape": spatial_nucleation(radius)}
