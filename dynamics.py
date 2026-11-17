"""Local impedance coupling and reduced mechanical verification (SI units)."""
import numpy as np
from model import traction, potential, aging_update, initial_state, VREF


def stage_potential(velocity, driving, theta, pressure, parameters, impedance):
    """J(v) = eta*v**2/2 + Phi(v) - S*v for a frozen constitutive stage."""
    if np.any(np.asarray(impedance) <= 0):
        raise ValueError("impedance must be positive")
    return (0.5*np.asarray(impedance)*np.asarray(velocity)**2
            + potential(velocity, theta, pressure, parameters) - np.asarray(driving)*velocity)


def impedance_velocity(driving, theta, pressure, parameters, impedance, previous=1e-12):
    """Solve eta*V + tau(V,theta) = S using safeguarded Newton in log(|V|).

    Broadcasts over material fields. The force tolerance is 1e-7 + 1e-13*|S| Pa.
    Elastic history and applied loading must already be included in S.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape[-1:] != (4,):
        raise ValueError("parameters must end in four coefficients")
    a, b, length, friction = np.moveaxis(parameters, -1, 0)
    arrays = np.broadcast_arrays(driving, theta, pressure, impedance, previous, a, b, length, friction)
    if any(np.any(~np.isfinite(v)) for v in arrays):
        raise ValueError("finite inputs required")
    if (np.any(arrays[1] <= 0) or np.any(arrays[2] < 0) or np.any(arrays[3] <= 0)
            or np.any(parameters <= 0)):
        raise ValueError("theta, impedance and coefficients must be positive; pressure nonnegative")
    output = np.empty(arrays[0].shape)
    for index in np.ndindex(output.shape):
        S, state, p, eta, old, aa, bb, dc, ff = (float(v[index]) for v in arrays)
        if S == 0 or p == 0:
            output[index] = S/eta
            continue
        target = abs(S)
        log_beta = (ff + bb*(np.log(VREF)+np.log(state)-np.log(dc)))/aa - np.log(2*VREF)
        lo, hi = np.log(np.nextafter(0., 1.)), np.log(target)-np.log(eta)
        if hi < lo:
            output[index] = 0.
            continue
        q = np.clip(np.log(max(abs(old), 1e-20)), lo, hi)
        for _ in range(100):
            u = log_beta+q
            y = u+np.log(2.) if u > 30 else np.arcsinh(np.exp(u))
            dy = 1. if u > 30 else np.exp(u)/np.hypot(1., np.exp(u))
            v = np.exp(q)
            residual = p*aa*y + eta*v - target
            if abs(residual) < 1e-7 + 1e-13*target:
                output[index] = np.copysign(v, S)
                break
            if residual > 0:
                hi = q
            else:
                lo = q
            derivative = p*aa*dy + eta*v
            candidate = q-residual/derivative if derivative > 0 else np.nan
            if not np.isfinite(candidate) or candidate <= lo or candidate >= hi:
                candidate = 0.5*(lo+hi)
            q = candidate
        else:
            raise RuntimeError("impedance root did not converge")
    return output


def predictor_corrector_step(slip, velocity, theta, dt, driving, old_history,
                            update_history, pressure, parameters, impedance):
    """One predictor/corrector stage with an externally supplied elastic operator.

    update_history(new_slip) must return the new causal elastic traction. It is
    called once, after correcting slip. Both state updates start from old theta.
    This routine does not approximate or replace the external wave-history solver.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    length = np.asarray(parameters)[..., 2]
    predicted_state = aging_update(theta, velocity, dt, length)
    predicted_velocity = impedance_velocity(np.asarray(driving)+old_history,
        predicted_state, pressure, parameters, impedance, previous=velocity)
    mean_velocity = 0.5*(np.asarray(velocity)+predicted_velocity)
    new_slip = np.asarray(slip)+dt*mean_velocity
    new_state = aging_update(theta, mean_velocity, dt, length)
    new_history = np.asarray(update_history(new_slip))
    new_velocity = impedance_velocity(np.asarray(driving)+new_history,
        new_state, pressure, parameters, impedance, previous=predicted_velocity)
    return new_slip, new_velocity, new_state, new_history


def spring_slider(parameters, dt=0.005, end=5.0, mass=267000., stiffness=1e8,
                  radiation=2670*3464/2, pressure=120e6, initial_stress=75e6):
    """Inertial single-degree-of-freedom verification, NOT a rupture/SBI solver.

    Implicit velocity/elasticity/friction uses old theta. Exact frozen-new-V aging
    follows. Energy accounting includes backward-Euler numerical dissipation.
    """
    from cases import temporal_nucleation
    if (not all(np.isfinite(x) for x in (dt, end, mass, stiffness, radiation, pressure, initial_stress))
            or min(dt, end, mass, stiffness, pressure, initial_stress) <= 0 or radiation < 0):
        raise ValueError("positive time, mass, stiffness, pressure and stress required")
    if not np.isclose(round(end/dt)*dt, end, rtol=0, atol=1e-12):
        raise ValueError("end must be an integer number of time steps")
    times = np.arange(int(round(end/dt))+1)*dt
    velocity, slip = 1e-12, 0.
    state = float(initial_state(initial_stress, velocity, pressure, parameters))
    energy = 0.5*mass*velocity**2
    rows = [[0., velocity, slip, state, float(traction(velocity, state, pressure, parameters)),
             energy, 0., 0., 0., 0.]]
    for time in times[1:]:
        forcing = initial_stress + 25e6*float(temporal_nucleation(time, 1.0))
        rhs = forcing-stiffness*slip+mass*velocity/dt
        linear = mass/dt+stiffness*dt+radiation
        bound = abs(rhs)/linear + 1e-12
        lo, hi = -bound, bound
        for _ in range(64):
            vnew = (lo+hi)/2
            residual = linear*vnew + float(traction(vnew, state, pressure, parameters))-rhs
            if residual > 0:
                hi = vnew
            else:
                lo = vnew
        vnew = (lo+hi)/2
        snew = slip+dt*vnew
        tau = float(traction(vnew, state, pressure, parameters))
        enew = 0.5*mass*vnew**2+0.5*stiffness*snew**2
        work = dt*forcing*vnew
        physical = dt*(radiation*vnew**2+tau*vnew)
        numerical = 0.5*mass*(vnew-velocity)**2+0.5*stiffness*(snew-slip)**2
        balance = enew-energy+physical+numerical-work
        state = float(aging_update(state, vnew, dt, parameters[2]))
        rows.append([time, vnew, snew, state, tau, enew, work, physical, numerical, balance])
        velocity, slip, energy = vnew, snew, enew
    return np.asarray(rows)


SLIDER_COLUMNS = "time_s,velocity_m_s,slip_m,theta_s,traction_Pa,energy_J_m2,work_increment_J_m2,physical_dissipation_increment_J_m2,numerical_dissipation_increment_J_m2,balance_residual_J_m2"
