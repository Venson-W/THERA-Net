"""THERA-Net: thermodynamically admissible neural friction modeling in NumPy.

SI units throughout: V [m/s], theta [s], L [m], p/tau [Pa], phi [W/m^2].
Only fixed material descriptors enter the parameter network. The network must
not receive instantaneous slip rate: that would invalidate the convexity proof.
The interface free energy is zero; theta is phenomenological kinetic memory.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

VREF = 1e-6
PARAMETER_NAMES = ("a", "b", "L_m", "f_ref")
LOWER = np.array([0.004, 0.006, 0.008, 0.45])
UPPER = np.array([0.024, 0.020, 0.040, 0.75])


def _positive_inputs(theta, pressure, parameters):
    if any(np.any(~np.isfinite(np.asarray(value))) for value in [theta, pressure, parameters]):
        raise ValueError("state, pressure and parameters must be finite")
    if np.any(np.asarray(theta) <= 0) or np.any(np.asarray(pressure) < 0):
        raise ValueError("theta must be positive and compression-positive pressure nonnegative")
    if np.any(np.asarray(parameters) <= 0):
        raise ValueError("a, b, L and f_ref must be positive")


def characteristic_speed(theta, parameters):
    """Positive regularization speed c(theta); parameters end in [a,b,L,f]."""
    a, b, length, friction = np.moveaxis(np.asarray(parameters), -1, 0)
    return 2 * VREF * np.exp(-(friction + b * np.log(VREF * theta / length)) / a)


def _log_rsf(velocity, theta, parameters):
    """Evaluate signed asinh(V/c) in log space, without exponential overflow."""
    velocity, theta = np.asarray(velocity), np.asarray(theta)
    if np.any(~np.isfinite(velocity)):
        raise ValueError("velocity must be finite")
    a, b, length, friction = np.moveaxis(np.asarray(parameters), -1, 0)
    log_state = np.log(VREF) + np.log(theta) - np.log(length)
    exponent = (friction + b*log_state)/a
    with np.errstate(divide="ignore"):
        log_ratio = np.log(np.abs(velocity))-np.log(2*VREF)+exponent
    positive_log = np.maximum(log_ratio, 0.)
    large = positive_log+np.log1p(np.sqrt(1+np.exp(-2*positive_log)))
    small = np.arcsinh(np.exp(np.minimum(log_ratio, 0.)))
    inverse = np.sign(velocity)*np.where(log_ratio >= 0, large, small)
    return inverse, exponent, log_state, log_ratio


def traction(velocity, theta, pressure, parameters, jacobian=False):
    """Signed resisting traction, with tau*V >= 0 (not traction on the + face)."""
    _positive_inputs(theta, pressure, parameters)
    velocity, theta, pressure = np.asarray(velocity), np.asarray(theta), np.asarray(pressure)
    a, b, length, friction = np.moveaxis(np.asarray(parameters), -1, 0)
    inverse, exponent, log_state, _ = _log_rsf(velocity, theta, parameters)
    result = pressure * a * inverse
    if not jacobian:
        return result
    reduced = np.tanh(inverse)
    derivative = pressure[..., None] * np.stack(
        [inverse - reduced * exponent, reduced * log_state,
         -reduced * b / length, reduced], axis=-1)
    return result, derivative


def potential(velocity, theta, pressure, parameters):
    """Convex dissipation potential, even in V and zero at V=0.

    The hyperbolic identity avoids exponential overflow and zero/zero at rest.
    phi is not the frictional power tau*V and not accumulated dissipated energy.
    """
    _positive_inputs(theta, pressure, parameters)
    velocity, theta, pressure = np.asarray(velocity), np.asarray(theta), np.asarray(pressure)
    a = np.asarray(parameters)[..., 0]
    inverse, _, _, _ = _log_rsf(velocity, theta, parameters)
    return pressure * a * velocity * (inverse - np.tanh(inverse/2))


def tangent(velocity, theta, pressure, parameters):
    """d tau / d V > 0 for p>0, at fixed theta and material descriptor."""
    _positive_inputs(theta, pressure, parameters)
    _, exponent, _, log_ratio = _log_rsf(velocity, theta, parameters)
    with np.errstate(divide="ignore"):
        log_value = (np.log(pressure)+np.log(np.asarray(parameters)[...,0])
                     +exponent-np.log(2*VREF)-0.5*np.logaddexp(0.,2*log_ratio))
    return np.exp(log_value)


def aging_update(theta, velocity, dt, length, jacobian=False):
    """Exact positive state update for frozen |V|, including V=0 and dt=0."""
    theta, velocity, dt, length = np.broadcast_arrays(theta, velocity, dt, length)
    if any(np.any(~np.isfinite(value)) for value in [theta, velocity, dt, length]):
        raise ValueError("state-update inputs must be finite")
    if np.any(theta <= 0) or np.any(dt < 0) or np.any(length <= 0):
        raise ValueError("theta,L > 0 and dt >= 0 required")
    z = np.abs(velocity) * dt / length
    small = z < 1e-5
    zsafe = np.where(small, 1.0, z)
    decay = np.exp(-z)
    e = np.where(small, 1-z/2+z*z/6-z**3/24, -np.expm1(-z)/zsafe)
    result = theta * decay + dt * e
    if not jacobian:
        return result
    de = np.where(small, -0.5+z/3-z*z/8+z**3/30,
                  ((z+1)*decay-1)/zsafe**2)
    derivative = z / length * (theta * decay - dt * de)
    return result, derivative


def steady_traction(speed, pressure, parameters):
    """Positive steady sliding: theta_ss = Dc/speed."""
    speed = np.asarray(speed)
    if not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("finite positive steady speed required")
    return traction(speed, np.asarray(parameters)[..., 2]/speed, pressure, parameters)


def steady_log_tangent(speed, pressure, parameters):
    """d tau_ss / d log(speed); a < b gives steady velocity weakening."""
    steady_traction(speed, pressure, parameters)  # Validate the constitutive domain.
    a, b, length, _ = np.moveaxis(np.asarray(parameters), -1, 0)
    inverse, _, _, _ = _log_rsf(speed, length/np.asarray(speed), parameters)
    return np.asarray(pressure)*(a-b)*np.tanh(inverse)


def initial_state(initial_traction, initial_velocity, pressure, parameters):
    """Invert regularized RSF at positive source initial velocity and traction.

    Using log(sinh(x)) avoids overflow. Negative/zero traction is incompatible
    with the strictly positive initial velocity used by the source cases.
    """
    _positive_inputs(1.0, pressure, parameters)
    if np.any(np.asarray(initial_traction) <= 0) or initial_velocity <= 0 or np.any(np.asarray(pressure) <= 0):
        raise ValueError("positive initial traction, velocity and pressure required")
    a, b, length, friction = np.moveaxis(np.asarray(parameters), -1, 0)
    z = initial_traction / (a * pressure)
    log_sinh = z - np.log(2.0) + np.log(-np.expm1(-2*z))
    logtheta = np.log(length/VREF) + (a*(np.log(2*VREF/initial_velocity)+log_sinh)-friction)/b
    return np.exp(logtheta)


class MLP:
    """Small tanh MLP with analytic reverse-mode gradients; no ML runtime needed."""
    def __init__(self, layers, seed=0):
        self.layers = tuple(layers)
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2/(i+o)), (i, o))
                        for i, o in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros(o) for o in layers[1:]]

    def forward(self, inputs, cache=False):
        values = [np.asarray(inputs)]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = values[-1] @ weight + bias
            values.append(np.tanh(value) if i < len(self.weights)-1 else value)
        return (values[-1], values) if cache else values[-1]

    def backward(self, values, output_gradient):
        gw, gb = [], []
        gradient = output_gradient
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights)-1:
                gradient = gradient * (1-values[i+1]**2)
            gw.append(values[i].T @ gradient)
            gb.append(gradient.sum(axis=0))
            gradient = gradient @ self.weights[i].T
        return np.concatenate([q.ravel() for pair in zip(gw[::-1], gb[::-1]) for q in pair])

    def pack(self):
        return np.concatenate([q.ravel() for pair in zip(self.weights, self.biases) for q in pair])

    def unpack(self, vector):
        offset = 0
        for weight, bias in zip(self.weights, self.biases):
            for array in [weight, bias]:
                size = array.size
                array[:] = vector[offset:offset+size].reshape(array.shape)
                offset += size


class THERANet:
    """Descriptor -> bounded constitutive parameters -> exact physics operators."""
    def __init__(self, seed=0, hidden=(12, 12)):
        self.network = MLP([1, *hidden, 4], seed)

    def parameters(self, descriptor, cache=False):
        raw, values = self.network.forward(np.asarray(descriptor).reshape(-1, 1), cache=True)
        logistic = 1 / (1 + np.exp(-raw))
        parameters = LOWER + (UPPER-LOWER)*logistic
        return (parameters, values, logistic) if cache else parameters

    def loss_and_gradient(self, vector, data):
        self.network.unpack(vector)
        parameters, values, logistic = self.parameters(data["descriptor"], cache=True)
        mu, jmu = traction(data["velocity"], data["theta"], 1.0, parameters, jacobian=True)
        newtheta, jstate = aging_update(data["theta"], data["velocity"], data["dt"],
                                        parameters[:, 2], jacobian=True)
        # Fixed physical scales, not statistics fitted using held-out samples.
        e_mu = (mu-data["mu"])/0.1
        e_state = np.log(newtheta/data["theta_next"])
        n = len(mu)
        loss = np.mean(e_mu**2 + e_state**2)
        gradient = (2/n)*e_mu[:, None]*jmu/0.1
        gradient[:, 2] += (2/n)*e_state*jstate/newtheta
        raw_gradient = gradient*(UPPER-LOWER)*logistic*(1-logistic)
        return float(loss), self.network.backward(values, raw_gradient)

    def save(self, filename):
        np.savez(filename, vector=self.network.pack(), layers=self.network.layers,
                 lower=LOWER, upper=UPPER)

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        model = cls(hidden=tuple(data["layers"][1:-1]))
        model.network.unpack(data["vector"])
        return model
