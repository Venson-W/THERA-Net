"""Data-independent constitutive, coupling, input, and diagnostic checks."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from model import (THERANet, LOWER, UPPER, traction, potential, tangent, aging_update,
                   initial_state, steady_traction, steady_log_tangent)
from material import source_family_parameters, load_source_model
from train import generate
from dynamics import impedance_velocity, stage_potential, predictor_corrector_step, spring_slider
from cases import fields, smooth_box, fractal_field, spatial_nucleation, temporal_nucleation, write_inputs
from diagnostics import (relative_l2, matching_indices, arrival_times, arrival_metrics,
                         moment_rate, integrate_power, region_masks, receiver_fraction,
                         receiver_status, load_run, compare_runs)

PARAMETERS = np.array([.008, .012, .02, .6])
PRESSURE = 120e6
IMPEDANCE = 2670.*3464./2.


class ConstitutiveTests(unittest.TestCase):
    def test_potential_derivative_convexity_and_dissipation(self):
        v = np.r_[-np.logspace(-12, 1, 40), np.logspace(-12, 1, 40)]
        h = abs(v)*1e-5
        fd = (potential(v+h, 2e4, PRESSURE, PARAMETERS)-potential(v-h, 2e4, PRESSURE, PARAMETERS))/(2*h)
        np.testing.assert_allclose(fd, traction(v, 2e4, PRESSURE, PARAMETERS), rtol=2e-8)
        self.assertTrue(np.all(tangent(v, 2e4, PRESSURE, PARAMETERS) > 0))
        self.assertTrue(np.all(potential(v, 2e4, PRESSURE, PARAMETERS) >= 0))
        self.assertTrue(np.all(traction(v, 2e4, PRESSURE, PARAMETERS)*v >= 0))
        self.assertEqual(float(potential(0., 1., PRESSURE, PARAMETERS)), 0.)

    def test_velocity_tangent(self):
        v = np.logspace(-10, 0, 30)
        h = v*1e-5
        fd = (traction(v+h, 3., PRESSURE, PARAMETERS)-traction(v-h, 3., PRESSURE, PARAMETERS))/(2*h)
        np.testing.assert_allclose(fd, tangent(v, 3., PRESSURE, PARAMETERS), rtol=2e-8)

    def test_parameter_jacobian(self):
        _, gradient = traction(-.003, 23400., 1., PARAMETERS, jacobian=True)
        for j in range(4):
            step = PARAMETERS[j]*1e-5
            plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
            plus[j] += step
            minus[j] -= step
            fd = (traction(-.003, 23400., 1., plus)-traction(-.003, 23400., 1., minus))/(2*step)
            np.testing.assert_allclose(fd, gradient[j], rtol=1e-7)

    def test_zero_pressure_and_signed_traction(self):
        value, derivative = traction(np.array([-.01, .01]), 10., np.array([0., PRESSURE]), PARAMETERS, jacobian=True)
        self.assertEqual(value[0], 0.)
        np.testing.assert_array_equal(derivative[0], 0.)
        self.assertEqual(traction(-.01, 10., PRESSURE, PARAMETERS), -traction(.01, 10., PRESSURE, PARAMETERS))

    def test_state_limits_semigroup_and_derivative(self):
        self.assertEqual(float(aging_update(3., 0., 7., .02)), 10.)
        self.assertEqual(float(aging_update(3., 5., 0., .02)), 3.)
        v = np.logspace(-15, 4, 100)
        value, gradient = aging_update(3., v, .1, .02, jacobian=True)
        self.assertTrue(np.all(value > 0))
        h = 2e-7
        fd = (aging_update(3., v, .1, .02+h)-aging_update(3., v, .1, .02-h))/(2*h)
        np.testing.assert_allclose(gradient, fd, rtol=2e-4, atol=1e-8)
        self.assertAlmostEqual(float(aging_update(8., .05, .3, .02)),
            float(aging_update(aging_update(8., .05, .15, .02), .05, .15, .02)), places=13)

    def test_steady_velocity_weakening(self):
        speed = np.logspace(-9, 0, 30)
        h = 1e-5
        fd = (steady_traction(speed*np.exp(h), PRESSURE, PARAMETERS)
              - steady_traction(speed*np.exp(-h), PRESSURE, PARAMETERS))/(2*h)
        slope = steady_log_tangent(speed, PRESSURE, PARAMETERS)
        np.testing.assert_allclose(fd, slope, rtol=2e-8)
        self.assertTrue(np.all(slope < 0))
        stronger = PARAMETERS.copy()
        stronger[0] = .016
        self.assertTrue(np.all(steady_log_tangent(speed, PRESSURE, stronger) > 0))

    def test_initial_state_inversion(self):
        for case in ("uniform", "heterogeneous", "unseen", "barrier"):
            data = fields(180, case)
            np.testing.assert_allclose(traction(1e-12, data["theta"], PRESSURE, data["parameters"]),
                                       data["prestress"], rtol=3e-15)

    def test_full_network_gradient(self):
        model = THERANet(14, hidden=(4,))
        data = generate(np.array([.007, .015]), 5, 11)
        vector = model.network.pack().copy()
        _, gradient = model.loss_and_gradient(vector, data)
        for j in range(len(vector)):
            plus, minus = vector.copy(), vector.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            fd = (model.loss_and_gradient(plus, data)[0]-model.loss_and_gradient(minus, data)[0])/2e-5
            np.testing.assert_allclose(fd, gradient[j], rtol=2e-5, atol=1e-6)

    def test_bounds_checkpoint_and_export(self):
        model = THERANet(12)
        self.assertEqual(len(model.network.pack()), 232)
        parameters = model.parameters(np.linspace(-100, 100, 1000))
        self.assertTrue(np.all(parameters > LOWER) and np.all(parameters < UPPER))
        data = fields(24)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.npz"
            model.save(path)
            loaded = load_source_model(path)
            np.testing.assert_array_equal(model.parameters([.2]), loaded.parameters([.2]))
            files = write_inputs(folder, data, loaded)
            reference = np.fromfile(files["reference"], dtype="<f8").reshape(24, 12, 7)
            predicted = np.fromfile(files["thera"], dtype="<f8").reshape(24, 12, 7)
            np.testing.assert_array_equal(predicted[..., 4:], reference[..., 4:])
            np.testing.assert_array_equal(predicted[..., :4], source_family_parameters(model, reference[..., 0]))
            self.assertGreater(float(np.max(abs(predicted[..., :4]-reference[..., :4]))), 0.)

    def test_extreme_states_and_invalid_inputs(self):
        q = np.array([.004, .020, .008, .75])
        for function in (traction, potential, tangent):
            self.assertTrue(np.isfinite(function(1., 1e100, PRESSURE, q)))
        self.assertEqual(float(potential(0., 1e100, PRESSURE, q)), 0.)
        for state in (-1., np.nan):
            with self.assertRaises(ValueError): traction(.1, state, PRESSURE, q)
        with self.assertRaises(ValueError): aging_update(1., .1, -1., .02)
        with self.assertRaises(ValueError): steady_traction(0., PRESSURE, q)


class CouplingTests(unittest.TestCase):
    def test_signed_impedance_roots_against_independent_solver(self):
        for state in (.001, 1., 1e12, 1e24):
            for driving in (-90e6, -1e5, 0., 1e5, 90e6):
                actual = float(impedance_velocity(driving, state, PRESSURE, PARAMETERS, IMPEDANCE))
                expected = brentq(lambda v: IMPEDANCE*v+traction(v, state, PRESSURE, PARAMETERS)-driving,
                    -abs(driving)/IMPEDANCE-1e-12, abs(driving)/IMPEDANCE+1e-12, xtol=5e-324, maxiter=2000)
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-19)
                self.assertLess(abs(IMPEDANCE*actual+traction(actual, state, PRESSURE, PARAMETERS)-driving), 2e-5)

    def test_stage_minimum_and_local_error_bound(self):
        driving, state = 75e6, 100.
        velocity = float(impedance_velocity(driving, state, PRESSURE, PARAMETERS, IMPEDANCE))
        minimum = stage_potential(velocity, driving, state, PRESSURE, PARAMETERS, IMPEDANCE)
        for delta in (-.1, -.01, .01, .1):
            self.assertGreater(stage_potential(velocity+delta, driving, state, PRESSURE, PARAMETERS, IMPEDANCE), minimum)
        altered = PARAMETERS.copy()
        altered[0] *= 1.01
        other = float(impedance_velocity(driving, state, PRESSURE, altered, IMPEDANCE))
        bound = abs(traction(velocity, state, PRESSURE, altered)-traction(velocity, state, PRESSURE, PARAMETERS))
        self.assertLessEqual(IMPEDANCE*abs(other-velocity), bound+1e-6)

    def test_zero_pressure_and_invalid_impedance(self):
        np.testing.assert_array_equal(impedance_velocity([-3., 0., 3.], 1., 0., PARAMETERS, 2.), [-1.5, 0., 1.5])
        with self.assertRaises(ValueError): impedance_velocity(1., 1., PRESSURE, PARAMETERS, 0.)

    def test_corrector_uses_old_state_and_updates_history_once(self):
        theta, velocity, dt = 1., .01, .02
        calls = []
        def history(slip):
            calls.append(float(slip))
            return -1e8*slip
        slip, new_velocity, state, new_history = predictor_corrector_step(0., velocity, theta, dt,
            75e6, 0., history, PRESSURE, PARAMETERS, IMPEDANCE)
        self.assertEqual(len(calls), 1)
        mean_velocity = slip/dt
        expected = aging_update(theta, mean_velocity, dt, PARAMETERS[2])
        self.assertEqual(float(state), float(expected))
        double_aged = aging_update(aging_update(theta, velocity, dt, PARAMETERS[2]), mean_velocity, dt, PARAMETERS[2])
        self.assertGreater(abs(float(state-double_aged)), 1e-6)
        self.assertLess(abs(IMPEDANCE*new_velocity+traction(new_velocity, state, PRESSURE, PARAMETERS)-75e6-new_history), 2e-5)

    def test_time_refinement_against_independent_ode(self):
        initial_velocity, initial_theta = .01, 1.
        driving = float(traction(initial_velocity, initial_theta, PRESSURE, PARAMETERS)+IMPEDANCE*initial_velocity)
        def derivative(t, y):
            v = brentq(lambda u: IMPEDANCE*u+traction(u, y[1], PRESSURE, PARAMETERS)-driving,
                       0., driving/IMPEDANCE, xtol=1e-15)
            return [v, 1-v*y[1]/PARAMETERS[2]]
        reference = solve_ivp(derivative, [0., .2], [0., initial_theta], rtol=1e-11, atol=1e-13).y[:, -1]
        errors = []
        for dt in (.04, .02, .01):
            slip, velocity, theta = 0., initial_velocity, initial_theta
            for _ in range(round(.2/dt)):
                slip, velocity, theta, _ = predictor_corrector_step(slip, velocity, theta, dt,
                    driving, 0., lambda _: 0., PRESSURE, PARAMETERS, IMPEDANCE)
            errors.append(np.linalg.norm(np.array([slip, theta])-reference))
        self.assertLess(errors[1], errors[0]/3.)
        self.assertLess(errors[2], errors[1]/3.)

    def test_slider_discrete_energy_balance(self):
        result = spring_slider(PARAMETERS, dt=.02, end=2.)
        self.assertTrue(np.all(result[:, 7:9] >= 0))
        self.assertTrue(np.all(result[:, 3] > 0))
        self.assertLess(float(np.max(abs(result[:, 9]))/max(np.max(abs(result[:, 6])), 1.)), 1e-12)


class SpatialTests(unittest.TestCase):
    def test_tapers_and_nucleation(self):
        box = smooth_box(np.linspace(0, 20, 2000), 10, 3)
        self.assertTrue(np.all(np.diff(box) <= 0))
        np.testing.assert_array_equal(smooth_box([0, 10, 13, 20], 10, 3), [1, 1, 0, 0])
        np.testing.assert_array_equal(spatial_nucleation([0, 3000, 4000]), [1, 0, 0])
        np.testing.assert_array_equal(temporal_nucleation([-1, 0, 1, 2]), [0, 0, 1, 1])

    def test_seeded_fractal_statistics(self):
        field = fractal_field((64, 32), 1.3, 22)
        self.assertAlmostEqual(float(field.mean()), 75e6, places=7)
        self.assertAlmostEqual(float(field.std()), 5e6, places=7)
        np.testing.assert_array_equal(field, fractal_field((64, 32), 1.3, 22))
        spectrum = np.fft.fftn(field-field.mean())
        partner = spectrum
        for axis, n in enumerate(field.shape):
            partner = np.take(partner, (-np.arange(n)) % n, axis=axis)
        np.testing.assert_allclose(spectrum, partner.conj(), atol=1e-6)

    def test_spatial_refinement_preserves_realization(self):
        for name in ("uniform", "heterogeneous", "unseen", "barrier"):
            coarse, fine = fields(180, name), fields(360, name)
            for key in ("parameters", "theta", "prestress", "nucleation"):
                np.testing.assert_array_equal(coarse[key], fine[key][::2, ::2])

    def test_barrier_contrast_and_no_band_control(self):
        data = fields(180, "barrier", 2., .014)
        self.assertTrue(np.all(data["nucleation"][data["x"] > 0] == 0))
        self.assertGreater(data["parameters"][90, 45, 0], data["parameters"][90, 45, 1])
        self.assertEqual(fields(180, "barrier", 0., .018)["parameters"][90, 45, 0], .008)
        with self.assertRaises(ValueError): fields(181)


class DiagnosticTests(unittest.TestCase):
    def test_masked_field_error_and_undefined_norm(self):
        reference = np.arange(1, 49, dtype=float).reshape(3, 4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        predicted = reference*1.01
        predicted[:, ~mask] = 1e20
        self.assertAlmostEqual(relative_l2(predicted, reference, mask), .01, places=14)
        with self.assertRaises(ValueError): relative_l2(np.ones(2), np.zeros(2))

    def test_censored_arrivals_and_first_crossing(self):
        velocity = np.array([[0., .7, 0.], [.4, .1, -.2], [.8, .1, -.4]])
        np.testing.assert_allclose(arrival_times(velocity, [0., 1., 2.]), [1.25, 0., -1.])
        metrics = arrival_metrics(np.array([[1.1, -1.], [3., 4.2]]),
                                  np.array([[1., 2.], [-1., 4.]]), np.ones((2, 2), dtype=bool))
        self.assertEqual(metrics["matched_reached_points"], 2)
        self.assertEqual(metrics["mask_disagreement_points"], 2)
        self.assertEqual(metrics["false_reached_points"], 1)
        self.assertEqual(metrics["missed_reached_points"], 1)
        self.assertAlmostEqual(metrics["RMSE_s"], np.sqrt(.025), places=14)

    def test_shared_samples_without_interpolation(self):
        np.testing.assert_array_equal(matching_indices([.02, .04], [.01, .02, .03, .04]), [1, 3])
        with self.assertRaises(ValueError): matching_indices([.025], [.01, .02, .03])

    def test_signed_moment_and_work(self):
        velocity = np.array([[[1., -2.], [3., -4.]], [[2., -4.], [6., -8.]]])
        np.testing.assert_array_equal(moment_rate(velocity, 2., 3.), [-12., -24.])
        np.testing.assert_allclose(integrate_power(np.array([-2., 3.]), [.1, .3]), .4)
        with self.assertRaises(ValueError): integrate_power([1., 2.], [.1, .1])

    def test_receiver_partition_and_classification(self):
        masks = region_masks(np.arange(-15000, 15001, 1000), np.arange(-7000, 7001, 1000), 1500.)
        parts = sum(masks[k].astype(int) for k in ("left", "barrier", "receiver"))
        np.testing.assert_array_equal(parts, masks["active"].astype(int))
        peaks = masks["receiver"].astype(float)
        self.assertEqual(receiver_fraction(peaks, masks["receiver"]), 1.)
        self.assertEqual([receiver_status(f) for f in (0., .01, .2, .5, 1.)],
                         ["arrested", "arrested", "partial", "activated", "activated"])

    def test_refinement_comparison_uses_common_grid(self):
        def run(n, scale):
            coords = np.linspace(0, 200, n)
            time = np.linspace(0, .04, n)
            diag = np.zeros(n-1, dtype=[("time", float), ("moment_rate_Nm_s", float)])
            diag["time"], diag["moment_rate_Nm_s"] = time[1:], scale
            return dict(x=coords, y=coords, time=time, diagnostics=diag,
                masks={"active": np.ones((n, n), dtype=bool)},
                velocity=np.full((n, n, n), scale), final=np.full((n, n, 5), scale),
                summary={"whole_fault_work_J": scale})
        result = compare_runs(run(3, 1.01), run(5, 1.))
        self.assertAlmostEqual(result["velocity_relative_L2"], .01, places=14)
        self.assertAlmostEqual(result["moment_rate_relative_L2"], .01, places=14)


@unittest.skipUnless(os.environ.get("THERA_RUPTURE_EXECUTABLE"), "external UGUCA adapter not configured")
class AdapterTests(unittest.TestCase):
    def run_adapter(self, folder, data, duration, dt, barrier=False):
        paths = write_inputs(folder, data)
        command = [os.environ["THERA_RUPTURE_EXECUTABLE"], str(len(data["x"])), str(duration),
                   str(dt), str(dt), str(paths["reference"]), "solution", "1"]
        if barrier:
            command.append("1500")
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return load_run(folder, snapshots=True)

    def test_uniform_external_step_matches_python(self):
        data = fields(24)
        data["parameters"][:] = PARAMETERS
        data["theta"][:] = 1.
        driving = float(traction(.01, 1., PRESSURE, PARAMETERS)+IMPEDANCE*.01)
        data["prestress"][:] = driving
        data["nucleation"][:] = 0.
        expected_slip, expected_v, expected_theta, _ = predictor_corrector_step(0., .01, 1., .01,
            driving, 0., lambda _: 0., PRESSURE, PARAMETERS, IMPEDANCE)
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_adapter(folder, data, .01, .01)
            np.testing.assert_array_equal(run["final"][..., 0], 0.)
            np.testing.assert_allclose(run["final"][..., 3], expected_theta, rtol=1e-10)
            np.testing.assert_allclose(run["final"][..., 4], expected_slip, rtol=1e-10)
            np.testing.assert_allclose(run["velocity"][-1], expected_v, rtol=1e-6)

    def test_barrier_outputs_and_incomplete_run_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_adapter(folder, fields(24, "barrier"), .2, .01, barrier=True)
            self.assertTrue(run["meta"]["barrier_enabled"])
            self.assertEqual(len(run["diagnostics"]), 20)
            path = Path(folder)/"solution_diagnostics.csv"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaises(ValueError): load_run(folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
