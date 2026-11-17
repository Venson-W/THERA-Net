"""Meaningful constitutive, differentiation, initialization and energy checks."""
import unittest
import numpy as np
from fault_pinet import (FaultPiNet, traction, potential, tangent, aging_update,
    true_parameters, initial_state, spring_slider, generate_dataset)
from source_cases import (smooth_box, fractal_field, source_case,
    spatial_nucleation, temporal_nucleation)


class TestFaultPiNet(unittest.TestCase):
    def setUp(self):
        self.parameters=true_parameters(0.2)

    def test_potential_derivative_and_convexity(self):
        velocities=np.concatenate([-np.logspace(-12,1,40),np.logspace(-12,1,40)])
        step=np.abs(velocities)*1e-5
        finite=(potential(velocities+step,2e4,120e6,self.parameters)
                -potential(velocities-step,2e4,120e6,self.parameters))/(2*step)
        np.testing.assert_allclose(finite,traction(velocities,2e4,120e6,self.parameters),rtol=2e-8)
        self.assertTrue(np.all(tangent(velocities,2e4,120e6,self.parameters)>0))
        self.assertTrue(np.all(potential(velocities,2e4,120e6,self.parameters)>=0))
        self.assertTrue(np.all(traction(velocities,2e4,120e6,self.parameters)*velocities>=0))
        self.assertEqual(float(potential(0.,2e4,120e6,self.parameters)),0.)
        self.assertEqual(float(traction(0.,2e4,120e6,self.parameters)),0.)

    def test_traction_parameter_gradient(self):
        velocity=-0.003; theta=23400.
        _,analytic=traction(velocity,theta,1.,self.parameters,jacobian=True)
        for j in range(4):
            step=self.parameters[j]*1e-5
            plus=self.parameters.copy(); minus=self.parameters.copy()
            plus[j]+=step; minus[j]-=step
            fd=(traction(velocity,theta,1.,plus)-traction(velocity,theta,1.,minus))/(2*step)
            self.assertAlmostEqual(float(fd/analytic[j]),1.,places=6)

    def test_vector_pressure_and_zero_pressure(self):
        params=true_parameters(np.array([-.4,.4]))
        value,derivative=traction(np.array([-.01,.01]),np.array([10.,10.]),
                                  np.array([0.,120e6]),params,jacobian=True)
        self.assertEqual(value.shape,(2,))
        self.assertEqual(derivative.shape,(2,4))
        self.assertEqual(value[0],0.)
        np.testing.assert_array_equal(derivative[0],0.)

    def test_positive_state_and_exact_limits(self):
        np.testing.assert_allclose(aging_update(3.,0.,7.,.02),10.,atol=0,rtol=0)
        np.testing.assert_allclose(aging_update(3.,5.,0.,.02),3.,atol=0,rtol=0)
        v=np.logspace(-15,4,100); length=.02; dt=.1
        theta=aging_update(3.,v,dt,length)
        self.assertTrue(np.all(theta>0))
        _,analytic=aging_update(3.,v,dt,length,jacobian=True)
        step=length*1e-5
        fd=(aging_update(3.,v,dt,length+step)-aging_update(3.,v,dt,length-step))/(2*step)
        np.testing.assert_allclose(analytic,fd,rtol=2e-4,atol=1e-8)
        v=.05; dt=.3
        once=aging_update(8.,v,dt,length)
        twice=aging_update(aging_update(8.,v,dt/2,length),v,dt/2,length)
        self.assertAlmostEqual(float(once),float(twice),places=13)

    def test_joint_loss_network_gradient(self):
        model=FaultPiNet(14,hidden=(4,))
        data=generate_dataset(np.array([-.7,.2]),5,11)
        vector=model.network.pack().copy()
        _,analytic=model.loss_and_gradient(vector,data)
        for j in range(len(vector)):
            step=1e-5
            plus=vector.copy();minus=vector.copy()
            plus[j]+=step;minus[j]-=step
            fd=(model.loss_and_gradient(plus,data)[0]-model.loss_and_gradient(minus,data)[0])/(2*step)
            np.testing.assert_allclose(analytic[j],fd,rtol=1e-5,atol=2e-8)

    def test_initial_state_inverts_friction(self):
        for shape in [(64,),(32,16)]:
            case=source_case(shape,seed=13)
            result=traction(1e-12,case["initial_theta_s"],120e6,case["parameters_a_b_L_f"])
            np.testing.assert_allclose(result,case["initial_stress_Pa"],rtol=3e-15)

    def test_fractal_statistics_and_determinism(self):
        for shape in [(129,),(64,32)]:
            field=fractal_field(shape,1.3,22)
            self.assertAlmostEqual(float(field.mean()),75e6,places=7)
            self.assertAlmostEqual(float(field.std()),5e6,places=7)
            np.testing.assert_array_equal(field,fractal_field(shape,1.3,22))
            spec=np.fft.fftn(field-field.mean())
            # The real inverse's complex spectrum has conjugate partner symmetry.
            partner=spec
            for axis,n in enumerate(shape): partner=np.take(partner,(-np.arange(n))%n,axis=axis)
            np.testing.assert_allclose(spec,np.conj(partner),atol=1e-6)

    def test_source_transition_and_nucleation(self):
        x=np.linspace(0,20,2000)
        box=smooth_box(x,10,3)
        self.assertTrue(np.all(np.diff(box)<=0))
        self.assertTrue(np.all((box>=0)&(box<=1)))
        np.testing.assert_allclose(smooth_box(np.array([0,10,13,20]),10,3),[1,1,0,0])
        np.testing.assert_allclose(spatial_nucleation(np.array([0.,3000.,4000.])),[1,0,0])
        np.testing.assert_allclose(temporal_nucleation(np.array([-1.,0.,1.,2.])),[0,0,1,1])

    def test_backward_euler_energy_identity(self):
        result=spring_slider(self.parameters,dt=.02,end=2.)
        self.assertTrue(np.all(result[:,7]>=0))
        self.assertTrue(np.all(result[:,8]>=0))
        scale=max(np.max(np.abs(result[:,6])),1.)
        self.assertLess(float(np.max(np.abs(result[:,9]))/scale),1e-12)
        self.assertTrue(np.all(result[:,3]>0))

    def test_material_bounds_outside_training_domain(self):
        from fault_pinet import LOWER, UPPER
        params=FaultPiNet(12).parameters(np.linspace(-100,100,1000))
        self.assertTrue(np.all(params>LOWER))
        self.assertTrue(np.all(params<UPPER))

    def test_invalid_state_rejected(self):
        with self.assertRaises(ValueError): traction(.1,-1.,120e6,self.parameters)
        with self.assertRaises(ValueError): aging_update(1.,.1,-1.,.02)
        with self.assertRaises(ValueError): traction(.1,np.nan,120e6,self.parameters)
        with self.assertRaises(ValueError): traction(np.inf,1.,120e6,self.parameters)

    def test_extreme_log_asinh_is_finite(self):
        parameters=np.array([.004,.020,.008,.75])
        # This state makes V/c overflow in a naive exp implementation.
        value=traction(1.,1e100,120e6,parameters)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(potential(1.,1e100,120e6,parameters)))
        self.assertTrue(np.isfinite(tangent(1.,1e100,120e6,parameters)))
        self.assertEqual(float(potential(0.,1e100,120e6,parameters)),0.)


if __name__=="__main__": unittest.main(verbosity=2)
