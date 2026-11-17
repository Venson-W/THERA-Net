"""Checks for actual spatial-output analysis plus metric and censoring semantics."""
import unittest
import json
import numpy as np
from postprocess_rupture import (OUTPUT,arrival_metrics,field_relative_l2,
                                 matching_indices,load_compact,run_audit)


class TestRupturePostprocess(unittest.TestCase):
    def test_relative_field_norm_and_region(self):
        reference=np.arange(1,49,dtype=float).reshape(3,4,4)
        mask=np.zeros((4,4),dtype=bool);mask[1:3,1:3]=True
        predicted=reference*1.01
        predicted[:,~mask]=1e20
        self.assertAlmostEqual(field_relative_l2(predicted,reference,mask),.01,places=14)

    def test_arrival_censoring_and_mismatches(self):
        true=np.array([[1.,2.],[-1.,4.]])
        pred=np.array([[1.1,-1.],[3.,4.2]])
        result=arrival_metrics(pred,true,np.ones((2,2),dtype=bool))
        self.assertEqual(result["matched_reached_points"],2)
        self.assertEqual(result["mask_disagreement_points"],2)
        self.assertEqual(result["false_reached_points"],1)
        self.assertEqual(result["missed_reached_points"],1)
        self.assertAlmostEqual(result["RMSE_s"],np.sqrt(.025),places=14)

    def test_common_time_alignment_rejects_noncommon_samples(self):
        np.testing.assert_array_equal(matching_indices(np.array([.02,.04]),np.array([.01,.02,.03,.04])),[1,3])
        with self.assertRaises(ValueError):matching_indices(np.array([.025]),np.array([.01,.02,.03,.04]))

    def test_actual_completed_compact_records(self):
        paths=list((OUTPUT/"compact").glob("*.npz"))
        self.assertGreater(len(paths),0,"Actual completed simulation records are required")
        for path in paths:
            run=load_compact(path) # validates every shape, time, final-state and station relation
            audit=run_audit(run)
            self.assertTrue(audit["positive_state_all_recorded_steps"])
            self.assertGreater(audit["maximum_speed_m_s"],.5)
            self.assertTrue(audit["all_stations_exceed_0p5_m_s"],f"Station activation failed: {run.name}")
            # Tiny local negative powers are retained; the integrated event energy
            # is checked separately, not manufactured by clipping local values.
            self.assertGreater(audit["integrated_frictional_energy_J_right_rectangle"],0)

    def test_time_control_is_separate_and_uses_common_times(self):
        control=load_compact(OUTPUT/"compact"/"uniform_N360_reference_dt05.npz")
        baseline=load_compact(OUTPUT/"compact"/"uniform_N360_reference.npz")
        self.assertAlmostEqual(baseline.meta["dt_s"],.01)
        self.assertAlmostEqual(control.meta["dt_s"],.005)
        np.testing.assert_array_equal(baseline.x,control.x)
        np.testing.assert_array_equal(baseline.y,control.y)
        indices=matching_indices(baseline.time,control.time)
        reference=control.velocity[indices][:,control.vw].astype(float)
        delta=baseline.velocity[:,baseline.vw].astype(float)-reference
        expected=float(np.sqrt(np.sum(delta**2)/np.sum(reference**2)))
        metrics=json.loads((OUTPUT/"rupture_metrics.json").read_text())
        controls=metrics["reference_controls"]
        time=next(row for row in controls if row["comparison"]=="same_grid_reference_time_step_refinement")
        spatial=next(row for row in controls if row["comparison"]=="same_time_step_reference_spatial_refinement")
        self.assertAlmostEqual(time["velocity_relative_L2_VW"],expected,places=13)
        self.assertEqual((time["nx"],time["reference_nx"]),(360,360))
        self.assertEqual((spatial["dt_s"],spatial["reference_dt_s"]),(.005,.005))
        learning=next(row for row in metrics["learning_comparisons"] if row["case"]=="uniform" and row["nx"]==360)
        self.assertEqual(learning["reference_dt_s"],.01,"Time control must not replace the baseline")


if __name__=="__main__":unittest.main(verbosity=2)
