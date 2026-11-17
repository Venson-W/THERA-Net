"""Check the actual trained source-family checkpoint and solver interface."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fault_pinet import traction
from source_family_pinet import (DEFAULT_OUTPUT, load_source_model,
    source_family_parameters, source_truth, export_source_field)


class TestSourceFamily(unittest.TestCase):
    def test_group_holdout_integrity(self):
        config=json.loads((DEFAULT_OUTPUT/"config.json").read_text())
        groups=[set(config[name+"_group_indices"]) for name in ["train","validation","test"]]
        self.assertTrue(all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3)))
        self.assertEqual(len(set.union(*groups)),97)
        actual_groups=np.array(config["all_material_groups_a"])
        for a in [.0065,.007,.008,.016,.017,.0175]:
            index=int(np.argmin(np.abs(actual_groups-a)))
            self.assertIn(index,groups[2])

    def test_saved_checkpoint_reproduces_heldout_metric(self):
        model=load_source_model()
        data=np.load(DEFAULT_OUTPUT/"data_test.npz")
        parameters=source_family_parameters(model,data["a_source"])
        predicted=traction(data["velocity"],data["theta"],1.,parameters)
        error=float(np.linalg.norm(predicted-data["mu"])/np.linalg.norm(data["mu"]))
        expected=json.loads((DEFAULT_OUTPUT/"metrics.json").read_text())["selected_test_metrics"]["traction_relative_L2"]
        self.assertAlmostEqual(error,expected,places=15)

    def test_export_uses_neural_weights_and_preserves_shape(self):
        model=load_source_model()
        field=np.array([[.007,.017],[.0065,.0175]])
        predicted=source_family_parameters(model,field)
        self.assertEqual(predicted.shape,(2,2,4))
        self.assertGreater(float(np.max(np.abs(predicted-source_truth(field)))),0.)
        with tempfile.TemporaryDirectory(dir=DEFAULT_OUTPUT) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(DEFAULT_OUTPUT.resolve()))
            filename=Path(directory)/"coefficients.npy"
            export_source_field(filename,field,model)
            np.testing.assert_array_equal(np.load(filename),predicted)
        # Perturb a local in-memory copy: changing a weight must affect the field.
        model.network.biases[-1][0]+=.01
        altered=source_family_parameters(model,field)
        self.assertGreater(float(np.max(np.abs(altered-predicted))),1e-8)

    def test_outside_material_training_range_is_rejected(self):
        with self.assertRaises(ValueError): source_family_parameters(load_source_model(),.019)


if __name__=="__main__": unittest.main(verbosity=2)
