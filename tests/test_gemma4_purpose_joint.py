from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_gemma4_purpose_recipes import COMMON, JOINT


class PurposeJointTests(unittest.TestCase):
    def test_minimum_diagonal_cost_and_explicit_allowed_loss(self):
        delta, proof = JOINT.minimum_step([[1., 1.], [0., 1.]], [1., -.2], [1., 2.])
        np.testing.assert_allclose(delta, [.8, .2], atol=1e-9)
        self.assertLess(proof["dual_residual"], 1e-9)
        delta, _ = JOINT.minimum_step([[1., 0.], [0., 0.]], [-.2, 0.], [1., 1.])
        np.testing.assert_array_equal(delta, [0., 0.])
        with self.assertRaisesRegex(ValueError, "zero"):
            JOINT.minimum_step([[0., 0.]], [.1], [1., 1.])
        with self.assertRaisesRegex(ValueError, "not converge"):
            JOINT.minimum_step([[1.], [-1.]], [1., 1.], [1.], maximum_iterations=8)

    def test_matrix_editor_limits_require_explicit_opt_in(self):
        edit = {"tensor": "x", "coordinates": [[0, i] for i in range(65)]}
        report = JOINT.editor_compatibility([edit])
        self.assertTrue(report["requires_matrix_batch"])
        self.assertFalse(report["spec_emission_allowed"])
        report = JOINT.editor_compatibility([edit], allow_batched_matrix=True)
        self.assertTrue(report["spec_emission_allowed"])
        observed_scope = [{"tensor": str(i), "coordinates": [[0, j] for j in range(512)]} for i in range(410)]
        report = JOINT.editor_compatibility(observed_scope, allow_batched_matrix=True)
        self.assertTrue(report["spec_emission_allowed"])
        excessive = [{"tensor": str(i), "coordinates": [[0, j] for j in range(512)]} for i in range(513)]
        report = JOINT.editor_compatibility(excessive, allow_batched_matrix=True)
        self.assertFalse(report["within_batched_editor_limit"])
        self.assertFalse(report["spec_emission_allowed"])

    def test_bf16_rounding_from_original_base_reports_predictive_limits(self):
        values, report = JOINT.quantized_proposal(
            [1., 2.], [[1., 0.], [0., 1.]], [.006, -.5], [1., 1.],
            relative_radius=.02, absolute_radius=1e-6, rounds=4)
        np.testing.assert_array_equal(values, [1.0078125, 2.])
        self.assertTrue(report["predicted_constraints_satisfied"])
        self.assertFalse(report["model_response_measured"])
        self.assertFalse(report["adopted"])
        self.assertEqual(report["actual_delta_l2"], .0078125)
        values, report = JOINT.quantized_proposal(
            [1.], [[1.]], [.01], [1.], relative_radius=0., absolute_radius=1e-6, rounds=2)
        np.testing.assert_array_equal(values, [1.])
        self.assertFalse(report["predicted_constraints_satisfied"])

    def test_collector_to_native_spec_preview_then_exclusive_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collection = root / "collection"
            collection.mkdir()
            name = "model.language_model.norm.weight"
            COMMON.save(collection / "summary.json", {"status": "completed", "candidate_selection": False,
                "source_values_unchanged": True, "mode": "raw-next-token",
                "folds": [{"purpose": "code", "fold": 0, "file": "gradient-code-0.npz"}]})
            COMMON.save(collection / "coverage.json", {"tensors": [{"name": name, "kind": "native"}]})
            COMMON.save(collection / "provenance.json", {"checkpoint": {"identity_level": "stat_only"},
                "mode": "raw-next-token"})
            np.savez(collection / "source-native-values.npz", **{name: np.array([1., 2.])})
            np.savez(collection / "gradient-code-0.npz", **{name: np.array([1., 0.])})
            plan = root / "plan.json"
            COMMON.save(plan, {"id": "fixture", "cost_metric": {"type": "euclidean"},
                "constraints": [{"purpose": "code", "fold": 0, "minimum_delta": .006}],
                "rounding": {"relative_radius": .02, "absolute_radius": 1e-6, "rounds": 2}})
            output = root / "output"
            argv = ["--collection", str(collection), "--plan", str(plan), "--output", str(output)]
            self.assertEqual(JOINT.main(argv), 0)
            self.assertFalse(output.exists())
            self.assertEqual(JOINT.main(argv + ["--write-spec"]), 0)
            spec = json.loads((output / "fixture.json").read_text())
            self.assertEqual(spec["edits"], [{"tensor": name, "indices": [0], "set_values": [1.0078125]}])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["represented_source_scope"], "selected native values")
            self.assertNotIn("relative_delta_l2_to_declared_reference", report)
            with self.assertRaisesRegex(ValueError, "new"):
                JOINT.main(argv + ["--write-spec"])

    def test_matrix_layout_generates_coordinate_spec_without_native_reinterpretation(self):
        with tempfile.TemporaryDirectory() as temp:
            root, name = Path(temp), "model.language_model.layers.0.mlp.down_proj.weight"
            collection = root / "collection"
            collection.mkdir()
            COMMON.save(collection / "summary.json", {"status": "completed", "candidate_selection": False,
                "source_values_unchanged": True, "mode": "raw-next-token",
                "folds": [{"purpose": "code", "fold": 0, "file": "gradient-code-0.npy"}]})
            COMMON.save(collection / "coverage.json", {"sampled_matrices": 1,
                "entries": [{"tensor": name, "shape": [3, 3], "count": 2, "start": 0}]})
            COMMON.save(collection / "provenance.json", {"checkpoint": {}, "mode": "raw-next-token"})
            np.savez(collection / "sample-plan.npz", source_values=np.array([1., 2.]),
                     coordinates=np.array([[0, 2], [2, 0]], dtype=np.int64))
            np.save(collection / "gradient-code-0.npy", [1., 0.])
            plan = root / "plan.json"
            COMMON.save(plan, {"id": "matrix-fixture", "cost_metric": {"type": "euclidean"},
                "constraints": [{"purpose": "code", "fold": 0, "minimum_delta": .006}],
                "rounding": {"relative_radius": .02, "absolute_radius": 1e-6, "rounds": 2}})
            output = root / "output"
            JOINT.main(["--collection", str(collection), "--plan", str(plan), "--output", str(output), "--write-spec"])
            edit = json.loads((output / "matrix-fixture.json").read_text())["edits"][0]
            self.assertEqual(edit["coordinates"], [[0, 2]])
            self.assertNotIn("indices", edit)


if __name__ == "__main__":
    unittest.main()
