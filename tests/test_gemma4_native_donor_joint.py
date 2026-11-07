"""Joint cost and mixed rounding are observable diagnostic contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"))
import propose_joint_native_donor as tool  # noqa: E402


class NativeDonorJointTests(unittest.TestCase):
    def test_joint_cost_signed_gate_and_native_only_rounding(self):
        values, gates, report = tool.joint_proposal(
            np.array([0.]), np.array([[1.]]), np.array([[-1.]]), np.array([.015625]), np.array([2.]))
        # Independent two-variable minimum: x - c = b, cost x² + 4c².
        self.assertAlmostEqual(report["continuous_weight_l2"], np.sqrt(.0125**2 + 4*.003125**2))
        self.assertEqual(gates[0], np.float32(-.003125))
        self.assertNotEqual(int(gates.view(np.uint32)[0]) & 0xFFFF, 0)  # Gate was not BF16-rounded.
        self.assertEqual(values[0], np.float32(.01251220703125))
        self.assertTrue(report["eligible_for_joint_finite"])
        self.assertEqual(report["solver_calls"], 1)
        self.assertFalse(report["donor_bf16_rounding_measured"])
        self.assertFalse(report["actual_joint_model_response_measured"])

    def test_native_rounding_can_reject_continuously_satisfied_target(self):
        values, gates, report = tool.joint_proposal(
            np.array([1.]), np.array([[1.]]), np.array([[0.]]), np.array([.001]), np.array([1.]))
        self.assertLess(report["continuous_maximum_shortfall"], 1e-10)
        np.testing.assert_array_equal(values, [1.])
        np.testing.assert_array_equal(gates, [0.])
        self.assertFalse(report["eligible_for_joint_finite"])
        self.assertAlmostEqual(report["maximum_shortfall"], .001)
        self.assertEqual(report["solver_calls"], 1)  # No unregistered amplitude/quantization repair.

    def test_native_collection_matches_frame_objective_and_original_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            labels = [("code", 0)]
            fold = {"purpose": "code", "fold": 0, "file": "g.npz",
                    "mean_correct_logprob": -1., "correct_logprobs": [{"id": "own-0", "correct_logprob": -1.}]}
            runtime = {"torch": "fixture", "transformers": "fixture", "attention": "eager",
                       "max_cases_per_microbatch": 1, "weight_dtype": "BF16"}
            summary = {"status": "completed", "candidate_selection": False, "mode": "chat-public-answer",
                       "source_values_unchanged": True, "objective_kind": "correct-logprob",
                       "answer_batching": "serial", "runtime": runtime, "folds": [fold]}
            provenance = {"mode": "chat-public-answer", "objective_kind": "correct-logprob",
                          "answer_batching": "serial", "virtual_unit_gains": False, "checkpoint": {"sha256": "a"*64},
                          "attestation_receipt_sha256": "b"*64, "threads": 1, "seed": 0,
                          "max_cases_per_microbatch": 1, "input_sha256": {"own-cases": "c"*64}}
            frames = {"frame_protocol": tool.positive.FRAME_PROTOCOL}
            coverage = {"tensors": [{"name": "norm", "kind": "native"}]}
            for filename, value in (("summary.json", summary), ("provenance.json", provenance),
                                    ("coverage.json", coverage), ("frame-evidence.json", frames)):
                (path / filename).write_text(json.dumps(value))
            np.savez(path / "source-native-values.npz", norm=np.array([1., 2.], dtype=np.float32))
            np.savez(path / "g.npz", norm=np.array([.1, .2], dtype=np.float32))
            donor = {"summary": {"mode": "chat-public-answer", "runtime": dict(runtime, source_dtype="BF16"),
                                 "folds": [dict(fold)]}, "provenance": provenance, "frames": frames}
            loaded = tool.native_collection(path, "correct-logprob", donor, labels)
            np.testing.assert_array_equal(loaded["base"], [1., 2.])
            donor["summary"]["folds"] = [dict(fold, mean_correct_logprob=-.9)]
            with self.assertRaisesRegex(ValueError, "per-case objective differs"):
                tool.native_collection(path, "correct-logprob", donor, labels)

    def test_explicit_residual_repair_freezes_native_and_does_not_clip_gate(self):
        base = np.array([1.])
        frozen = np.array([1.], dtype=np.float32)
        values, gates, report = tool.repair_proposal(base, frozen, [[1.]], [[1.]], [.001], [1.])
        np.testing.assert_array_equal(values.view(np.uint32), frozen.view(np.uint32))
        self.assertEqual(gates[0], np.float32(.001))
        self.assertTrue(report["eligible_for_joint_finite"])
        self.assertFalse(report["native_rescaled_or_rerounded"])
        self.assertEqual(report["solver_calls"], 1)
        _, gates, rejected = tool.repair_proposal(base, frozen, [[1.]], [[1.]], [.3], [1.])
        self.assertEqual(gates[0], np.float32(.3))
        self.assertFalse(rejected["eligible_for_joint_finite"])
        self.assertIn("donor_coefficient_outside_box", rejected["rejection_reasons"])
        _, _, unchanged = tool.repair_proposal(base, frozen, [[1.]], [[1.]], [0.], [1.])
        self.assertFalse(unchanged["eligible_for_joint_finite"])
        self.assertIn("no_change", unchanged["rejection_reasons"])

    def test_component_export_preserves_values_and_binds_companion_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair, donor = root / "repair", root / "donor"
            repair.mkdir()
            donor.mkdir()
            layout = [{"tensor": "norm", "count": 2}]
            values = np.array([1.0078125, 2.], dtype=np.float32)
            gates = np.zeros(410, dtype=np.float32)
            gates[0] = .001
            sources = []
            for name in ("native-m", "native-c"):
                p = root / name
                p.mkdir()
                path = p / "source-native-values.npz"
                np.savez(path, norm=np.array([1., 2.], dtype=np.float32))
                sources.append(path)
            np.savez(repair / "proposal.npz", native_bf16_values=values, donor_fp32_coefficients=gates)
            entries = [{"tensor": f"weight{i}"} for i in range(410)]
            (donor / "summary.json").write_text(json.dumps({"coverage": {"entries": entries}}))
            np.save(donor / "unit-gate-weight-l2.npy", np.ones(410))
            scalar = root / "scalar.json"
            scalar.write_text(json.dumps({"schema_version": 2, "kind": "dense_lora_recipe_v1", "coefficient": .25}))
            plan = {"kind": tool.positive.HEADROOM_KIND, "source_checkpoint": str(root / "source"),
                    "source_attestation_sha256": "a"*64, "plans": [{"constraints": [{"minimum_delta": 0.}]}],
                    "margin_collection": str(donor), "correct_collection": str(root / "donor-c"),
                    "donor_spec": str(scalar), "donor_spec_sha256": tool.sha(scalar),
                    "unit_costs_path": str(donor / "unit-gate-weight-l2.npy"),
                    "unit_costs_sha256": tool.sha(donor / "unit-gate-weight-l2.npy")}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            hashes = {str(p): tool.sha(p) for p in [*sources, plan_path, scalar,
                       donor / "summary.json", donor / "unit-gate-weight-l2.npy"]}
            report = {"eligible_for_joint_finite": True, "proposal_mode": "frozen_native_donor_residual_repair",
                      "source_sha256": hashes, "source_checkpoint": plan["source_checkpoint"],
                      "source_attestation_sha256": "a"*64, "constraints": plan["plans"][0]["constraints"],
                      "layout": layout, "proposal_sha256": tool.sha(repair / "proposal.npz"),
                      "hybrid_weight_l2": float(np.hypot(.0078125, float(gates[0])))}
            report_path = repair / "report.json"
            report_path.write_text(json.dumps(report))
            result = tool.export_components(plan_path, report_path, root / "components")
            native = json.loads(Path(result["components"][0]["path"]).read_text())
            module = json.loads(Path(result["components"][1]["path"]).read_text())
            self.assertEqual(native["edits"], [{"tensor": "norm", "indices": [0], "set_values": [1.0078125]}])
            self.assertEqual(len(module["module_coefficients"]), 410)
            self.assertEqual(module["module_coefficients"]["weight0"], float(gates[0]))
            for spec in (native, module):
                self.assertTrue(spec["companion_required"])
                self.assertEqual(spec["joint_report_sha256"], tool.sha(report_path))
            with self.assertRaisesRegex(ValueError, "fresh component"):
                tool.export_components(plan_path, report_path, root / "components")
            np.savez(repair / "proposal.npz", native_bf16_values=values + 1, donor_fp32_coefficients=gates)
            with self.assertRaisesRegex(ValueError, "pinned balanced repair"):
                tool.export_components(plan_path, report_path, root / "changed")


if __name__ == "__main__":
    unittest.main()
