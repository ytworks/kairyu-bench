from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_gemma4_purpose_recipes import load
from test_wsp_weights import weights
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

COUPLED = load("propose_native_objectives")


class NativeCoupledTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.norm = "model.language_model.norm.weight"
        self.scalar = "model.language_model.layers.0.layer_scalar"
        self.matrix = "model.language_model.layers.0.mlp.down_proj.weight"
        model = self.root / "model"
        weights(model / "model.safetensors", {
            self.norm: ("BF16", (2,), bytes.fromhex("803f0040")),
            self.scalar: ("BF16", (), bytes.fromhex("0080")),
            self.matrix: ("BF16", (1, 1), bytes.fromhex("803f")),
        })
        self.checkpoint = Checkpoint(model)
        identity = self.checkpoint.identity()
        attestation = self.root / "attestation.json"
        attestation.write_text(json.dumps({"checkpoint": identity}))
        self.cases = self.root / "cases.json"
        self.cases.write_text(json.dumps({"science": [[{
            "id": "authored", "prompt": "p", "correct": "good", "foil": "bad"}]]}))
        self.case_sha = self.sha(self.cases)
        coverage = {"native_tensors": 2, "virtual_tensors": 0, "native_coordinates": 3,
                    "virtual_coordinates": 0, "tensors": [
                        {"name": self.norm, "shape": [2], "dtype": "torch.bfloat16", "kind": "native"},
                        {"name": self.scalar, "shape": [], "dtype": "torch.bfloat16", "kind": "native"}]}
        self.directories = {}
        for objective in COUPLED.OBJECTIVES:
            path = self.root / objective
            path.mkdir()
            self.directories[objective] = str(path)
            values = np.array([1, 2], dtype=np.float32)
            np.savez(path / "source-native-values.npz", **{self.norm: values, self.scalar: np.array(-0., dtype=np.float32)})
            gradients = [1., 0.] if objective == "correct-logprob" else [-1., 1.]
            np.savez(path / "gradient.npz", **{self.norm: np.array(gradients), self.scalar: np.array(0.)})
            field, value = ("correct_logprobs", "correct_logprob") if objective == "correct-logprob" else ("margins", "margin")
            summary = {"status": "completed", "candidate_selection": False, "source_values_unchanged": True,
                       "mode": "chat-public-answer", "objective_kind": objective, "answer_batching": "serial",
                       "runtime": {"torch": "fixture", "transformers": "fixture", "weight_dtype": "BF16",
                                   "attention": "eager", "gradients": "FP32 accumulation",
                                   "max_cases_per_microbatch": 1,
                                   "gpu_peak_allocated_GiB": 1 if objective == "margin" else 2},
                       "folds": [{"purpose": "science", "fold": 0, "examples": 1, "file": "gradient.npz",
                                  field: [{"id": "authored", value: -1. if objective == "correct-logprob" else 0.}]}]}
            provenance = {"mode": "chat-public-answer", "objective_kind": objective, "answer_batching": "serial",
                          "virtual_unit_gains": False, "max_cases_per_microbatch": 1, "threads": 4, "seed": 0,
                          "checkpoint": self.checkpoint.identity(False),
                          "attestation_receipt_sha256": self.sha(attestation),
                          "input_sha256": {str(self.cases): self.case_sha}, "input_stat": {},
                          "calibration_scope": "authored fixture"}
            frame = {"schema_version": 2, "frame_protocol": "gemma4-public-answer-v2-forced-empty-thought",
                     "forced_empty_thought": True, "frames": {"science": [[{"id": "authored"}]]}}
            for name, data in [("summary.json", summary), ("provenance.json", provenance),
                               ("coverage.json", coverage), ("frame-evidence.json", frame)]:
                (path / name).write_text(json.dumps(data))
        self.plan = {"schema_version": 1, "kind": "native_coupled_objectives_v1",
                     "checkpoint": {"path": str(model), "tree_sha256": identity["sha256"],
                                    "attestation": {"path": str(attestation), "sha256": self.sha(attestation)}},
                     "cases": {"path": str(self.cases), "sha256": self.case_sha},
                     "collections": self.directories,
                     "candidates": [{"id": "scope-a", "constraints": [
                         {"objective": o, "purpose": "science", "fold": 0, "minimum_delta": 1 / 128}
                         for o in COUPLED.OBJECTIVES], "cost_metric": {"type": "euclidean"},
                         "rounding": {"relative_radius": .025, "absolute_radius": 1e-6, "rounds": 2}}]}
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_two_objectives_produce_exact_bf16_spec_without_touching_source(self):
        before = {p: self.sha(p) for p in self.checkpoint.files}
        _, _, _, labels, _ = COUPLED.prepare(self.plan_path)
        self.assertEqual(labels, [("science", 0)])
        with unittest.mock.patch.object(COUPLED.joint, "read_collection", side_effect=AssertionError("preview read gradient")):
            COUPLED.main(["--plan", str(self.plan_path), "--output", str(self.root / "preview")])
        self.assertFalse((self.root / "preview").exists())
        output = self.root / "proposal"
        report = COUPLED.solve(self.plan_path, output, write_spec=True)
        actual = report["methods"][0]
        self.assertTrue(actual["predicted_constraints_satisfied"])
        self.assertEqual(actual["changed_elements"], 2)
        spec = json.loads((output / "scope-a.json").read_text())
        self.assertEqual(spec["edits"], [{"tensor": self.norm, "indices": [0, 1], "set_values": [1.0078125, 2.015625]}])
        self.assertEqual(actual["prediction"], [1 / 128, 1 / 128])
        self.assertEqual({p: self.sha(p) for p in self.checkpoint.files}, before)
        self.assertEqual(self.checkpoint.read_raw(self.scalar), bytes.fromhex("0080"))
        with self.assertRaisesRegex(ValueError, "fresh output"):
            COUPLED.solve(self.plan_path, output, write_spec=True)

    def test_rejects_full_checkpoint_mismatch_even_with_same_native_values_and_bad_cohort(self):
        path = Path(self.directories["margin"]) / "provenance.json"
        original = json.loads(path.read_text())
        altered = copy.deepcopy(original)
        altered["checkpoint"]["files"][0]["stat"]["inode"] += 1
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            COUPLED.prepare(self.plan_path)
        path.write_text(json.dumps(original))
        summary_path = path.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["folds"][0]["margins"][0]["id"] = "another-case"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "cohort"):
            COUPLED.prepare(self.plan_path)
        self.assertFalse((self.root / "proposal").exists())


if __name__ == "__main__":
    unittest.main()
