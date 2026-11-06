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
from kairyu_bench.wsp.weights import editor

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

    def mixed_fixture(self):
        tensors = {
            self.norm: ("BF16", (2,), bytes.fromhex("803f0040")),
            self.scalar: ("BF16", (), bytes.fromhex("0080")),
        }
        for kind, shape in (("gate", (16, 32)), ("up", (16, 32)), ("down", (32, 16)),
                            ("o", (32, 32)), ("q", (32, 32))):
            area = "mlp" if kind in {"gate", "up", "down"} else "self_attn"
            name = f"model.language_model.layers.0.{area}.{kind}_proj.weight"
            tensors[name] = ("BF16", shape, bytes.fromhex("803f") * int(np.prod(shape)))
        tensors["model.language_model.layers.1.self_attn.o_proj.weight"] = (
            "BF16", (32, 32), bytes.fromhex("803f") * 1024)
        weights(self.checkpoint.path / "model.safetensors", tensors)
        self.checkpoint = Checkpoint(self.checkpoint.path)
        identity = self.checkpoint.identity()
        attestation = self.root / "attestation.json"
        attestation.write_text(json.dumps({"checkpoint": identity}))
        self.plan["schema_version"] = 2
        self.plan["kind"] = "native_matrix_coupled_objectives_v2"
        self.plan["checkpoint"].update(tree_sha256=identity["sha256"],
                                       attestation={"path": str(attestation), "sha256": self.sha(attestation)})
        self.plan["baseline_tolerance"] = 1e-6
        self.plan["collections"] = {"native": dict(self.directories), "matrix": {}}
        self.plan["candidates"][0]["cost_metric"] = {
            "type": "relative", "floor": 1e-6, "block_scales": {"native": 1., "matrix": 1.}}
        coherent = COUPLED.matrix_recipe.mlp_triplet_samples(self.checkpoint.tensors, 0)
        entries, coords, source = [], [], []
        for name, tensor in sorted(self.checkpoint.tensors.items()):
            match = COUPLED.matrix_recipe.PATTERN.fullmatch(name)
            if not match:
                continue
            rc, strata = coherent.get(name, (np.array([[0, 0]], dtype=np.int64), {}))
            entries.append({"tensor": name, "shape": list(tensor.shape), "count": len(rc),
                            "start": len(source), "family": match[3], "layer": int(match[1]), "strata": strata})
            coords.extend(rc)
            source.extend([1.] * len(rc))
        coverage = {"entries": entries, "sampled_matrices": len(entries),
                    "sampled_coordinates": len(source), "sampling_mode": "mlp-triplets"}
        for objective, directory in self.directories.items():
            path = Path(directory)
            provenance = json.loads((path / "provenance.json").read_text())
            provenance.update(checkpoint=self.checkpoint.identity(False), attestation_receipt_sha256=self.sha(attestation))
            (path / "provenance.json").write_text(json.dumps(provenance))
            if objective == "correct-logprob":
                np.savez(path / "gradient.npz", **{self.norm: np.zeros(2), self.scalar: np.array(0.)})
            target = self.root / ("matrix-" + objective)
            target.mkdir()
            self.plan["collections"]["matrix"][objective] = str(target)
            np.savez(target / "sample-plan.npz", coordinates=np.asarray(coords), source_values=np.asarray(source, dtype=np.float32))
            gradient = np.zeros(len(source))
            if objective == "correct-logprob":
                for entry in entries:
                    if entry["family"] == "gate":
                        gradient[entry["start"]] = 1.
                    if entry["family"] == "q":
                        gradient[entry["start"]] = 1e6  # Diagnostic-only temptation must not enter the solve.
            np.save(target / "gradient.npy", gradient)
            summary = json.loads((path / "summary.json").read_text())
            summary["folds"][0]["file"] = "gradient.npy"
            summary["runtime"].pop("gradients")
            summary["runtime"].update(gradient_accumulation="FP32 factors, FP64 sum", gradient_checkpointing=False)
            matrix_provenance = dict(provenance, sampling_mode="mlp-triplets")
            matrix_provenance.pop("virtual_unit_gains")
            for name, document in (("summary.json", summary), ("provenance.json", matrix_provenance),
                                   ("coverage.json", coverage),
                                   ("frame-evidence.json", json.loads((path / "frame-evidence.json").read_text()))):
                (target / name).write_text(json.dumps(document))
        self.plan_path.write_text(json.dumps(self.plan))
        return entries

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
        entries = self.mixed_fixture()
        before = {p: self.sha(p) for p in self.checkpoint.files}
        output = self.root / "mixed-proposal"
        report = COUPLED.solve(self.plan_path, output, write_spec=True)
        actual = report["methods"][0]
        spec_path = output / "scope-a.json"
        spec = json.loads(spec_path.read_text())
        self.assertTrue(spec["matrix_batch"])
        self.assertTrue(any("indices" in e for e in spec["edits"]))
        self.assertTrue(any("coordinates" in e for e in spec["edits"]))
        self.assertFalse(any(".q_proj." in e["tensor"] for e in spec["edits"]))
        self.assertTrue(actual["predicted_constraints_satisfied"])
        self.assertEqual(actual["column_scope"]["excluded_qkv_columns"], 1)
        self.assertTrue(any(row["changed"] == 0 for row in actual["realized_matrix_support"]))
        candidate, manifest = self.root / "saved", self.root / "saved.json"
        editor.create(self.checkpoint.path, candidate, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate)
        qname = next(e["tensor"] for e in entries if e["family"] == "q")
        self.assertEqual(saved.read_raw(qname), self.checkpoint.read_raw(qname))
        self.assertEqual(saved.read_raw(self.scalar), bytes.fromhex("0080"))
        self.assertEqual({p: self.sha(p) for p in self.checkpoint.files}, before)
        # A new, explicit matrix-only hypothesis keeps all native source bytes fixed.
        self.plan["eligible_blocks"] = ["matrix"]
        self.plan["candidates"][0]["id"] = "matrix-only"
        margin_dir = Path(self.plan["collections"]["matrix"]["margin"])
        gradient = np.load(margin_dir / "gradient.npy", allow_pickle=False)
        up = next(e for e in entries if e["family"] == "up")
        gradient[up["start"]] = 1.
        np.save(margin_dir / "gradient.npy", gradient)
        plan_path = self.root / "matrix-only-plan.json"
        plan_path.write_text(json.dumps(self.plan))
        output = self.root / "matrix-only-proposal"
        report = COUPLED.solve(plan_path, output, write_spec=True)
        actual = report["methods"][0]
        spec_path = output / "matrix-only.json"
        spec = json.loads(spec_path.read_text())
        self.assertTrue(actual["predicted_constraints_satisfied"])
        self.assertEqual(actual["column_scope"]["native_columns"], 0)
        self.assertEqual(actual["column_scope"]["observed_native_columns"], 3)
        self.assertEqual(actual["realized_blocks"]["native"],
                         {"changed_elements": 0, "delta_l2": 0., "weighted_delta_l2": 0.})
        self.assertTrue(spec["edits"])
        self.assertTrue(all("coordinates" in e and ".q_proj." not in e["tensor"] for e in spec["edits"]))
        candidate, manifest = self.root / "matrix-only-saved", self.root / "matrix-only-saved.json"
        editor.create(self.checkpoint.path, candidate, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate)
        for name in (self.norm, self.scalar, qname):
            self.assertEqual(saved.read_raw(name), self.checkpoint.read_raw(name))
        self.assertEqual({p: self.sha(p) for p in self.checkpoint.files}, before)


        # A protected norm stays byte-exact even though its gradient could satisfy
        # the margin. Both objectives instead use the eligible matrix coordinates.
        self.plan["eligible_blocks"] = ["native", "matrix"]
        self.plan["native_tensors"] = [self.scalar]
        self.plan["candidates"][0]["id"] = "protected-norm"
        protected_plan = self.root / "protected-norm-plan.json"
        protected_plan.write_text(json.dumps(self.plan))
        protected_output = self.root / "protected-norm-proposal"
        protected_report = COUPLED.solve(protected_plan, protected_output, write_spec=True)
        self.assertTrue(protected_report["methods"][0]["predicted_constraints_satisfied"])
        self.assertEqual(protected_report["methods"][0]["native_scope"],
                         {"observed_columns": 3, "eligible_columns": 1,
                          "eligible_tensors": [self.scalar]})
        protected_spec = protected_output / "protected-norm.json"
        self.assertTrue(all(edit["tensor"] != self.norm for edit in
                            json.loads(protected_spec.read_text())["edits"]))
        protected_saved, protected_manifest = self.root / "protected-saved", self.root / "protected.json"
        editor.create(self.checkpoint.path, protected_saved, protected_spec, protected_manifest,
                      apply=True, reserve_bytes=0)
        editor.verify(protected_manifest)
        self.assertEqual(Checkpoint(protected_saved).read_raw(self.norm),
                         self.checkpoint.read_raw(self.norm))
        self.assertEqual({p: self.sha(p) for p in self.checkpoint.files}, before)
        # Typoed exact names fail before any gradient or output work.
        self.plan["native_tensors"] = [self.norm + ".typo"]
        protected_plan.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "native_tensors"):
            COUPLED.prepare(protected_plan)
        self.plan.pop("native_tensors")
        self.plan["eligible_blocks"] = ["matrix"]

        # Exact tensor scope excludes a much larger competing gradient in another
        # layer of the same matrix family. All unselected saved bytes stay exact.
        onames = sorted(e["tensor"] for e in entries if e["family"] == "o")
        self.plan["matrix_tensors"] = [onames[0]]
        self.plan["candidates"][0]["id"] = "one-output-group"
        for directory in self.plan["collections"]["matrix"].values():
            target = Path(directory) / "gradient.npy"
            values = np.load(target, allow_pickle=False)
            for entry in entries:
                if entry["tensor"] in onames:
                    values[entry["start"]] = 1. if entry["tensor"] == onames[0] else 1e6
            np.save(target, values)
        restricted_plan = self.root / "selected-matrix-plan.json"
        restricted_plan.write_text(json.dumps(self.plan))
        restricted_output = self.root / "selected-matrix-proposal"
        selected_report = COUPLED.solve(restricted_plan, restricted_output, write_spec=True)
        self.assertTrue(selected_report["methods"][0]["predicted_constraints_satisfied"])
        restricted_spec = restricted_output / "one-output-group.json"
        self.assertEqual([e["tensor"] for e in json.loads(restricted_spec.read_text())["edits"]], [onames[0]])
        selected_saved, selected_manifest = self.root / "selected-matrix-saved", self.root / "selected-matrix.json"
        editor.create(self.checkpoint.path, selected_saved, restricted_spec, selected_manifest,
                      apply=True, reserve_bytes=0)
        editor.verify(selected_manifest)
        saved = Checkpoint(selected_saved)
        for name in self.checkpoint.tensors:
            self.assertEqual(saved.read_raw(name) == self.checkpoint.read_raw(name), name != onames[0])
        self.assertEqual({p: self.sha(p) for p in self.checkpoint.files}, before)
        self.plan.pop("matrix_tensors")

        # Explicit Q-only scope selects Q cells and leaves every other byte intact.
        self.plan["matrix_families"] = ["q"]
        self.plan["candidates"][0]["id"] = "q-only"
        qentry = next(e for e in entries if e["family"] == "q")
        for directory in self.plan["collections"]["matrix"].values():
            target = Path(directory) / "gradient.npy"
            values = np.zeros_like(np.load(target, allow_pickle=False))
            values[qentry["start"]] = 1.
            np.save(target, values)
        qplan = self.root / "q-only-plan.json"
        qplan.write_text(json.dumps(self.plan))
        report = COUPLED.solve(qplan, self.root / "q-only-proposal", write_spec=True)
        self.assertEqual(report["methods"][0]["column_scope"]["active_matrix_families"], ["q"])
        self.assertEqual(report["methods"][0]["column_scope"]["excluded_qkv_columns"], 0)
        qspec = self.root / "q-only-proposal/q-only.json"
        self.assertEqual([e["tensor"] for e in json.loads(qspec.read_text())["edits"]], [qname])
        candidate, manifest = self.root / "q-only-saved", self.root / "q-only-saved.json"
        editor.create(self.checkpoint.path, candidate, qspec, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate)
        self.assertNotEqual(saved.read_raw(qname), self.checkpoint.read_raw(qname))
        for name in self.checkpoint.tensors:
            if name != qname:
                self.assertEqual(saved.read_raw(name), self.checkpoint.read_raw(name))
        for invalid in ([], ["q", "q"], ["unknown"], [True]):
            self.plan["matrix_families"] = invalid
            qplan.write_text(json.dumps(self.plan))
            with self.assertRaisesRegex(ValueError, "matrix_families"):
                COUPLED.prepare(qplan)

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
        # Fresh four-way data cannot accept stale whole-source evidence or permuted matrix columns.
        summary["folds"][0]["margins"][0]["id"] = "authored"
        summary_path.write_text(json.dumps(summary))
        self.mixed_fixture()
        path = Path(self.plan["collections"]["matrix"]["margin"]) / "provenance.json"
        original = json.loads(path.read_text())
        stale = copy.deepcopy(original)
        stale["checkpoint"]["files"][0]["stat"]["inode"] += 1
        path.write_text(json.dumps(stale))
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            COUPLED.prepare(self.plan_path)
        path.write_text(json.dumps(original))
        sample_path = path.parent / "sample-plan.npz"
        with np.load(sample_path, allow_pickle=False) as sample:
            coordinates, values = sample["coordinates"].copy(), sample["source_values"].copy()
        coordinates[[0, 1]] = coordinates[[1, 0]]
        np.savez(sample_path, coordinates=coordinates, source_values=values)
        with self.assertRaisesRegex(ValueError, "coordinates|columns"):
            COUPLED.solve(self.plan_path, self.root / "bad-mixed", write_spec=True)
        self.assertFalse((self.root / "bad-mixed").exists())


if __name__ == "__main__":
    unittest.main()
