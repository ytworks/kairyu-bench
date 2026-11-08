from __future__ import annotations

import hashlib
import importlib.util
import copy
import numpy as np
import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights.finite_replay import ParameterEdits, batches, load_plan, prepare_frames, replay_sequence
from test_wsp_weights import weights


class ReplayTests(unittest.TestCase):
    def test_public_answer_requires_explicit_empty_thought_boundary(self) -> None:
        class Tokenizer:
            chat_template = "canonical-control-contract fixture"
            wrong_off = False
            def apply_chat_template(self, messages, *, tokenize, enable_thinking, add_generation_prompt=False):
                prefix = ("<|turn>system\n<|think|>\n<turn|>\n" if enable_thinking else "")
                prefix += "<|turn>user\n" + messages[0]["content"] + "<turn|>\n<|turn>model\n"
                if add_generation_prompt:
                    return prefix + ("" if enable_thinking or self.wrong_off else "<|channel>thought\n<channel|>")
                return prefix + messages[1]["content"] + "<turn|>"
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        tokenizer = Tokenizer()
        cases = {"code": [[{"id": "public", "prompt": "return sum", "correct": "x + y", "foil": "x - y"}]]}
        prepared, proof = prepare_frames(tokenizer, cases)
        _, prefix, good, bad = prepared["code"][0][0]
        old_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "return sum"}], tokenize=False,
            enable_thinking=True, add_generation_prompt=True)
        self.assertEqual(bytes(prefix).decode(), old_prefix + "<|channel>thought\n<channel|>")
        self.assertEqual(bytes(good), b"x + y")
        self.assertEqual(bytes(bad), b"x - y")
        self.assertNotEqual(proof["frames"]["code"][0][0]["prefix_sha256"], hashlib.sha256(old_prefix.encode()).hexdigest())
        self.assertEqual(proof["frame_protocol"], "gemma4-public-answer-v2-forced-empty-thought")
        self.assertTrue(proof["forced_empty_thought"])
        self.assertEqual(proof["schema_version"], 2)
        tokenizer.wrong_off = True
        with self.assertRaisesRegex(ValueError, "boundary contract"):
            prepare_frames(tokenizer, cases)

    def test_base_candidates_base_order_and_failure_restore(self) -> None:
        calls = []
        candidates = [{"id": f"candidate-{i}"} for i in range(6)]
        def apply(candidate):
            calls.append(("apply", None if candidate is None else candidate["id"]))
        def evaluate(**flags):
            calls.append(("evaluate", flags))
            return {"value": len(calls)}
        baseline, results, repeat, failure = replay_sequence(
            candidates, apply, evaluate,
            lambda name, result: calls.append(("save", name)))
        self.assertIsNotNone(baseline)
        self.assertEqual(len(results), 6)
        self.assertIsNotNone(repeat)
        self.assertIsNone(failure)
        self.assertEqual([c[1] for c in calls if c[0] == "apply"],
                         [None, *[c["id"] for c in candidates], None])
        calls.clear()
        def broken(**flags):
            if not flags.get("store_base") and not flags.get("recovery") and calls[-1] == ("apply", "candidate-4"):
                raise MemoryError("synthetic late allocation error")
            return {"base": True}
        _, results, repeat, failure = replay_sequence(
            candidates, apply, broken, lambda *args: None)
        self.assertEqual(len(results), 4)
        self.assertIsNotNone(repeat)
        self.assertIsInstance(failure, MemoryError)
        self.assertIsNone(failure.__traceback__)
        self.assertEqual([c[1] for c in calls if c[0] == "apply"],
                         [None, *[c["id"] for c in candidates[:5]], None])

    def test_initial_base_failure_and_writer_failure_restore(self) -> None:
        calls = []
        def evaluate(**flags):
            calls.append(flags)
            if not flags.get("recovery"):
                raise RuntimeError("initial base failure")
            return {"base": "restored"}
        baseline, _, repeat, failure = replay_sequence([], lambda _: None, evaluate, lambda *args: None)
        self.assertIsNone(baseline)
        self.assertIsNotNone(repeat)
        self.assertIsInstance(failure, RuntimeError)
        self.assertEqual(calls[-1], {"store_base": True, "recovery": True})

    def test_plan_hash_coordinates_and_output_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "checkpoint"
            weights(model / "model.safetensors", {
                "model.language_model.norm.weight": ("BF16", (2,), b"\x80\x3f\x00\x40"),
            })
            index = model / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"model.language_model.norm.weight": "model.safetensors"}}))
            case, frame = root / "cases.json", root / "frames.json"
            case.write_text("{}")
            frame.write_text("{}")
            spec = root / "spec.json"
            document = {"id": "candidate", "edits": [{"tensor": "model.language_model.norm.weight",
                "indices": [0], "set_values": [1.0078125]}]}
            spec.write_text(json.dumps(document))
            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            attestation = root / "attestation.json"
            attestation.write_text(json.dumps(Checkpoint(model).identity()))
            plan = {"schema_version": 1, "architecture": "gemma4_native_bf16",
                "checkpoint_index_sha256": sha(index),
                "checkpoint_attestation": {"path": str(attestation), "sha256": sha(attestation)},
                "cases": {"path": str(case), "sha256": sha(case)},
                "frame_evidence": {"path": str(frame), "sha256": sha(frame)},
                "candidates": [{"id": "candidate", "spec": str(spec), "sha256": sha(spec)}]}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            _, specs, _ = load_plan(plan_path, Checkpoint(model))
            self.assertEqual(specs[0]["_verified_metrics"]["delta_l2"], 1 / 128)
            for value in [1.001, 1.0, float("nan")]:
                document["edits"][0]["set_values"] = [value]
                spec.write_text(json.dumps(document))
                plan["candidates"][0]["sha256"] = sha(spec)
                plan_path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    load_plan(plan_path, Checkpoint(model))
            document["id"] = "../escape"
            document["edits"][0]["set_values"] = [1.0078125]
            spec.write_text(json.dumps(document))
            plan["candidates"][0].update(id="../escape", sha256=sha(spec))
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(plan_path, Checkpoint(model))
            plan["candidates"][0]["sha256"] = "0" * 64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(plan_path, Checkpoint(model))

    def test_batches_preserve_matched_order_and_bound(self) -> None:
        rows = [("long", [1] * 700, [2] * 10, [3] * 5),
                ("short", [1] * 10, [2] * 2, [3] * 5)]
        packed = batches(rows)
        self.assertEqual([row[0] for batch in packed for row in batch], ["short", "long"])
        self.assertEqual(len(packed), 2)


class SparseReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "checkpoint"
        self.matrix = "model.language_model.layers.0.mlp.down_proj.weight"
        self.norm = "model.language_model.norm.weight"
        raw = (np.arange(1, 7, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        weights(self.model / "model.safetensors", {
            self.matrix: ("BF16", (2, 3), raw),
            self.norm: ("BF16", (2,), b"\x80\x3f\x00\x80"),
        })
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {
            self.matrix: "model.safetensors", self.norm: "model.safetensors"}}))
        self.checkpoint = Checkpoint(self.model)
        receipt = self.root / "identity.json"
        receipt.write_text(json.dumps(self.checkpoint.identity()))
        cases, frame = self.root / "cases.json", self.root / "frame.json"
        cases.write_text("{}")
        frame.write_text("{}")
        self.documents = [
            {"id": "mixed", "edits": [
                {"tensor": self.matrix, "coordinates": [[0, 2], [1, 0]], "set_values": [3.03125, 4.0625]},
                {"tensor": self.norm, "indices": [0], "set_values": [1.0078125]}]},
            {"id": "other", "edits": [
                {"tensor": self.matrix, "coordinates": [[1, 2]], "set_values": [6.0625]},
                {"tensor": self.norm, "indices": [1], "set_values": [.0078125]}]},
        ]
        self.documents.extend(
            {"id": f"later-{i}", "edits": [
                {"tensor": self.matrix, "coordinates": [[1, 2]], "set_values": [6.125 + i * .0625]}]}
            for i in range(4))
        self.plan = {"schema_version": 1, "architecture": "gemma4_sparse_bf16",
                     "checkpoint_index_sha256": self.digest(index),
                     "checkpoint_attestation": {"path": str(receipt), "sha256": self.digest(receipt)},
                     "cases": {"path": str(cases), "sha256": self.digest(cases)},
                     "frame_evidence": {"path": str(frame), "sha256": self.digest(frame)}}
        self.plan_path = self.root / "plan.json"
        self.write_specs()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_specs(self):
        self.plan["candidates"] = []
        for document in self.documents:
            path = self.root / (document["id"] + ".json")
            path.write_text(json.dumps(document))
            self.plan["candidates"].append({"id": document["id"], "spec": str(path), "sha256": self.digest(path)})
        self.plan_path.write_text(json.dumps(self.plan))

    def test_selected_reader_and_matrix_plan_metrics(self):
        before = self.checkpoint.read_counts["tensor_payload"]
        np.testing.assert_array_equal(self.checkpoint.read_indices(self.matrix, [5, 0, 5, 3]), [6, 1, 6, 4])
        self.assertEqual(self.checkpoint.read_counts["tensor_payload"] - before, 6)
        _, specs, _ = load_plan(self.plan_path, self.checkpoint)
        self.assertEqual(specs[0]["edits"][0]["_flat_indices"], [2, 3])
        self.assertEqual(specs[0]["_verified_metrics"]["changed_elements"], 3)
        self.assertAlmostEqual(specs[0]["_verified_metrics"]["delta_l2"], np.linalg.norm([.03125, .0625, .0078125]))
        # A bad coordinate is rejected before it can address a different row.
        self.documents[0]["edits"][0]["coordinates"] = [[0, 3], [1, 0]]
        self.write_specs()
        with self.assertRaisesRegex(ValueError, "matrix coordinates"):
            load_plan(self.plan_path, self.checkpoint)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch RAM-replay boundary")
    def test_torch_sparse_restore_on_switch_and_failed_forward(self):
        import torch
        _, specs, _ = load_plan(self.plan_path, self.checkpoint)
        params = {self.matrix: torch.arange(1, 7, dtype=torch.bfloat16).reshape(2, 3),
                  self.norm: torch.tensor([1, -0.], dtype=torch.bfloat16)}
        pristine = {key: value.clone() for key, value in params.items()}
        # A decoded equality check would incorrectly accept +0 in place of source -0.
        params[self.norm][1] = 0.
        with self.assertRaisesRegex(ValueError, "differ from original"):
            ParameterEdits(self.checkpoint, specs, params, "cpu")
        params[self.norm][1] = -0.
        state = ParameterEdits(self.checkpoint, specs, params, "cpu")
        proof = state.apply(specs[0])
        self.assertTrue(proof["storage_bytes_exact"])
        self.assertEqual(proof["method"], "selected_bf16_cpu_readback")
        self.assertEqual(float(params[self.matrix][0, 2]), 3.03125)
        self.assertEqual(float(params[self.norm][0]), 1.0078125)
        state.apply(specs[1])
        self.assertEqual(float(params[self.matrix][0, 2]), 3.)
        self.assertEqual(float(params[self.matrix][1, 2]), 6.0625)
        self.assertEqual(float(params[self.norm][1]), .0078125)
        restored = state.apply(None)
        self.assertTrue(all(row["actual_bf16_sha256"] == row["original_bf16_sha256"]
                            for row in restored["tensors"]))
        self.assertTrue(torch.equal(params[self.norm].view(torch.int16), pristine[self.norm].view(torch.int16)))
        wrong = copy.deepcopy(specs[0])
        wrong["edits"][0]["_flat_indices"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "unchanged preverified"):
            state.apply(wrong)
        def successful(**flags):
            return [value.clone() for value in params.values()]
        _, results, repeated, failure = replay_sequence(specs, state.apply, successful, lambda *args: None)
        self.assertEqual(len(results), 6)
        self.assertIsNone(failure)
        self.assertTrue(all(torch.equal(params[key].view(torch.int16), value.view(torch.int16)) for key, value in pristine.items()))
        candidate_calls = 0
        def evaluate(**flags):
            nonlocal candidate_calls
            if not flags.get("store_base") and not flags.get("recovery"):
                candidate_calls += 1
                if candidate_calls == 5:
                    raise RuntimeError("synthetic late forward failure")
            return [value.clone() for value in params.values()]
        _, results, repeated, failure = replay_sequence(specs, state.apply, evaluate, lambda *args: None)
        self.assertEqual(len(results), 4)
        self.assertIsInstance(failure, RuntimeError)
        self.assertIsNotNone(repeated)
        self.assertTrue(all(torch.equal(params[key].view(torch.int16), value.view(torch.int16)) for key, value in pristine.items()))
        self.assertEqual(sum(value.numel() for value in state.originals.values()), 5)
        self.checkpoint.assert_stable()


if __name__ == "__main__":
    unittest.main()
