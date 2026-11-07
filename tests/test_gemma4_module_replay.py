"""CPU-only exact RAM merge, late restoration, and public-answer metric checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights.finite_replay import PublicAnswerEvaluator, replay_sequence
from test_wsp_weights import weights

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"
SPEC = importlib.util.spec_from_file_location("finite_module_replay", SCRIPTS / "finite_module_replay.py")
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "optional Torch CPU metric contract")
class PublicAnswerMetricTest(unittest.TestCase):
    def test_shared_metrics_match_independent_all_answer_oracle_and_recovery(self):
        import torch
        class Net:
            config = SimpleNamespace(final_logit_softcapping=3.)
            seen = []
            def __call__(self, input_ids, **kwargs):
                self.seen.append(tuple(input_ids.shape))
                return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 4).float())
        net = Net()
        head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[.1, .2, -.3, .7], [.6, -.2, .1, .4],
                                           [.3, .5, .8, -.1], [-.4, .2, .1, .3]]))
        head.requires_grad_(False)
        original = head.weight.clone()
        rows = [("short", [0, 1], [2, 3], [1]), ("long", [2], [1], [0, 3])]
        calls = []
        evaluate = PublicAnswerEvaluator(net, head, SimpleNamespace(pad_token_id=0),
                                         {"science": [rows]}, "cpu", lambda: calls.append(1))
        def oracle():
            all_rows = []
            for _, prefix, good, foil in rows:
                probabilities = []
                for answer in (good, foil):
                    sequence = prefix + answer
                    selected = net(torch.tensor([sequence])).last_hidden_state[0, len(prefix)-1:-1]
                    logits = head(selected)
                    logp = torch.log_softmax(3 * torch.tanh(logits / 3), -1)
                    score = logp[torch.arange(len(answer)), torch.tensor(answer)].mean().item()
                    probabilities.append((score, logp.detach().clone()))
                all_rows.append(probabilities)
            return all_rows
        old = oracle()
        base = evaluate(store_base=True)
        serial = PublicAnswerEvaluator(net, head, SimpleNamespace(pad_token_id=0),
                                       {"science": [rows]}, "cpu", answer_batching="serial")
        net.seen.clear()
        serial_base = serial(store_base=True)
        self.assertEqual(net.seen, [(1, 3), (1, 3), (1, 4), (1, 4)])
        np.testing.assert_allclose(serial_base["values"]["science"][0], base["values"]["science"][0], atol=1e-7)
        np.testing.assert_allclose(serial_base["correct_nll"], base["correct_nll"], atol=1e-7)
        np.testing.assert_allclose(base["values"]["science"][0],
                                   [a[0][0] - a[1][0] for a in old], atol=1e-7)
        self.assertAlmostEqual(base["correct_nll"][0], -np.mean([a[0][0] for a in old]), places=6)
        self.assertEqual(base["kl"], [0.])
        self.assertTrue(calls)
        with torch.no_grad():
            head.weight[2, 1] += .2
        new = oracle()
        changed = evaluate()
        serial_changed = serial()
        for key in ("means", "correct_nll", "kl"):
            np.testing.assert_allclose(serial_changed[key], changed[key], atol=1e-7)
        expected_kl = []
        for before, after in zip(old, new, strict=True):
            ref, cur = before[0][1], after[0][1]
            positions = sorted({0, len(ref) - 1})
            expected_kl.append(float((ref[positions].exp() * (ref[positions] - cur[positions])).sum(-1).mean()))
        np.testing.assert_allclose(changed["values"]["science"][0],
                                   [a[0][0] - a[1][0] for a in new], atol=1e-7)
        self.assertAlmostEqual(changed["kl"][0], np.mean(expected_kl), places=6)
        self.assertAlmostEqual(changed["correct_nll"][0], -np.mean([a[0][0] for a in new]), places=6)
        with torch.no_grad():
            head.weight.copy_(original)
        evaluate.deadline = lambda: (_ for _ in ()).throw(TimeoutError("expired"))
        restored = evaluate(recovery=True)
        serial.deadline = evaluate.deadline
        serial_restored = serial(recovery=True)
        self.assertEqual(serial_restored["values"], serial_base["values"])
        self.assertEqual(serial_restored["correct_nll"], serial_base["correct_nll"])
        self.assertEqual(serial_restored["kl"], [0.])
        self.assertEqual(restored["values"], base["values"])
        self.assertEqual(restored["correct_nll"], base["correct_nll"])
        self.assertEqual(restored["kl"], [0.])


@unittest.skipUnless(TORCH and shutil.which("c++"), "optional Torch and C++ CPU RAM contract")
class ModuleRamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        library = Path(cls.build.name) / "kernel.so"
        subprocess.run(["sh", str(SCRIPTS / "build-kernel.sh"), str(library)], check=True, capture_output=True)
        with patch.dict(os.environ, {"KAIRYU_BENCH_BF16_LORA_LIBRARY": str(library)}):
            spec = importlib.util.spec_from_file_location("ram_test_ops", SCRIPTS / "bf16_lora_ops.py")
            cls.ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.ops)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_six_exact_candidates_cached_once_and_late_failure_restores_bits(self):
        import torch
        for device_proof in ("roundtrip", "staged-int16"):
            with self.subTest(device_proof=device_proof), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                names = ["model.language_model.layers.0.mlp.up_proj.weight",
                         "model.language_model.layers.1.self_attn.o_proj.weight"]
                values = np.array([[1., -0., -1., .125], [2., -2., .5, -.5], [4., .25, -4., .0625]], np.float32)
                values = np.tile(values, (86, 1))[:257]
                raw = (values.view(np.uint32) >> 16).astype("<u2")
                weights(root / "model.safetensors", {n: ("BF16", (257, 4), raw.tobytes()) for n in names})
                checkpoint = Checkpoint(root)
                params = {n: torch.from_numpy(raw.copy()).view(torch.bfloat16) for n in names}
                mapping = [{"tensor": n, "shape": [257, 4], "lora_A": "A", "lora_B": "B"} for n in names]
                a = np.full((16, 4), 1 / 8, np.float32)
                b = np.full((257, 16), 1 / 16, np.float32)
                calls = []
                def delta_block(av, bv):
                    calls.append(1)
                    return (bv @ av) * np.float32(2)
                def metrics(records):
                    return {"changed_elements": sum(r["changed_count"] for r in records),
                            "changed_tensors": sum(r["changed_count"] > 0 for r in records),
                            "delta_l2": math.hypot(*(r["delta_l2"] for r in records))}
                module = SimpleNamespace(BLOCK_ROWS=128, delta_block=delta_block, merge_block=self.ops.merge1,
                                         metrics=metrics, dense=SimpleNamespace(tensor=lambda _, h, o, name: {"A": a, "B": b}[name]))
                coefficients = [-.25, -.125, 0., .0625, .125, .25]
                recipes = {f"c{i}": {"donor_path": "unused", "module_coefficients": {names[0]: c, names[1]: -c}}
                           for i, c in enumerate(coefficients)}
                candidates = [{"id": name, "recipe": recipe} for name, recipe in recipes.items()]
                cache = tool.StreamedModuleEdits(checkpoint, params, mapping, recipes, module, {}, 0,
                                                maximum_bytes=raw.size * 2 * 6, device_proof=device_proof)
                seen = []
                transfers = {"read": 0, "write_verify": 0}
                raw_method, copy_method = cache.raw, cache.copy_and_check
                def tracked_read(tensor):
                    transfers["read"] += 1
                    return raw_method(tensor)
                def tracked_write(tensor, bits):
                    transfers["write_verify"] += 1
                    return copy_method(tensor, bits)
                cache.raw, cache.copy_and_check = tracked_read, tracked_write
                def evaluate(**flags):
                    current = {n: raw_method(p).copy() for n, p in params.items()}
                    if not flags.get("store_base") and not flags.get("recovery"):
                        index = len(seen)
                        c = coefficients[index]
                        for n, coefficient in ((names[0], c), (names[1], -c)):
                            expected = raw.copy() if coefficient == 0 else self.ops.merge1(raw, (b @ a) * np.float32(2), coefficient)[0]
                            np.testing.assert_array_equal(current[n], expected)
                            record = cache.receipts[f"c{index}"]["edits"][names.index(n)]
                            self.assertEqual(record["candidate_tensor_sha256"], hashlib.sha256(expected.tobytes()).hexdigest())
                        seen.append(current)
                    return current
                _, results, repeat, failure = replay_sequence(candidates, cache.apply, evaluate, lambda *args: None)
                self.assertIsNone(failure)
                self.assertEqual(len(results), 6)
                self.assertEqual(len(calls), 6)  # two matrices, three unchanged 128-row GEMM blocks each
                self.assertEqual(transfers, {"read": 16 if device_proof == "roundtrip" else 0, "write_verify": 14})  # matrix transfers, never row-block transfers
                self.assertEqual(sum(v.nbytes for v in cache.originals.values()) + sum(v.nbytes for v in cache.deltas.values()), raw.size * 2 * 6)
                for value in repeat.values():
                    np.testing.assert_array_equal(value, raw)  # includes negative-zero bits
                count = 0
                def fail_late(**flags):
                    nonlocal count
                    if not flags.get("store_base") and not flags.get("recovery"):
                        count += 1
                        if count == 5:
                            raise MemoryError("synthetic late forward allocation")
                    return True
                _, completed, repeated, failure = replay_sequence(candidates, cache.apply, fail_late, lambda *args: None)
                self.assertEqual(len(completed), 4)
                self.assertTrue(repeated)
                self.assertIsInstance(failure, MemoryError)
                self.assertIsNone(failure.__traceback__)
                for p in params.values():
                    np.testing.assert_array_equal(cache.raw(p), raw)
                self.assertTrue(cache.restoration["storage_bytes_exact"])
                self.assertEqual(cache.restoration["proof_method"], device_proof)
                if device_proof == "roundtrip":
                    self.assertTrue(cache.restoration["readback_exact"])
                else:
                    self.assertNotIn("readback_exact", cache.restoration)
                for receipt in cache.receipts.values():
                    self.assertTrue(receipt["actual_ram_storage_bytes_verified"])
                    self.assertEqual(receipt["actual_ram_readback_verified"], device_proof == "roundtrip")
                    self.assertEqual(receipt["device_proof"], device_proof)
                if device_proof == "staged-int16":
                    bad_params = {name: value.clone() for name, value in params.items()}
                    bad_params[names[0]].view(torch.int16)[0, 0] ^= 1
                    before_bad = bad_params[names[0]].clone()
                    bad = tool.StreamedModuleEdits(checkpoint, bad_params, mapping, recipes, module, {}, 0,
                                                   maximum_bytes=raw.size * 2 * 6, device_proof=device_proof)
                    with self.assertRaisesRegex(ValueError, "device int16"):
                        bad.prepare()
                    torch.testing.assert_close(bad_params[names[0]], before_bad, rtol=0, atol=0)
                    self.assertFalse(bad.ready)
                    self.assertFalse(bad.originals)
                checkpoint.assert_stable()
                with self.assertRaisesRegex(ValueError, "host cache"):
                    tool.StreamedModuleEdits(checkpoint, params, mapping, recipes, module, {}, 0, maximum_bytes=1)


@unittest.skipUnless(TORCH, "optional Torch CPU joint restoration contract")
class JointRamTest(unittest.TestCase):
    def test_native_and_module_bits_metrics_and_both_partial_failure_restorations(self):
        import torch
        for failure_mode in ("none", "native_apply", "module_apply", "native_restore"):
            with self.subTest(failure_mode=failure_mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                matrix, norm = "model.language_model.layers.0.mlp.up_proj.weight", "model.language_model.norm.weight"
                initial = np.array([1., -0.], np.float32)
                bits = (initial.view(np.uint32) >> 16).astype("<u2")
                weights(root / "model.safetensors", {
                    matrix: ("BF16", (1, 2), bits.tobytes()), norm: ("BF16", (2,), bits.tobytes())})
                checkpoint = Checkpoint(root)
                params = {matrix: torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(1, 2),
                          norm: torch.from_numpy(bits.copy()).view(torch.bfloat16)}
                spec = {"id": "native", "edits": [{"tensor": norm, "indices": [0], "_flat_indices": [0],
                                                 "set_values": [1.5]}],
                        "_verified_metrics": {"changed_elements": 1, "changed_tensors": 1, "delta_l2": .5}}
                component = {"spec": spec, "spec_sha256": "a" * 64}
                candidate = {"id": "joint", "native_component": component}
                calls = []
                class Modules:
                    mapping = [{"tensor": matrix}]
                    receipts = {}
                    restoration = {}
                    def apply(self, value):
                        calls.append("module_restore" if value is None else "module_apply")
                        params[matrix].copy_(torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(1, 2))
                        if value is None:
                            self.restoration = {"storage_bytes_exact": True, "matrices": 1}
                            return
                        params[matrix][0, 0] = 1.25
                        if failure_mode == "module_apply":
                            raise RuntimeError("partial module apply")
                        self.receipts["joint"] = {"changed_elements": 1, "changed_tensors": 1, "delta_l2": .25,
                                                  "actual_ram_storage_bytes_verified": True}
                modules = Modules()
                joint = tool.JointEdits(modules, checkpoint, [candidate], params, "cpu")
                native_apply = joint.native.apply
                native_modified = False
                def native(value):
                    nonlocal native_modified
                    calls.append("native_restore" if value is None else "native_apply")
                    native_apply(value)
                    if value is not None:
                        native_modified = True
                        if failure_mode == "native_apply":
                            raise RuntimeError("partial native apply")
                    elif native_modified and failure_mode == "native_restore":
                        raise RuntimeError("native restore verification failure")
                joint.native.apply = native
                def evaluate(**flags):
                    if not flags.get("store_base") and not flags.get("recovery"):
                        self.assertEqual(float(params[norm][0]), 1.5)
                        self.assertEqual(float(params[matrix][0, 0]), 1.25)
                    return True
                _, completed, repeated, error = replay_sequence([candidate], joint.apply, evaluate, lambda *args: None)
                for name in (matrix, norm):
                    np.testing.assert_array_equal(joint.raw(params[name]).reshape(-1), bits)
                self.assertEqual(calls[-2:], ["native_restore", "module_restore"])
                if failure_mode == "none":
                    self.assertIsNone(error)
                    self.assertEqual(len(completed), 1)
                    receipt = joint.receipts["joint"]
                    self.assertEqual(receipt["changed_elements"], 2)
                    self.assertAlmostEqual(receipt["delta_l2"], math.hypot(.5, .25))
                    self.assertEqual(receipt["native"]["edits"][0]["after_hex"], "c03f")
                    self.assertTrue(joint.restoration["storage_bytes_exact"])
                else:
                    self.assertIsInstance(error, RuntimeError)
                self.assertEqual(repeated, None if failure_mode == "native_restore" else True)
                with self.assertRaisesRegex(ValueError, "tensor overlap"):
                    modules.mapping = [{"tensor": norm}]
                    tool.JointEdits(modules, checkpoint, [candidate], params, "cpu")

    def test_native_plan_reuses_exact_loader_and_rejects_frame_or_companion_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            norm = "model.language_model.norm.weight"
            matrix = "model.language_model.layers.0.mlp.up_proj.weight"
            raw = (np.array([1.], np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
            weights(base / "model.safetensors", {norm: ("BF16", (1,), raw), matrix: ("BF16", (1, 1), raw)})
            (base / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {norm: "model.safetensors", matrix: "model.safetensors"}}))
            checkpoint = Checkpoint(base)
            def document(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return {"path": str(path), "sha256": tool.sha(path)}
            plan = {field: document(field + ".json", checkpoint.identity(hash_files=True)
                                    if field == "checkpoint_attestation" else {})
                    for field in ("cases", "frame_evidence", "checkpoint_attestation")}
            plan["checkpoint_index_sha256"] = tool.sha(base / "model.safetensors.index.json")
            report = document("joint-report.json", {"fixed": True})
            spec = {"id": "native", "edits": [{"tensor": norm, "indices": [0], "set_values": [1.5]}],
                    "companion_required": True, "joint_report_path": report["path"],
                    "joint_report_sha256": report["sha256"]}
            spec_ref = document("native.json", spec)
            module_spec = {"base_tree_sha256": "b" * 64, "companion_required": True,
                           "joint_report_path": report["path"], "joint_report_sha256": report["sha256"]}
            native_plan = dict(plan, schema_version=1, architecture="gemma4_native_bf16",
                               base_tree_sha256="b" * 64,
                               candidates=[{"id": "native", "spec": spec_ref["path"], "sha256": spec_ref["sha256"]}])
            def check(value=native_plan, module=module_spec, mapping=None):
                reference = document("native-plan.json", value)
                row = {"native_component": {"plan": reference["path"], "sha256": reference["sha256"]}}
                hashes, states = {}, {}
                result = tool.native_component(row, module, plan, checkpoint,
                                               mapping or [{"tensor": matrix}], hashes, states)
                self.assertIn(report["path"], hashes)
                self.assertIn(spec_ref["path"], states)
                return result
            result = check()
            self.assertEqual(result["spec"]["_verified_metrics"]["delta_l2"], .5)
            with self.assertRaisesRegex(ValueError, "source/frame/cases"):
                check(dict(native_plan, frame_evidence=document("other-frame.json", {})))
            with self.assertRaisesRegex(ValueError, "tensor overlap"):
                check(mapping=[{"tensor": norm}])
            with self.assertRaisesRegex(ValueError, "report mismatch"):
                check(module=dict(module_spec, joint_report_sha256="c" * 64))
            with self.assertRaisesRegex(ValueError, "companion missing"):
                tool.native_component({}, module_spec, plan, checkpoint, [], {}, {})


if __name__ == "__main__":
    unittest.main()
