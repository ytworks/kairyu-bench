"""CPU-only exact RAM merge, late restoration, and public-answer metric checks."""
from __future__ import annotations

import hashlib
import importlib.util
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
            def __call__(self, input_ids, **kwargs):
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
        np.testing.assert_allclose(base["values"]["science"][0],
                                   [a[0][0] - a[1][0] for a in old], atol=1e-7)
        self.assertAlmostEqual(base["correct_nll"][0], -np.mean([a[0][0] for a in old]), places=6)
        self.assertEqual(base["kl"], [0.])
        self.assertTrue(calls)
        with torch.no_grad():
            head.weight[2, 1] += .2
        new = oracle()
        changed = evaluate()
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["model.language_model.layers.0.mlp.up_proj.weight",
                     "model.language_model.layers.1.self_attn.o_proj.weight"]
            values = np.array([[1., -0., -1., .125], [2., -2., .5, -.5], [4., .25, -4., .0625]], np.float32)
            raw = (values.view(np.uint32) >> 16).astype("<u2")
            weights(root / "model.safetensors", {n: ("BF16", (3, 4), raw.tobytes()) for n in names})
            checkpoint = Checkpoint(root)
            params = {n: torch.from_numpy(raw.copy()).view(torch.bfloat16) for n in names}
            mapping = [{"tensor": n, "shape": [3, 4], "lora_A": "A", "lora_B": "B"} for n in names]
            a = np.full((16, 4), 1 / 8, np.float32)
            b = np.full((3, 16), 1 / 16, np.float32)
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
                                            maximum_bytes=raw.size * 2 * 6)
            seen = []
            def evaluate(**flags):
                current = {n: cache.raw(p).copy() for n, p in params.items()}
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
            self.assertEqual(len(calls), 2)  # each projection's GEMM cache is built exactly once
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
            self.assertTrue(cache.restoration["readback_exact"])
            checkpoint.assert_stable()
            with self.assertRaisesRegex(ValueError, "host cache"):
                tool.StreamedModuleEdits(checkpoint, params, mapping, recipes, module, {}, 0, maximum_bytes=1)


if __name__ == "__main__":
    unittest.main()
